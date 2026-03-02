#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fluid cube (no emitter) dropping on car URDF.
# Exports exactly:
#   metadata.json, rollout_seed.pkl, train.npz, valid.npz, test.npz, boundary_points.npy
#
# .npz uses:
#   - key 'gns_data': object array, each item is (positions, particle_type_scalar)
#     positions: (T, N, 3) float32
#     particle_type_scalar: int (这里统一用 0)
#
# 每一帧 positions 已在末尾拼接了静态 URDF 边界点。

import json, pickle
from pathlib import Path
from collections import deque

import numpy as np
import genesis as gs

# ===== Fixed parameters =====
DURATION        = 8.0      # total simulated seconds
FPS             = 60
EXPORT_DIR      = "GNSDataset3D"
SEQ_LEN         = 320      # length T of each trajectory slice
STRIDE          = 320      # stride between slice starts (use =SEQ_LEN for non-overlap)
EXPORT_STRIDE   = 3        # sample every k physics steps
RADIUS_FACTOR   = 2.5
ENABLE_VIEWER   = False
RECORD_VIDEO    = True
VIDEO_FILE      = "cube_drop_on_car.mp4"

# ===== Train/Valid/Test split =====
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # must sum to 1.0

# ===== Boundary particle sampling (from URDF visual mesh) =====
BOUNDARY_VOXEL   = 0.02   # voxel size for downsample
BOUNDARY_MAX_PTS = 2000   # cap the boundary point count


# ---------- helpers ----------

def _as_numpy(x):
    """Robustly convert backend buffers/tensors to numpy array."""
    if x is None:
        return None
    if hasattr(x, "cpu"):
        try:
            x = x.cpu()
        except Exception:
            pass
    if hasattr(x, "numpy"):
        try:
            x = x.numpy()
        except Exception:
            pass
    if hasattr(x, "to_numpy"):
        try:
            x = x.to_numpy()
        except Exception:
            pass
    return np.asarray(x)


def _sample_boundary_pts_from_urdf_entity(urdf_entity):
    """Collect vertices from URDF links' visual/collision meshes and voxel-downsample."""
    arrays = []
    links = getattr(urdf_entity, "links", None) or (
        urdf_entity.get_links() if hasattr(urdf_entity, "get_links") else []
    )
    for ln in links:
        vv = None
        if hasattr(ln, "get_vverts"):
            vv = ln.get_vverts()
        elif hasattr(ln, "get_verts"):
            vv = ln.get_verts()
        vv = _as_numpy(vv)
        if vv is None:
            continue
        vv = vv.reshape(-1, 3).astype(np.float32, copy=False)
        if vv.shape[0] > 0:
            arrays.append(vv)
    if not arrays:
        raise RuntimeError("URDF links provided no vertices (vverts/verts).")

    raw = np.concatenate(arrays, axis=0)  # (M,3)

    # voxel downsample
    coords = np.floor(raw / BOUNDARY_VOXEL).astype(np.int64)
    _, uniq_idx = np.unique(coords, axis=0, return_index=True)
    pts = raw[np.sort(uniq_idx)]
    if pts.shape[0] > BOUNDARY_MAX_PTS:
        sel = np.random.choice(pts.shape[0], BOUNDARY_MAX_PTS, replace=False)
        pts = pts[sel]
    return pts.astype(np.float32)  # (Nb,3)


def _slice_from_deque(frames_deque):
    """deque[(N,3)] -> (T, N_min, 3) float32; align to min N within the window."""
    if len(frames_deque) < SEQ_LEN:
        return None
    min_N = min(f.shape[0] for f in frames_deque)
    if min_N < 2:
        return None
    trimmed = [f[:min_N] for f in frames_deque]
    return np.stack(trimmed, axis=0).astype(np.float32)  # (T, N, 3)


def _stats_bounds_from_slices(slices, dt):
    """Compute velocity/acc stats and xyz bounds from a list of (T,N,3) arrays."""
    if not slices:
        z3 = [0.0, 0.0, 0.0]
        o3 = [1.0, 1.0, 1.0]
        return (
            np.array(z3, np.float32),
            np.array(o3, np.float32),
            np.array(z3, np.float32),
            np.array(o3, np.float32),
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        )

    # big: (sum_T, N, 3)
    big = np.concatenate(slices, axis=0)

    # velocity (time diff)
    if big.shape[0] >= 2:
        V = (big[1:] - big[:-1]) / dt
    else:
        V = np.zeros((0, 1, 3), dtype=np.float32)

    # acceleration (time diff of V)
    if V.shape[0] >= 2:
        A = (V[1:] - V[:-1]) / dt
    else:
        A = np.zeros((0, 1, 3), dtype=np.float32)

    if V.size:
        vel_mean = V.reshape(-1, 3).mean(0)
        vel_std  = V.reshape(-1, 3).std(0) + 1e-12
    else:
        vel_mean = np.zeros(3, np.float32)
        vel_std  = np.ones(3, np.float32)

    if A.size:
        acc_mean = A.reshape(-1, 3).mean(0)
        acc_std  = A.reshape(-1, 3).std(0) + 1e-12
    else:
        acc_mean = np.zeros(3, np.float32)
        acc_std  = np.ones(3, np.float32)

    if big.size:
        flat = big.reshape(-1, 3)
        xyz_min = flat.min(0)
        xyz_max = flat.max(0)
    else:
        xyz_min = np.zeros(3, np.float32)
        xyz_max = np.ones(3, np.float32)

    bounds = [[float(xyz_min[i]), float(xyz_max[i])] for i in range(3)]
    return vel_mean, vel_std, acc_mean, acc_std, bounds


def _save_split_npz(path: Path, trajectories_list):
    """
    写成 GNS 期望的格式：
      - key 'gns_data': object array, 每个元素是 (positions, particle_type_scalar)
        positions: (T,N,3) float32
        particle_type_scalar: int
    """
    n = len(trajectories_list)
    gns_data = np.empty(n, dtype=object)
    for i, arr in enumerate(trajectories_list):
        positions = arr.astype(np.float32, copy=False)
        particle_type_scalar = 0  # 统一类型 ID（全部记作 0）
        gns_data[i] = (positions, particle_type_scalar)

    np.savez(path, gns_data=gns_data)


def _split_dataset(trajs, ratios):
    n = len(trajs)
    r_train, r_valid, _r_test = ratios
    n_train = int(round(n * r_train))
    n_valid = int(round(n * r_valid))
    n_train = min(n_train, n)
    n_valid = min(n_valid, max(0, n - n_train))
    n_test  = max(0, n - n_train - n_valid)
    return (
        trajs[:n_train],
        trajs[n_train:n_train + n_valid],
        trajs[n_train + n_valid:n_train + n_valid + n_test],
    )


# ---------- main ----------

def main():
    gs.init(seed=0, precision="32", logging_level="debug")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=100),
        sph_options=gs.options.SPHOptions(particle_size=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35, max_FPS=120,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=ENABLE_VIEWER,
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Car URDF (fixed) -- real collision geometry
    car_ent = scene.add_entity(
        morph=gs.morphs.URDF(
            file="/home/cui_wei/Autoware/ros/src/vehicle/vehicle_description/urdf/lexus.urdf",
            pos=(0.5, 1.75, 0.25),
            fixed=True, collision=True, convexify=True, scale=0.35,
            merge_fixed_links=True,
        ),
        surface=gs.surfaces.Collision(),
    )

    # Fluid cube (no emitter)
    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
        morph=gs.morphs.Box(
            pos  =(0.5, 1.75, 0.25 + 1.2),
            size =(0.30, 0.30, 0.30),
        ),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode='particle'),
    )

    cam = scene.add_camera(
        res=(1280, 720), pos=(5.5, 6.5, 3.2), lookat=(0.5, 1.5, 1.5), fov=35, GUI=False,
    )

    scene.build(n_envs=1)

    # Sample static boundary points from URDF links (not added to scene)
    boundary_pts = _sample_boundary_pts_from_urdf_entity(car_ent)  # (Nb,3)

    horizon = int(DURATION / scene.sim.options.dt)
    if RECORD_VIDEO:
        cam.start_recording()

    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    # buffers
    frames_deque = deque(maxlen=SEQ_LEN)  # each item: (N_total,3) = [fluid | boundary]
    collected = []  # list of (T,N,3) arrays
    step_count_for_stride = 0

    sph_ent = liquid

    # main loop
    for i in range(horizon):
        scene.step()

        # downsample physics steps
        if (i % EXPORT_STRIDE) == 0:
            if sph_ent.n_particles > 0:
                pos_buf = np.zeros((1, sph_ent.n_particles, 3), dtype=np.float32)
                vel_buf = np.zeros_like(pos_buf)
                sph_ent.get_frame(sph_ent.sim.cur_substep_local, pos_buf, vel_buf)

                pos_fluid = pos_buf[0]  # (Nf,3)
                # concat static boundary points at the tail
                pos_concat = np.concatenate([pos_fluid, boundary_pts], axis=0)  # (Nf+Nb,3)
                frames_deque.append(pos_concat)

                step_count_for_stride += 1
                if len(frames_deque) == SEQ_LEN and (
                    step_count_for_stride >= (STRIDE // max(1, EXPORT_STRIDE))
                ):
                    sl = _slice_from_deque(frames_deque)  # (T,N,3)
                    if sl is not None:
                        collected.append(sl)
                    # prepare for next slice with non-overlap
                    frames_deque.clear()
                    step_count_for_stride = 0

        if RECORD_VIDEO:
            cam.render()

    if RECORD_VIDEO:
        cam.stop_recording(save_to_filename=VIDEO_FILE, fps=FPS)

    # metadata & splits
    dt_effective = float(scene.sim.options.dt * EXPORT_STRIDE)
    try:
        particle_size = float(scene.sph.options.particle_size)
    except Exception:
        particle_size = 0.01
    default_radius = float(RADIUS_FACTOR * particle_size)

    # stats for normalization & bounds
    vel_mean, vel_std, acc_mean, acc_std, bounds = _stats_bounds_from_slices(
        collected, dt_effective
    )
    metadata = {
        "bounds": bounds,
        "sequence_length": int(SEQ_LEN),
        "default_connectivity_radius": float(default_radius),
        "dim": 3,
        "dt": float(dt_effective),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "acc_mean": acc_mean.tolist(),
        "acc_std": acc_std.tolist(),
    }
    with open(export_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # rollout seed：首个 trajectory 的首帧 positions
    if collected:
        seed = {
            "positions": collected[0][0].astype(np.float32),  # (N,3) at t0
            "dt": float(dt_effective),
            "default_connectivity_radius": float(default_radius),
        }
        with open(export_path / "rollout_seed.pkl", "wb") as f:
            pickle.dump(seed, f)

    # split -> train/valid/test
    train_list, valid_list, test_list = _split_dataset(collected, SPLIT_RATIOS)

    _save_split_npz(export_path / "train.npz", train_list)
    _save_split_npz(export_path / "valid.npz", valid_list)
    _save_split_npz(export_path / "test.npz",  test_list)

    # 额外保留边界点云（GNS 本身不用）
    np.save(export_path / "boundary_points.npy", boundary_pts.astype(np.float32))

    print(f"[DONE] Wrote to {export_path}:")
    print(" - metadata.json")
    print(" - rollout_seed.pkl")
    print(f" - train.npz ({len(train_list)} samples)")
    print(f" - valid.npz ({len(valid_list)} samples)")
    print(f" - test.npz  ({len(test_list)} samples)")
    print(" - boundary_points.npy")
    print(f"(Under key 'gns_data': each entry is (T={SEQ_LEN}, N, 3) + particle_type_scalar=0)")


if __name__ == "__main__":
    main()
