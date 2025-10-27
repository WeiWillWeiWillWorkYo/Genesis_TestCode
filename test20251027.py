#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fluid cube (no emitter) dropping on car URDF.
# Exports exactly:
#   metadata.json, rollout_seed.pkl, train.npz, valid.npz, test.npz, boundary_points.npy
#
# .npz uses key 'trajectories' where each item is a TUPLE:
#   (positions:(T, N, 3) float32, particle_type:(N,) int64)
# positions 已在每帧末尾拼接了静态的 URDF 边界点；particle_type: 流体=0, 边界=1
#
# 统计量仅基于“流体粒子”计算；边界点不参与统计与动力学运算。

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
BOUNDARY_VOXEL   = 0.008   # voxel size for downsample
BOUNDARY_MAX_PTS = 80000   # cap the boundary point count

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
    """Collect vertices from URDF links' visual/collision meshes and voxel-downsample to static boundary points."""
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

def _slice_tuple_from_deques(frames_deque, Nb):
    """From a deque of concatenated frames [(Nf+Nb,3)] build a tuple:
       (positions:(T, Nf_min+Nb, 3), particle_type:(Nf_min+Nb,))
       Align fluid count to the minimum Nf across the window; boundary Nb stays fixed at the tail.
    """
    if len(frames_deque) < SEQ_LEN:
        return None

    # Estimate Nf per frame and find min
    Nf_list = []
    for arr in frames_deque:
        total = arr.shape[0]
        Nf_list.append(total - Nb)
    Nf_min = max(0, min(Nf_list))
    if Nf_min < 1:
        return None

    T = len(frames_deque)
    out = np.empty((T, Nf_min + Nb, 3), dtype=np.float32)
    for t, arr in enumerate(frames_deque):
        out[t, :Nf_min, :] = arr[:Nf_min, :]
        out[t, Nf_min:, :] = arr[-Nb:, :]

    ptype = np.empty((Nf_min + Nb,), dtype=np.int64)
    ptype[:Nf_min] = 0  # fluid
    ptype[Nf_min:] = 1  # boundary
    return (out, ptype)

def _stats_bounds_from_slices_tuples(tuples_list, dt):
    """Compute stats using ONLY fluid points from a list of tuples:
       each item is (positions(T,N,3), particle_type(N,))
       Returns vel_mean, vel_std, acc_mean, acc_std, bounds
    """
    if not tuples_list:
        z3 = [0.0, 0.0, 0.0]; o3 = [1.0, 1.0, 1.0]
        return np.array(z3), np.array(o3), np.array(z3), np.array(o3), [[0.0,1.0],[0.0,1.0],[0.0,1.0]]

    pos_list = []
    for (pos, ptype) in tuples_list:
        fluid_mask = (ptype == 0)
        if fluid_mask.sum() <= 0:
            continue
        pos_list.append(pos[:, fluid_mask, :])  # (T, Nf, 3)

    if not pos_list:
        z3 = [0.0, 0.0, 0.0]; o3 = [1.0, 1.0, 1.0]
        return np.array(z3), np.array(o3), np.array(z3), np.array(o3), [[0.0,1.0],[0.0,1.0],[0.0,1.0]]

    big = np.concatenate(pos_list, axis=0)  # (sum_T, Nf, 3)
    if big.shape[0] >= 2:
        V = (big[1:] - big[:-1]) / dt
    else:
        V = np.zeros((0,1,3), dtype=np.float32)
    if V.shape[0] >= 2:
        A = (V[1:] - V[:-1]) / dt
    else:
        A = np.zeros((0,1,3), dtype=np.float32)

    vel_mean = (V.reshape(-1,3).mean(0) if V.size else np.zeros(3, np.float32))
    vel_std  = (V.reshape(-1,3).std(0)  if V.size else np.ones(3, np.float32)) + 1e-12
    acc_mean = (A.reshape(-1,3).mean(0) if A.size else np.zeros(3, np.float32))
    acc_std  = (A.reshape(-1,3).std(0)  if A.size else np.ones(3, np.float32)) + 1e-12
    xyz_min  = (big.reshape(-1,3).min(0) if big.size else np.zeros(3, np.float32))
    xyz_max  = (big.reshape(-1,3).max(0) if big.size else np.ones(3, np.float32))
    bounds = [[float(xyz_min[i]), float(xyz_max[i])] for i in range(3)]
    return vel_mean, vel_std, acc_mean, acc_std, bounds

def _save_split_npz_tuples(path: Path, tuples_list):
    """Write .npz with key 'trajectories'; each element is a tuple:
       (positions:(T,N,3) float32, particle_type:(N,) int64)
    """
    obj = np.empty(len(tuples_list), dtype=object)
    for i, tup in enumerate(tuples_list):
        positions, ptype = tup
        obj[i] = (positions.astype(np.float32, copy=False),
                  ptype.astype(np.int64,   copy=False))
    np.savez(path, trajectories=obj)

def _split_dataset(items, ratios):
    n = len(items)
    r_train, r_valid, _r_test = ratios
    n_train = int(round(n * r_train))
    n_valid = int(round(n * r_valid))
    n_train = min(n_train, n)
    n_valid = min(n_valid, max(0, n - n_train))
    n_test  = max(0, n - n_train - n_valid)
    return items[:n_train], items[n_train:n_train+n_valid], items[n_train+n_valid:n_train+n_valid+n_test]

def _get_n_envs(scene):
    return getattr(scene, "n_envs", getattr(scene.sim, "_B", 1))

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

    # Car URDF (fixed) -- use as real collision geometry
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
    Nb = int(boundary_pts.shape[0])

    horizon = int(DURATION / scene.sim.options.dt)
    if RECORD_VIDEO:
        cam.start_recording()

    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    # buffers
    n_envs = 1
    frames_deque = deque(maxlen=SEQ_LEN)  # will store concatenated frames: [fluid(Nf,3) | boundary(Nb,3)]
    collected = []  # list of tuples: (positions(T,N,3), particle_type(N,))
    step_count_for_stride = 0

    sph_ent = liquid

    # main loop
    for i in range(horizon):
        scene.step()

        # downsample physics steps
        if (i % EXPORT_STRIDE) == 0:
            if sph_ent.n_particles > 0:
                pos_buf = np.zeros((n_envs, sph_ent.n_particles, 3), dtype=np.float32)
                vel_buf = np.zeros_like(pos_buf)
                sph_ent.get_frame(sph_ent.sim.cur_substep_local, pos_buf, vel_buf)

                pos_fluid = pos_buf[0]  # (Nf,3)
                # concat static boundary points at the tail
                pos_concat = np.concatenate([pos_fluid, boundary_pts], axis=0)  # (Nf+Nb,3)
                frames_deque.append(pos_concat)

                step_count_for_stride += 1
                if len(frames_deque) == SEQ_LEN and (step_count_for_stride >= (STRIDE // max(1, EXPORT_STRIDE))):
                    tup = _slice_tuple_from_deques(frames_deque, Nb)  # -> (positions(T,N,3), particle_type(N,))
                    if tup is not None:
                        collected.append(tup)
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

    # stats for normalization & bounds (fluid only)
    vel_mean, vel_std, acc_mean, acc_std, bounds = _stats_bounds_from_slices_tuples(collected, dt_effective)

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
        # extra notes
        "particle_types": {"fluid": 0, "urdf_boundary": 1},
        "concat_boundary": True,
        "boundary_points_file": "boundary_points.npy",
        "boundary_voxel_size": float(BOUNDARY_VOXEL),
        "boundary_point_count": int(Nb),
    }
    with open(export_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # rollout seed (minimal seed: positions at t0 + particle_type)
    if collected:
        seed_positions, seed_ptype = collected[0]  # (T,N,3), (N,)
        seed = {
            "positions": seed_positions[0].astype(np.float32),  # (N,3) at t0
            "particle_type": seed_ptype.astype(np.int64),
            "dt": float(dt_effective),
            "default_connectivity_radius": float(default_radius),
        }
        with open(export_path / "rollout_seed.pkl", "wb") as f:
            pickle.dump(seed, f)

    # split -> train/valid/test
    train_list, valid_list, test_list = _split_dataset(collected, SPLIT_RATIOS)

    _save_split_npz_tuples(export_path / "train.npz", train_list)
    _save_split_npz_tuples(export_path / "valid.npz", valid_list)
    _save_split_npz_tuples(export_path / "test.npz",  test_list)

    # also save the static boundary point cloud (for downstream optional use)
    np.save(export_path / "boundary_points.npy", boundary_pts.astype(np.float32))

    print(f"[DONE] Wrote to {export_path}:")
    print(" - metadata.json")
    print(" - rollout_seed.pkl")
    print(f" - train.npz ({len(train_list)} samples)")
    print(f" - valid.npz ({len(valid_list)} samples)")
    print(f" - test.npz  ({len(test_list)} samples)")
    print(" - boundary_points.npy")
    print(f"(Each trajectory is a tuple (positions(T={SEQ_LEN}, N, 3), particle_type(N,))).")

if __name__ == "__main__":
    main()
