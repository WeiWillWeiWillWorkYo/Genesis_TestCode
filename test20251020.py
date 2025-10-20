#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fluid cube (no emitter) dropping on car URDF.
# Exports exactly:
#   metadata.json, rollout_seed.pkl, train.npz, valid.npz, test.npz
# .npz uses key 'trajectories' where each item is a (T, N, 3) float32 array.
# No 'shards/' directory, no tuples inside trajectories.

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

# ---------- helpers ----------
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
    """Compute velocity/acc stats and xyz bounds."""
    if not slices:
        z3 = [0.0, 0.0, 0.0]; o3 = [1.0, 1.0, 1.0]
        return np.array(z3), np.array(o3), np.array(z3), np.array(o3), [[0.0,1.0],[0.0,1.0],[0.0,1.0]]

    big = np.concatenate(slices, axis=0)  # (sum_T, N, 3)
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

def _save_split_npz(path: Path, trajectories_list):
    """Write .npz with key 'trajectories' (object array of (T,N,3) arrays)."""
    obj = np.empty(len(trajectories_list), dtype=object)
    for i, arr in enumerate(trajectories_list):
        obj[i] = arr.astype(np.float32, copy=False)
    np.savez(path, trajectories=obj)

def _split_dataset(trajs, ratios):
    n = len(trajs)
    r_train, r_valid, _r_test = ratios
    n_train = int(round(n * r_train))
    n_valid = int(round(n * r_valid))
    n_train = min(n_train, n)
    n_valid = min(n_valid, max(0, n - n_train))
    n_test  = max(0, n - n_train - n_valid)
    return trajs[:n_train], trajs[n_train:n_train+n_valid], trajs[n_train+n_valid:n_train+n_valid+n_test]

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

    # Car URDF (fixed)
    scene.add_entity(
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
    horizon = int(DURATION / scene.sim.options.dt)
    if RECORD_VIDEO:
        cam.start_recording()

    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    # buffers
    n_envs = 1
    frames_deque = deque(maxlen=SEQ_LEN)
    collected = []  # list of (T,N,3) arrays
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
                frames_deque.append(pos_buf[0])  # (N,3)

                step_count_for_stride += 1
                if len(frames_deque) == SEQ_LEN and (step_count_for_stride >= (STRIDE // max(1, EXPORT_STRIDE))):
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
    vel_mean, vel_std, acc_mean, acc_std, bounds = _stats_bounds_from_slices(collected, dt_effective)
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

    # rollout seed (store the first sequence's first frame as a minimal seed)
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

    print(f"[DONE] Wrote to {export_path}:")
    print(" - metadata.json")
    print(" - rollout_seed.pkl")
    print(f" - train.npz ({len(train_list)} samples)")
    print(f" - valid.npz ({len(valid_list)} samples)")
    print(f" - test.npz  ({len(test_list)} samples)")
    print(f"(All trajectories are (T={SEQ_LEN}, N, 3) float32 under key 'trajectories')")

if __name__ == "__main__":
    main()
