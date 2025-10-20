#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Full script: fluid cube (no emitter), car URDF only, single env export.
# English comments only, minimal changes from your working pipeline.

import json
from pathlib import Path
from collections import deque
import numpy as np
import genesis as gs

# ===== Fixed parameters =====
DURATION        = 8.0           # seconds to simulate
FPS             = 60
EXPORT_DIR      = "GNSDataset3D"
SEQ_LEN         = 320           # GNS sequence length
STRIDE          = 320           # non-overlapping slices
EXPORT_STRIDE   = 3             # sample every k steps (increase to reduce I/O)
RADIUS_FACTOR   = 2.5           # connectivity_radius = factor * particle_size
ENABLE_VIEWER   = False
RECORD_VIDEO    = True          # enable video to inspect the scene
VIDEO_FILE      = "cube_drop_on_car.mp4"

# ---------- helpers ----------
def _slice_from_deque(frames_deque):
    """deque[ (N,3) ] -> (T, N_min, 3); align to min N within the window."""
    if len(frames_deque) < SEQ_LEN:
        return None
    min_N = min(f.shape[0] for f in frames_deque)
    if min_N < 2:
        return None
    trimmed = [f[:min_N] for f in frames_deque]
    return np.stack(trimmed, axis=0).astype(np.float32)  # (T, N, 3)

def _stats_bounds_from_slices(slices, dt):
    """Compute stats on velocity/acceleration and xyz bounds based on slices.
       Assumes constant N across slices once export begins (true for fluid cube)."""
    if not slices:
        z3 = [0.0, 0.0, 0.0]
        o3 = [1.0, 1.0, 1.0]
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

def _merge_shards_write_pair(export_dir, dt, default_radius):
    """Read shards/ and write metadata.json + train.npz with (positions, particle_type) tuples."""
    export_path = Path(export_dir)
    shard_files = sorted((export_path / "shards").glob("traj_*.npz"))
    if not shard_files:
        raise RuntimeError("No slices found under shards/. Increase DURATION or lower SEQ_LEN / increase EXPORT_STRIDE.")

    # load slices
    slices = []
    for sf in shard_files:
        with np.load(sf) as z:
            slices.append(z["positions"])  # (T,N,3)

    # stats + metadata
    vel_mean, vel_std, acc_mean, acc_std, bounds = _stats_bounds_from_slices(slices, dt)
    metadata = {
        "bounds": bounds,
        "sequence_length": int(SEQ_LEN),
        "default_connectivity_radius": float(default_radius),
        "dim": 3,
        "dt": float(dt),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "acc_mean": acc_mean.tolist(),
        "acc_std": acc_std.tolist(),
    }
    with open(export_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # (positions, particle_type) tuples
    pairs = []
    for sl in slices:
        N = sl.shape[1]
        ptype = np.zeros((N,), dtype=np.int32)   # all fluid for now
        pairs.append((sl.astype(np.float32), ptype))

    obj = np.empty(len(pairs), dtype=object)
    for i, tup in enumerate(pairs):
        obj[i] = tup
    np.savez(export_path / "train.npz", trajectories=obj)
    print(f"[MERGE] wrote {export_path/'train.npz'} with {len(pairs)} samples.")
    print(f"[MERGE] wrote {export_path/'metadata.json'}")

def _get_n_envs(scene):
    return getattr(scene, "n_envs", getattr(scene.sim, "_B", 1))

# ---------- main ----------
def main():
    gs.init(seed=0, precision="32", logging_level="debug")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        # Use official particle_size=0.01 for stability and predictable neighbor radius
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

    # Car URDF only (wheel removed)
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="/home/cui_wei/Autoware/ros/src/vehicle/vehicle_description/urdf/lexus.urdf",
            pos=(0.5, 1.75, 0.25),
            fixed=True, collision=True, convexify=True, scale=0.35,
            merge_fixed_links=True,
        ),
        surface=gs.surfaces.Collision(),
    )

    # Fluid cube initialized directly (no emitter). Smaller size + lower drop height to reduce N and initial impact.
    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
        morph=gs.morphs.Box(
            pos  =(0.5, 1.75, 0.25 + 1.2),   # directly above car
            size =(0.30, 0.30, 0.30),        # smaller cube => fewer particles
        ),
        surface=gs.surfaces.Default(
            color    =(0.4, 0.8, 1.0),
            vis_mode ='particle',
        ),
    )

    # Camera for recording
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(5.5, 6.5, 3.2),
        lookat=(0.5, 1.5, 1.5),
        fov=35,
        GUI=False,
    )

    # Single environment first (keeps memory/I-O manageable)
    scene.build(n_envs=1)
    horizon = int(DURATION / scene.sim.options.dt)
    if RECORD_VIDEO:
        cam.start_recording()

    # Export folders
    export_path = Path(EXPORT_DIR)
    shards_path = export_path / "shards"
    shards_path.mkdir(parents=True, exist_ok=True)

    # Single env buffers
    n_envs = 1
    frames_deque = deque(maxlen=SEQ_LEN)
    stride_frames = max(1, STRIDE // max(1, EXPORT_STRIDE))

    # Use the fluid entity handle directly
    sph_ent = liquid

    # Main loop
    for i in range(horizon):
        scene.step()

        # Sampling (downsample by EXPORT_STRIDE)
        if (i % EXPORT_STRIDE) == 0:
            n_particles = sph_ent.n_particles
            if n_particles > 0:
                pos_buf = np.zeros((n_envs, n_particles, 3), dtype=np.float32)
                vel_buf = np.zeros_like(pos_buf)
                sph_ent.get_frame(sph_ent.sim.cur_substep_local, pos_buf, vel_buf)

                frames_deque.append(pos_buf[0])

                # When window is full and stride aligns, write one shard
                if len(frames_deque) == SEQ_LEN and ((i // EXPORT_STRIDE) % stride_frames == 0):
                    sl = _slice_from_deque(frames_deque)
                    if sl is not None:
                        shard_id = len(list(shards_path.glob("traj_*.npz")))
                        np.savez_compressed(shards_path / f"traj_{shard_id:06d}.npz", positions=sl)

        if RECORD_VIDEO:
            cam.render()

    if RECORD_VIDEO:
        cam.stop_recording(save_to_filename=VIDEO_FILE, fps=FPS)

    # Merge -> train.npz + metadata.json
    dt_effective = float(scene.sim.options.dt * EXPORT_STRIDE)
    try:
        particle_size = float(scene.sph.options.particle_size)
    except Exception:
        particle_size = 0.01
    default_radius = float(RADIUS_FACTOR * particle_size)

    _merge_shards_write_pair(EXPORT_DIR, dt_effective, default_radius)
    print(f"[DONE] Dataset ready at {EXPORT_DIR}/ (train.npz + metadata.json)")

if __name__ == "__main__":
    main()
