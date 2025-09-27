#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import deque
import numpy as np
import genesis as gs

# ===== 固定参数（点三角直接跑） =====
DURATION        = 8.0          # 模拟秒数
FPS             = 60
EXPORT_DIR      = "GNSDataset3D"
SEQ_LEN         = 320          # 与 GNS 一致
STRIDE          = 320          # 不重叠切片
EXPORT_STRIDE   = 3            # 每步都采（你要更轻 I/O 可调大）
RADIUS_FACTOR   = 2.5          # default_connectivity_radius = factor * particle_size
ENABLE_VIEWER   = False        # 降低卡顿
RECORD_VIDEO    = False        # 需要视频再开
VIDEO_FILE      = "rain_on_wheel.mp4"

# ---------- 工具 ----------
def _slice_from_deque(frames_deque):
    """deque -> (T, N, 3)。窗口内对齐到 min_N；不够长返回 None"""
    if len(frames_deque) < SEQ_LEN:
        return None
    min_N = min(f.shape[0] for f in frames_deque)
    if min_N < 2:
        return None
    trimmed = [f[:min_N] for f in frames_deque]
    return np.stack(trimmed, axis=0).astype(np.float32)  # (T, N, 3)

def _stats_bounds_from_slices(slices, dt):
    """基于切片（而非全帧）计算 vel/acc 统计与 bounds"""
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
    """读取 shards/，写 metadata.json + train.npz（**二元组**格式）"""
    export_path = Path(export_dir)
    shard_files = sorted((export_path / "shards").glob("traj_*.npz"))
    if not shard_files:
        raise RuntimeError("没有任何切片（shards 为空）。请增大 DURATION 或减小 SEQ_LEN/增大 EXPORT_STRIDE。")

    # 读取所有切片
    slices = []
    for sf in shard_files:
        with np.load(sf) as z:
            slices.append(z["positions"])  # (T,N,3)

    # 统计 + 元数据
    vel_mean, vel_std, acc_mean, acc_std, bounds = _stats_bounds_from_slices(slices, dt)
    metadata = {
        "bounds": bounds,
        "sequence_length": int(SEQ_LEN),
        "default_connectivity_radius": float(default_radius),
        "dim": 3,                      # 我们是 3D
        "dt": float(dt),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "acc_mean": acc_mean.tolist(),
        "acc_std": acc_std.tolist(),
    }
    with open(export_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # **二元组** (positions, particle_type) —— 顶层直接是 Python tuple
    pairs = []
    for sl in slices:
        N = sl.shape[1]
        ptype = np.zeros((N,), dtype=np.int32)   # 全部流体，后续要加静态边界再扩展
        pairs.append((sl.astype(np.float32), ptype))

    obj = np.empty(len(pairs), dtype=object)
    for i, tup in enumerate(pairs):
        obj[i] = tup

    np.savez(export_path / "train.npz", trajectories=obj)
    print(f"[MERGE] wrote {export_path/'train.npz'} with {len(pairs)} samples.")
    print(f"[MERGE] wrote {export_path/'metadata.json'}")

def _get_n_envs(scene):
    return getattr(scene, "n_envs", getattr(scene.sim, "_B", 1))

# ---------- 主程序 ----------
def main():
    gs.init(seed=0, precision="32", logging_level="debug")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        sph_options=gs.options.SPHOptions(particle_size=0.02),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35, max_FPS=120,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=ENABLE_VIEWER,
    )

    # 几何
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 1.6),
            fixed=True, collision=True, convexify=False, scale=1.0,
        ),
        surface=gs.surfaces.Collision(),
    )
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="/home/wei/Autoware/ros/src/vehicle/vehicle_description/urdf/lexus.urdf",
            pos=(0.5, 1.75, 0.25),
            fixed=True, collision=True, convexify=True, scale=0.35,
            merge_fixed_links=True,
        ),
        surface=gs.surfaces.Collision(),
    )

    # 流体发射器
    emitter = scene.add_emitter(
        material=gs.materials.SPH.Liquid(sampler="regular"),
        max_particles=100000,
        surface=gs.surfaces.Glass(color=(0.7, 0.85, 1.0, 0.7), vis_mode="particle"),
    )

    # 相机（可关）
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(5.5, 6.5, 3.2),
        lookat=(0.5, 1.5, 1.5),
        fov=35,
        GUI=False,
    )

    scene.build(n_envs=2)
    horizon = int(DURATION / scene.sim.options.dt)
    if RECORD_VIDEO: cam.start_recording()

    # 导出目录
    export_path = Path(EXPORT_DIR)
    shards_path = export_path / "shards"
    shards_path.mkdir(parents=True, exist_ok=True)

    # 每个 env 一个滚动窗口
    n_envs = _get_n_envs(scene)
    frames_deques = [deque(maxlen=SEQ_LEN) for _ in range(n_envs)]
    stride_frames = max(1, STRIDE // max(1, EXPORT_STRIDE))

    # 找到 SPH 实体（使用官方 get_frame）
    sph_ent = None
    for ent in scene.entities:
        if hasattr(ent, "get_frame") and hasattr(ent, "n_particles"):
            sph_ent = ent
            break
    if sph_ent is None:
        raise RuntimeError("未找到 SPHEntity（需要 get_frame & n_particles）")

    # 主循环
    for i in range(horizon):
        emitter.emit(
            pos=np.array([0.5, 1.0, 3.5]),
            direction=np.array([0.0, 0.0, -1.0]),
            speed=5.0,
            droplet_shape="circle",
            droplet_size=0.22,
        )
        scene.step()

        # 采样（降 I/O 可调 EXPORT_STRIDE）
        if (i % EXPORT_STRIDE) == 0:
            n_particles = sph_ent.n_particles
            pos_buf = np.zeros((n_envs, n_particles, 3), dtype=np.float32)
            vel_buf = np.zeros_like(pos_buf)
            sph_ent.get_frame(sph_ent.sim.cur_substep_local, pos_buf, vel_buf)

            for e in range(n_envs):
                frames_deques[e].append(pos_buf[e])  # (N_t,3)
                # 窗口满 & 对齐步长：立刻落盘一片，避免堆内存
                if len(frames_deques[e]) == SEQ_LEN and ((i // EXPORT_STRIDE) % stride_frames == 0):
                    sl = _slice_from_deque(frames_deques[e])
                    if sl is not None:
                        shard_id = len(list(shards_path.glob("traj_*.npz")))
                        np.savez_compressed(shards_path / f"traj_{shard_id:06d}.npz", positions=sl)

        if RECORD_VIDEO: cam.render()

    if RECORD_VIDEO:
        cam.stop_recording(save_to_filename=VIDEO_FILE, fps=FPS)

    # 合并 -> train.npz（二元组） + metadata.json
    dt_effective = float(scene.sim.options.dt * EXPORT_STRIDE)
    try:
        particle_size = float(scene.sph.options.particle_size)
    except Exception:
        particle_size = 0.02
    default_radius = float(RADIUS_FACTOR * particle_size)

    _merge_shards_write_pair(EXPORT_DIR, dt_effective, default_radius)
    print(f"[DONE] Dataset ready at {EXPORT_DIR}/ (train.npz + metadata.json)")

if __name__ == "__main__":
    main()
