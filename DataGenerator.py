#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis -> GNS exporter with wheel + external URDF car (Lexus)
- Creates a scene with a wheel URDF (local), an external Lexus URDF (car body),
  an SPH liquid entity (sphere droplet) and an emitter (rain).
- Samples the Lexus mesh to create static boundary points for GNS.
- Exports train/valid/test .npz compatible with GNS:
    npz contains `data` key: array of items (positions, particle_types, n_particles)
  where positions shape: (T, N_total, 3), particle_types dtype=int64,
  and boundary particles use type 3 (kinematic) as GNS expects.
- Writes metadata.json with dt, sequence_length, bounds, connectivity radius, and statistics.

This version has **no argparse/CLI dependency**. Edit `get_args()` for defaults or
call `main(args=make_args(...))` from another Python script to override.
"""

import os
import json
import math
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import random

from types import SimpleNamespace as _NS

# trimesh optional but required for mesh sampling
try:
    import trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False


# ============================ helper functions ============================

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def resolve_package_uri(uri: str, pkg_root_hint: str | None = None) -> str:
    """Resolve package://... URIs to filesystem paths. Try common Autoware locations as fallback."""
    if not isinstance(uri, str):
        return uri
    if uri.startswith("package://"):
        rel = uri[len("package://"):]
        candidates = [pkg_root_hint] if pkg_root_hint else [
            "/home/cui_wei/Autoware/ros/src",
            "/home/cui_wei/Autoware/src",
            "/home/cui_wei/Autoware",
            os.getcwd(),
        ]
        for root in candidates:
            if not root:
                continue
            cand = os.path.join(root, rel)
            if os.path.exists(cand):
                return cand
        # try raw relative path
        if os.path.exists(rel):
            return os.path.abspath(rel)
    return uri

def get_first_mesh_from_urdf(urdf_path: str, prefer_tag="collision", pkg_root_hint=None):
    """Parse URDF and return first mesh filename + scale + origin found in <collision> or <visual>."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    # handle optional namespaces: strip if present
    def strip(ns_name):
        return ns_name.split('}')[-1] if '}' in ns_name else ns_name
    for link in root.findall(".//"):
        if strip(link.tag) != "link":
            continue
        node = link.find(prefer_tag) or link.find("visual")
        if node is None:
            continue
        geom = node.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        fname = mesh.attrib.get("filename", "")
        scale_s = mesh.attrib.get("scale", "1 1 1")
        scale = tuple(float(x) for x in scale_s.split())
        origin = node.find("origin")
        if origin is not None:
            xyz = tuple(float(x) for x in origin.attrib.get("xyz", "0 0 0").split())
            rpy = tuple(float(x) for x in origin.attrib.get("rpy", "0 0 0").split())
        else:
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
        fname = resolve_package_uri(fname, pkg_root_hint)
        return {"file": fname, "scale": scale, "xyz": xyz, "rpy": rpy}
    raise FileNotFoundError(f"No mesh found in URDF: {urdf_path}")

def sample_surface_even(mesh_path: str, n_points: int, scale=(1.0,1.0,1.0), translate=(0.0,0.0,0.0)):
    """Use trimesh to sample the mesh surface evenly. Returns (n_points,3) float32."""
    if not _HAS_TRIMESH:
        raise RuntimeError("trimesh is required for mesh sampling. Install with `pip install trimesh`.")
    mesh = trimesh.load(mesh_path, force='mesh')
    # apply scale and translation
    if isinstance(scale, (float, int)):
        mesh.apply_scale(float(scale))
    else:
        mesh.apply_scale(scale)
    mesh.apply_translation(translate)
    pts, _ = trimesh.sample.sample_surface_even(mesh, n_points)
    return np.asarray(pts, dtype=np.float32)

def to_Nx3(a):
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 3:
        return a
    if a.ndim == 3 and a.shape[-1] == 3:
        if a.shape[0] == 1:
            return a[0]
        return a.reshape(-1,3)
    raise RuntimeError(f"Unexpected particle shape {a.shape}")


# ============================ export helpers ============================

def pack_and_save_gns(episodes, out_dir, T_target=None, dt=0.004, connectivity_radius=0.02):
    """
    episodes: list of dict with keys:
      'pos' : np.array (T, Nfluid, 3)
      'B'   : np.array (Nb, 3)
    Exports:
      out_dir/train.npz  (data key: array of (positions, particle_types, N_total))
      and metadata.json
    """
    ensure_dir(out_dir)
    # clamp lengths to T_target if provided; otherwise to shortest
    if T_target is None:
        T_target = min(ep["pos"].shape[0] for ep in episodes)
    samples = []
    for ep in episodes:
        posF = ep["pos"][:T_target].astype(np.float32)  # (T,Nf,3)
        B = ep["B"].astype(np.float32)                 # (Nb,3)
        T, Nf, _ = posF.shape
        Nb = B.shape[0]
        posB = np.broadcast_to(B[None,...], (T, Nb, 3))
        pos_all = np.concatenate([posF, posB], axis=1)  # (T, Nf+Nb, 3)
        N_total = Nf + Nb
        ptypes = np.zeros((N_total,), dtype=np.int64)
        ptypes[Nf:] = 3  # kinematic boundary id used by GNS
        samples.append((pos_all, ptypes, int(N_total)))
    # save as data key to match your gns loader
    np.savez_compressed(os.path.join(out_dir, "train.npz"), data=np.array(samples, dtype=object))
    # also duplicate for valid/test as placeholders
    for split in ["valid", "test"]:
        dst = os.path.join(out_dir, f"{split}.npz")
        if os.path.exists(dst):
            os.remove(dst)
        # portable copy without shell
        with open(os.path.join(out_dir, "train.npz"), "rb") as src, open(dst, "wb") as dstf:
            dstf.write(src.read())
    # metadata
    # compute stats across episodes
    vel_means, vel_stds, acc_means, acc_stds = [], [], [], []
    for ep in episodes:
        posF = ep["pos"][:T_target].astype(np.float32)
        if posF.shape[0] >= 2:
            v = (posF[1:] - posF[:-1]) / dt
            vel_means.append(v.reshape(-1,3).mean(axis=0))
            vel_stds.append(v.reshape(-1,3).std(axis=0))
        if posF.shape[0] >= 3:
            a = ( (posF[2:] - posF[1:]) / dt - (posF[1:] - posF[:-1]) / dt ) / dt
            acc_means.append(a.reshape(-1,3).mean(axis=0))
            acc_stds.append(a.reshape(-1,3).std(axis=0))
    if vel_means:
        vm = np.mean(np.stack(vel_means), axis=0).tolist()
        vs = np.mean(np.stack(vel_stds), axis=0).tolist()
    else:
        vm = [0.0,0.0,0.0]; vs = [1.0,1.0,1.0]
    if acc_means:
        am = np.mean(np.stack(acc_means), axis=0).tolist()
        asd = np.mean(np.stack(acc_stds), axis=0).tolist()
    else:
        am=[0.0,0.0,0.0]; asd=[1.0,1.0,1.0]
    # bounds = envelope of particles:
    all_positions = np.concatenate([ep["pos"][:T_target].reshape(-1,3) for ep in episodes], axis=0)
    mins = all_positions.min(axis=0).tolist()
    maxs = all_positions.max(axis=0).tolist()
    meta = {
        "bounds": [[float(mins[0]), float(maxs[0])],
                   [float(mins[1]), float(maxs[1])],
                   [float(mins[2]), float(maxs[2])]],
        "sequence_length": int(T_target),
        "default_connectivity_radius": float(connectivity_radius),
        "dim": 3,
        "dt": float(dt),
        "vel_mean": vm,
        "vel_std": vs,
        "acc_mean": am,
        "acc_std": asd
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[GNS export] written train/valid/test npz and metadata.json to {out_dir}")


# ============================ defaults / main ============================

def get_args():
    """Hard-coded defaults. Edit here if you want different behavior by default."""
    return _NS(
        # 内置 Lexus URDF 路径（可按你机器实际路径修改）
        lexus_urdf="/home/cui_wei/Autoware/ros/src/lexus_description/urdf/lexus.urdf",
        lexus_scale=0.35,

        # 本地小轮子 URDF（如果不存在也无妨，可改为你有的 URDF）
        wheel_urdf="urdf/wheel/fancy_wheel.urdf",

        # 模拟设置
        frames=None,            # 如果不为 None，则优先使用固定步数
        duration=8.0,           # seconds, 当 frames=None 时，用 dt 转换为步数
        dt=4e-3,
        substeps=10,
        particle_size=0.02,
        boundary_points=2000,

        # 输出
        dataset_dir="./dataset_gns",
        no_video=False,
        video="rain_on_car.mp4",
        fps=60,

        # 其它
        n_episodes=1,
        seed=0,
        pkg_root_hint="/home/cui_wei/Autoware/ros/src",
    )

def make_args(**overrides):
    """Programmatic override helper: main(args=make_args(...))"""
    args = get_args()
    for k, v in overrides.items():
        setattr(args, k, v)
    return args

def main(args=None):
    if args is None:
        args = get_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # import genesis and init
    import genesis as gs
    gs.init(seed=0, precision="32", logging_level="debug")

    # compute steps
    dt = args.dt
    if args.frames is not None:
        steps = int(args.frames)
    else:
        steps = int(max(1, args.duration / dt))

    episodes = []
    for epi_i in range(args.n_episodes):
        # Build scene per-episode to avoid field contamination
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=args.substeps),
            sph_options=gs.options.SPHOptions(particle_size=args.particle_size),
            mpm_options=gs.options.MPMOptions(lower_bound=(0.0, -1.5, 0.0), upper_bound=(1.0, 1.5, 4.0)),
            viewer_options=gs.options.ViewerOptions(camera_pos=(5.5,6.5,3.2),
                                                    camera_lookat=(0.5,1.5,1.5),
                                                    camera_fov=35, max_FPS=120),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
        )

        # add simple ground plane
        scene.add_entity(gs.morphs.Plane())

        # wheel (as collision rigid body)
        try:
            scene.add_entity(
                morph=gs.morphs.URDF(
                    file=args.wheel_urdf,
                    pos=(0.5, 0.25, 1.6),
                    euler=(0,0,0),
                    fixed=True,
                    collision=True,
                    convexify=False,
                    scale=1.0
                ),
                surface=gs.surfaces.Collision()
            )
        except Exception as e:
            print(f"[WARN] wheel URDF skipped: {e}")

        # lexus URDF (external)
        lexus_pos = (0.5, 1.75, 0.25)
        lexus_euler = (0.0, 0.0, 0.0)
        lexus_loaded = False
        if args.lexus_urdf and os.path.exists(args.lexus_urdf):
            try:
                scene.add_entity(
                    morph=gs.morphs.URDF(
                        file=args.lexus_urdf,
                        pos=lexus_pos,
                        euler=lexus_euler,
                        fixed=True,
                        collision=True,
                        convexify=True,
                        scale=args.lexus_scale,
                        merge_fixed_links=True,
                    ),
                    surface=gs.surfaces.Collision()
                )
                lexus_loaded = True
            except Exception as e:
                print(f"[WARN] Lexus URDF add_entity failed, will fall back later: {e}")
        else:
            print("[INFO] Lexus URDF path not found or empty; will use plane-grid boundary fallback.")

        # create a SPH liquid entity as an initial droplet (we'll also emit rain)
        liquid_center = (0.5, 1.0, 1.0)
        liquid_radius = 0.12
        liquid_entity = scene.add_entity(
            material=gs.materials.SPH.Liquid(),
            morph=gs.morphs.Sphere(pos=liquid_center, radius=liquid_radius)
        )

        # keep an emitter for rain
        emitter = scene.add_emitter(
            material=gs.materials.SPH.Liquid(sampler="regular"),
            max_particles=100000,
            surface=gs.surfaces.Glass(
                color=(0.7, 0.85, 1.0, 0.7),
                vis_mode="particle"
            )
        )

        # camera (headless recorder)
        cam = scene.add_camera(res=(1280,720), pos=(5.5,6.5,3.2), lookat=(0.5,1.5,1.5), fov=35, GUI=False)

        # build (single env to be safe)
        scene.build(n_envs=1)

        # Warmup a few steps so the initial sphere turns to particles
        for _ in range(4):
            scene.step()

        # boundary points: try lexus mesh; fallback to a roof plane grid
        try:
            if lexus_loaded:
                urdf_info = get_first_mesh_from_urdf(args.lexus_urdf, prefer_tag="collision", pkg_root_hint=args.pkg_root_hint)
                mesh_file = urdf_info["file"]
                internal_scale = urdf_info["scale"]
                # combine internal and provided scale (uniform external scale)
                final_scale = (internal_scale[0]*args.lexus_scale,
                               internal_scale[1]*args.lexus_scale,
                               internal_scale[2]*args.lexus_scale)
                # sample mesh surface; apply translation = lexus_pos + urdf origin offset
                mesh_translate = tuple(np.array(lexus_pos) + np.array(urdf_info.get("xyz", (0.0,0.0,0.0))))
                if not _HAS_TRIMESH:
                    raise RuntimeError("trimesh required to sample mesh: pip install trimesh")
                boundary_points = sample_surface_even(mesh_file, args.boundary_points, scale=final_scale, translate=mesh_translate)
            else:
                raise RuntimeError("Lexus URDF not loaded; force fallback")
        except Exception as e:
            print("[INFO] Using plane-grid boundary fallback:", e)
            nx = max(1, int(math.sqrt(args.boundary_points)))
            xs = np.linspace(0.2, 0.8, nx)
            ys = np.linspace(0.9, 1.6, nx)
            XX, YY = np.meshgrid(xs, ys, indexing='xy')
            ZZ = np.full_like(XX, fill_value=lexus_pos[2], dtype=np.float32)
            boundary_points = np.stack([XX, YY, ZZ], axis=-1).reshape(-1,3)[:args.boundary_points].astype(np.float32)

        # simulate and record SPH positions
        frames = []
        if not args.no_video:
            cam.start_recording()

        for step_i in range(steps):
            # emit some rain from upstream (direction -z)
            emitter.emit(
                pos=np.array([0.5, 1.0, 3.5]),
                direction=np.array([0.0, 0.0, -1.0]),
                speed=5.0,
                droplet_shape="circle",
                droplet_size=0.22,
            )
            scene.step()
            if not args.no_video:
                cam.render()

            # get particles from liquid_entity
            try:
                pts = liquid_entity.get_particles()
                pts = to_Nx3(pts)
            except Exception:
                # fallback: try dump_ckpt_to_numpy search for sph-like arrays
                ck = scene.dump_ckpt_to_numpy()
                found = None
                for k,v in ck.items():
                    if isinstance(v, np.ndarray) and v.ndim==2 and v.shape[1]==3 and v.shape[0] >= 8:
                        kl = k.lower()
                        if 'sph' in kl or 'particle' in kl or '.x' in k or '/x' in k:
                            found = v; break
                if found is None:
                    for k,v in ck.items():
                        if isinstance(v, np.ndarray) and v.ndim==2 and v.shape[1]==3:
                            found = v; break
                if found is None:
                    raise RuntimeError("No fluid particle array found in ckpt; scene likely did not spawn SPH particles.")
                pts = np.asarray(found).copy()
            frames.append(pts.copy())

        if not args.no_video:
            cam.stop_recording(save_to_filename=args.video, fps=args.fps)

        # make dense (T, Nf_max, 3) with zero padding
        Nf_max = max(p.shape[0] for p in frames)
        T = len(frames)
        pos_array = np.zeros((T, Nf_max, 3), dtype=np.float32)
        for t_idx, arr in enumerate(frames):
            pos_array[t_idx, :arr.shape[0], :] = arr

        episodes.append({"pos": pos_array, "B": boundary_points})

    # After episodes, export to GNS format
    pack_and_save_gns(
        episodes,
        args.dataset_dir,
        T_target=None,
        dt=dt,
        connectivity_radius=3.0*args.particle_size
    )
    print("Done. Dataset saved to:", args.dataset_dir)


# ============================ script entry ============================

if __name__ == "__main__":
    main()
