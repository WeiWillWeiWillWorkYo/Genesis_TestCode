import numpy as np
import genesis as gs

########################## 初期化 ##########################
gs.init(seed=0, precision='32', logging_level='debug')

######################## シーンの作成 ##########################
dt = 4e-3
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=dt,
        substeps=10,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.3, 1.0, 0.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    rigid_options=gs.options.RigidOptions(
        dt=dt,
        gravity=(0, 0, -9.8),
        enable_collision=True,
        enable_self_collision=False,
    ),
    sph_options=gs.options.SPHOptions(
        dt=dt,
        lower_bound=(-1.5, -1.5, -0.5),
        upper_bound=(1.5, 1.5, 2.0),
        particle_size=0.008,
        gravity=(0, 0, -9.8),
    ),
    pbd_options=gs.options.PBDOptions(
        dt=dt,
        gravity=(0, 0, -9.8),
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        visualize_sph_boundary=True,
    ),
    show_viewer=False,
)

########################## エンティティ ##########################
# 地面
scene.add_entity(morph=gs.morphs.Plane())

# 布料
cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file='meshes/cloth.obj',
        scale=2.0,
        pos=(0, 0, 0.5),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0),
        vis_mode='visual',
    )
)

# 液体エミッター
liquid_emitter = scene.add_emitter(
    material=gs.materials.SPH.Liquid(
        sampler='regular',
    ),
    max_particles=2000,
    surface=gs.surfaces.Default(
        color=(0.4, 0.8, 1.0, 1.0),
        vis_mode='particle',
    ),
)

# カメラ
cam = scene.add_camera(
    res=(1280, 720),
    pos=(2, 1.5, 0.7),
    lookat=(0.0, 0.0, 0.0),
    fov=40,
    GUI=False,
)

########################## ビルド ##########################
scene.build()

########################## 布料固定 ##########################
cloth.fix_particle(0, cloth.find_closest_particle((-1, -1, 1.0)))
cloth.fix_particle(0, cloth.find_closest_particle((1, 1, 1.0)))
cloth.fix_particle(0, cloth.find_closest_particle((-1, 1, 1.0)))
cloth.fix_particle(0, cloth.find_closest_particle((1, -1, 1.0)))

########################## 実行 ##########################
scene.reset()

print("ビデオ録画を開始...")
cam.start_recording()

emit_counter = 0
for i in range(1000):
    if i % 5 == 0 and emit_counter < 200:
        try:
            liquid_emitter.emit(
                droplet_shape="circle",
                droplet_size=0.03,
                pos=(0.0, 0.0, 1.3),
                direction=(0, 0, -1.0),
                theta=0.0,
                speed=1.5,
                p_size=0.008,
            )
            emit_counter += 1
            if i % 25 == 0:
                print(f"フレーム {i}: {emit_counter} 回放出済み")
        except Exception as e:
            print(f"フレーム {i} で放出失敗: {e}")

    scene.step()
    cam.render()
    
    if i % 100 == 0:
        print(f"シミュレーション進行中... フレーム {i}")

print("録画を停止して動画を保存中...")
cam.stop_recording(save_to_filename="水流布料シミュレーション.mp4", fps=60)
print("動画を '水流布料シミュレーション.mp4' として保存完了")


python -m gns.train \
  --data_path="/home/cui_wei/Genesis/dataset_gns/" \
  --model_path="/home/cui_wei/Genesis/models/gns_model_newstack" \
  --ntraining_steps=100



python3 -m gns.train \
  --mode="rollout" \
  --data_path="/home/cui_wei/Genesis/dataset_gns/" \
  --model_path="/home/cui_wei/Genesis/models/" \
  --output_path="/home/cui_wei/Genesis/rollout/gns_model_newstack-100/" \
  --model_file="gns_model_newstackmodel-100.pt" \
  --train_state_file="gns_model_newstacktrain_state-100.pt"


python3 -m gns.render_rollout \
  --output_mode="gif" \
  --rollout_dir="/home/cui_wei/Genesis/rollout/gns_model_newstack-100/" \
  --rollout_name="rollout_ex0"

