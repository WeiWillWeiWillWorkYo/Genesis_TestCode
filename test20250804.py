import numpy as np
import genesis as gs
import math
import time

# Check Genesis environment
print("🔍 Checking Genesis environment...")
print(f"Genesis version: {getattr(gs, '__version__', 'Unknown')}")

# Check CouplerOptions availability
coupler_available = False
try:
    # Try different possible paths
    if hasattr(gs.options, 'CouplerOptions'):
        coupler_available = True
        print("✓ CouplerOptions available in gs.options")
    elif hasattr(gs.options, 'solvers') and hasattr(gs.options.solvers, 'CouplerOptions'):
        coupler_available = True
        print("✓ CouplerOptions available in gs.options.solvers")
    else:
        print("⚠️ CouplerOptions not available, using default coupling")
        print("   (This won't affect simulation, Genesis enables solver coupling by default)")
except Exception as e:
    print(f"⚠️ Error checking CouplerOptions: {e}")

# Check MPM-PBD coupling availability
coupler_available = False
try:
    # Check LegacyCouplerOptions and MPM-PBD coupling
    if hasattr(gs.options, 'solvers') and hasattr(gs.options.solvers, 'LegacyCouplerOptions'):
        coupler_available = True
        print("✓ LegacyCouplerOptions available in gs.options.solvers")
        print("✓ Enable MPM-PBD coupling for fluid-cloth interaction")
        print("✓ Removed ineffective side emitter, focus on single powerful impact")
    else:
        print("⚠️ LegacyCouplerOptions not available, using default coupling")
        print("   MPM-PBD interaction may be limited")
except Exception as e:
    print(f"⚠️ Error checking coupling options: {e}")

########################## Initialization ##########################
gs.init(seed=0, precision='32', logging_level='info')

######################## Scene Creation ##########################
dt = 4e-3
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=dt,
        substeps=10,
    ),
    # Use MPM-PBD coupling instead of problematic FEM approach
    coupler_options=gs.options.solvers.LegacyCouplerOptions(
        rigid_sph=True,    # Enable rigid-SPH coupling (sphere interaction)
        mpm_pbd=True,      # Enable MPM-PBD coupling (fluid-cloth interaction)
        rigid_pbd=True,    # Enable rigid-PBD coupling (ground-cloth interaction)
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, 1.5, 1.2), 
        camera_lookat=(0.0, 0.0, 0.6),
        camera_fov=45,
    ),
    rigid_options=gs.options.RigidOptions(
        dt=dt,
        gravity=(0, 0, -9.8),
        enable_collision=True,
        enable_self_collision=False,
    ),
    sph_options=gs.options.SPHOptions(
        dt=dt,
        lower_bound=(-2.5, -2.5, -0.5),  # Expand boundaries for larger fluid flow
        upper_bound=(2.5, 2.5, 2.5),
        particle_size=0.015,  # Further increase particle size for obvious interaction
        gravity=(0, 0, -9.8),
    ),
    # Add MPM options to support coupling with PBD
    mpm_options=gs.options.MPMOptions(
        dt=dt,
        lower_bound=(-2.5, -2.5, -0.5),
        upper_bound=(2.5, 2.5, 2.5),
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

########################## Entity Setup ##########################
# Ground
ground = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(),
    surface=gs.surfaces.Default(
        color=(0.4, 0.4, 0.4, 1.0),
        vis_mode='visual',
    )
)

# Cloth - using PBD material to avoid mesh compatibility issues
cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),  # Back to PBD
    morph=gs.morphs.Mesh(
        file='meshes/cloth.obj',
        scale=2.2,  # Slightly larger cloth
        pos=(0, 0, 0.6),  # Raise cloth height
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.8, 0.3, 0.3, 0.9),  # Red semi-transparent for better observation
        vis_mode='visual',
    )
)

# Central liquid emitter - focus on effective cloth impact
liquid_emitter_center = scene.add_emitter(
    material=gs.materials.MPM.Liquid(
        sampler='regular',  # Use regular sampling for consistency
    ),
    max_particles=5000,  # More particles focused on single powerful impact
    surface=gs.surfaces.Default(
        color=(0.2, 0.7, 1.0, 0.8),
        vis_mode='particle',
    ),
)

# Reference sphere to verify liquid interaction
reference_sphere = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.Sphere(
        radius=0.08,
        pos=(0.3, 0.3, 0.3),
    ),
    surface=gs.surfaces.Default(
        color=(1.0, 0.8, 0.2, 1.0),  # Golden color
        vis_mode='visual',
    )
)

# Camera
cam = scene.add_camera(
    res=(1280, 720),
    pos=(2.5, 2.0, 1.0),  # Adjust camera position for better view
    lookat=(0.0, 0.0, 0.6),
    fov=40,
    GUI=False,
)

########################## Build ##########################
scene.build()

########################## Execution ##########################
scene.reset()

########################## PBD Cloth Constraints ##########################
print("Setting PBD cloth constraints...")
# Back to original PBD cloth fixing method
corners = [
    (-1.1, -1.1, 0.6),  # Bottom-left corner
    (1.1, 1.1, 0.6),    # Top-right corner
    (-1.1, 1.1, 0.6),   # Top-left corner
    (1.1, -1.1, 0.6),   # Bottom-right corner
]

for corner in corners:
    particle_id = cloth.find_closest_particle(corner)
    if particle_id is not None:
        cloth.fix_particle(0, particle_id)
        print(f"Fixed particle {particle_id} at position {corner}")

print("PBD cloth constraints setup completed")

########################## Enhanced Emission Control System ##########################
class EnhancedEmissionController:
    def __init__(self):
        self.phase = 0
        self.center_counter = 0
        self.max_center_emissions = 500  # All particles focused on central impact
        
    def update_phase(self, frame):
        """Smart phase control"""
        if frame < 200:
            self.phase = 0  # Warm-up phase
        elif frame < 600:
            self.phase = 1  # Powerful impact phase
        elif frame < 900:
            self.phase = 2  # Sustained impact phase
        else:
            self.phase = 3  # Finishing phase
            
    def should_emit_center(self, frame):
        """Central emitter control - focus on effective impact"""
        if self.center_counter >= self.max_center_emissions:
            return False
            
        if self.phase == 0:
            return frame % 4 == 0   # Faster warm-up
        elif self.phase == 1:
            return frame % 2 == 0   # Very dense main impact
        elif self.phase == 2:
            return frame % 3 == 0   # Sustained impact
        else:
            return frame % 8 == 0   # Sparse finishing
            
    def get_center_emission_params(self, frame):
        """Central emitter parameters - optimize single impact effect"""
        base_speed = 2.2  # Enhanced speed
        base_size = 0.04   # Larger droplet size
        
        if self.phase == 0:
            speed_mult = 0.8
            size_mult = 0.9
        elif self.phase == 1:
            # Strong fluctuation effects in main impact phase
            speed_mult = 1.0 + 0.5 * math.sin(frame * 0.08)
            size_mult = 1.0 + 0.4 * math.cos(frame * 0.12)
        elif self.phase == 2:
            speed_mult = 1.0 + 0.2 * math.sin(frame * 0.05)
            size_mult = 1.0
        else:
            speed_mult = 0.6  # Weakened finishing
            size_mult = 0.8
            
        return {
            "speed": base_speed * speed_mult,
            "droplet_size": base_size * size_mult,
            "pos": (0.0, 0.0, 1.4),
            "direction": (0, 0, -1.0),
        }

# Create enhanced controller
controller = EnhancedEmissionController()

# Start recording
print("Starting video recording...")
cam.start_recording()

print("Starting enhanced simulation...")
start_time = time.time()

# Main simulation loop
for i in range(1200):  # Extended simulation time
    controller.update_phase(i)
    
    # Focus on central emitter's powerful impact
    if controller.should_emit_center(i):
        try:
            params = controller.get_center_emission_params(i)
            liquid_emitter_center.emit(
                droplet_shape="circle",
                droplet_size=params["droplet_size"],
                pos=params["pos"],
                direction=params["direction"],
                theta=0.0,
                speed=params["speed"],
                p_size=0.015,  # Keep consistent with particle settings
            )
            controller.center_counter += 1
        except Exception as e:
            if i % 100 == 0:
                print(f"Central emitter error (frame {i}): {e}")

    # Physics step
    scene.step()
    cam.render()
    
    # Simplified progress report
    if i % 150 == 0:
        elapsed = time.time() - start_time
        fps = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"Progress: {i}/1200 | Phase: {controller.phase} | "
              f"Emissions: {controller.center_counter}/{controller.max_center_emissions} | "
              f"FPS: {fps:.1f}")

# Complete statistics
total_time = time.time() - start_time
print("Stopping recording and saving video...")
cam.stop_recording(save_to_filename="enhanced_fluid_cloth_interaction.mp4", fps=60)

print(f"\n=== Focused Central Impact MPM-PBD Simulation Complete ===")
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Average FPS: {1200/total_time:.1f}")
print(f"Total emissions: {controller.center_counter}")
print(f"Emission efficiency: {controller.center_counter}/{controller.max_center_emissions} ({100*controller.center_counter/controller.max_center_emissions:.1f}%)")
print(f"Video saved: enhanced_fluid_cloth_interaction.mp4")
print(f"\nImprovements:")
print(f"✅ Focus on single powerful central impact effect")
print(f"✅ Increased particle count (5000) and emission frequency")
print(f"✅ MPM-PBD coupling ensures realistic interaction")
print(f"✅ Enhanced speed (2.2) and droplet size (0.04)")
print(f"✅ Four-phase dynamic emission control")

# Check Genesis version information
try:
    import genesis as gs
    print(f"\n🔍 Debug information:")
    print(f"Genesis version: {gs.__version__ if hasattr(gs, '__version__') else 'Unknown'}")
    print(f"Genesis installation path: {gs.__file__ if hasattr(gs, '__file__') else 'Unknown'}")
except Exception as e:
    print(f"Cannot get Genesis version info: {e}")
