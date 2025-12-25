---
title: Isaac Sim - Photorealistic Simulation
sidebar_label: Isaac Sim
---

# Save as docs/module-3/isaac-sim.md

# Isaac Sim: Photorealistic Simulation for Physical AI

## Learning Outcomes
By the end of this module, you will be able to:
- Set up and configure Isaac Sim for high-fidelity robotics simulation
- Create complex photorealistic environments with accurate physics
- Implement domain randomization for robust sim-to-real transfer
- Generate synthetic training data using Isaac Sim's capabilities
- Integrate Isaac Sim with ROS/ROS 2 for seamless development workflows
- Optimize simulation performance for large-scale training

## Introduction to Isaac Sim

Isaac Sim is NVIDIA's high-fidelity simulation environment built on the Omniverse platform. It provides:
- Physically accurate simulation using NVIDIA PhysX
- Photorealistic rendering with RTX technology
- Multi-GPU support for complex scenes
- Hardware-accelerated AI and perception
- Synthetic data generation capabilities
- Domain randomization for robust training

### Key Features of Isaac Sim

1. **High-Fidelity Physics**: NVIDIA PhysX engine with accurate collision detection and response
2. **Photorealistic Rendering**: RTX-accelerated rendering with global illumination
3. **USD-Based Scenes**: Universal Scene Description for complex scene composition
4. **Python API**: Extensive Python API for programmatic scene creation and control
5. **ROS/ROS 2 Integration**: Native integration with ROS/ROS 2 for robotics workflows
6. **Synthetic Data Generation**: Tools for generating labeled training data

## Setting Up Isaac Sim

### Installation and Prerequisites

Isaac Sim requires:
- NVIDIA GPU with RTX capabilities
- CUDA-compatible drivers
- Omniverse Nucleus server (optional for local development)
- Compatible USD viewer

### Basic Isaac Sim Configuration

```python
import omni
import carb
from omni.isaac.core import World
from omni.isaac.core.utils.stage import set_stage_up_axis
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.carb import set_carb_setting
import numpy as np

class IsaacSimConfig:
    def __init__(self):
        # Set up stage
        set_stage_up_axis("Z")  # Set up axis to Z
        
        # Configure simulation settings
        self.configure_simulation()
        
        # Create world instance
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/60.0,  # Physics timestep
            rendering_dt=1.0/60.0,  # Rendering timestep
            sim_params={
                "use_gpu": True,
                "use_gpu_dynamics": True,
                "solver_type": 1,  # 0: PGS, 1: TGS
                "num_position_iterations": 4,
                "num_velocity_iterations": 1,
                "max_depenetration_velocity": 10.0
            }
        )
    
    def configure_simulation(self):
        """Configure simulation parameters"""
        # Set physics solver settings
        set_carb_setting(
            carb.settings.get_settings(),
            "/physics/solverType", 1  # TGS solver
        )
        set_carb_setting(
            carb.settings.get_settings(),
            "/physics/iterations", 4
        )
        set_carb_setting(
            carb.settings.get_settings(),
            "/physics/solverPositionIterationCount", 4
        )
        set_carb_setting(
            carb.settings.get_settings(),
            "/physics/solverVelocityIterationCount", 1
        )
    
    def create_basic_scene(self):
        """Create a basic simulation scene"""
        # Add default ground plane
        self.world.scene.add_default_ground_plane(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.8
        )
        
        # Add lighting
        self.add_lighting()
        
        # Add basic objects
        self.add_basic_objects()
    
    def add_lighting(self):
        """Add lighting to the scene"""
        from omni.isaac.core.utils.prims import create_prim
        
        # Create dome light for ambient lighting
        create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            attributes={"color": (0.8, 0.8, 0.8), "intensity": 3000}
        )
        
        # Create directional light for shadows
        create_prim(
            prim_path="/World/DirectionalLight",
            prim_type="DistantLight",
            position=np.array([10, 10, 10]),
            attributes={
                "color": (0.9, 0.9, 0.9),
                "intensity": 1000,
                "shadows": True,
                "shadow_range": 100,
                "shadow_softness": 0.5
            }
        )
    
    def add_basic_objects(self):
        """Add basic objects to the scene"""
        # Add a robot (example with a simple cuboid)
        self.robot = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Robot",
                name="robot",
                position=np.array([0, 0, 0.5]),
                size=0.2,
                color=np.array([0.2, 0.8, 0.1])
            )
        )
        
        # Add obstacles
        for i in range(5):
            self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=np.array([2 + i, 0, 0.5]),
                    size=0.3,
                    color=np.array([0.8, 0.2, 0.1])
                )
            )

def main():
    # Initialize Isaac Sim configuration
    sim_config = IsaacSimConfig()
    sim_config.create_basic_scene()
    
    # Reset the world to apply changes
    sim_config.world.reset()
    
    # Run simulation loop
    for i in range(1000):
        sim_config.world.step(render=True)
        
        if i % 100 == 0:
            robot_pos, robot_ori = sim_config.robot.get_world_pose()
            print(f"Step {i}: Robot position: {robot_pos}")
    
    # Cleanup
    sim_config.world.clear()

if __name__ == "__main__":
    main()
```

## Creating Complex Environments

### Procedural Environment Generation

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.materials import OmniPBR
from pxr import UsdGeom, Gf, Sdf
import numpy as np
import random

class ProceduralEnvironment:
    def __init__(self, world: World):
        self.world = world
        self.stage = omni.usd.get_context().get_stage()
        
    def generate_indoor_environment(self, room_count=5, room_size_range=(3, 8)):
        """Generate a procedural indoor environment"""
        # Create ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add lighting
        self._add_environment_lighting()
        
        # Generate rooms
        for i in range(room_count):
            self._generate_room(i, room_size_range)
        
        # Add furniture and objects
        self._add_furniture_and_objects()
        
    def _add_environment_lighting(self):
        """Add environment lighting"""
        # Create dome light
        create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            attributes={"color": (0.7, 0.7, 0.8), "intensity": 2000}
        )
        
        # Create multiple point lights for different areas
        for i in range(5):
            create_prim(
                prim_path=f"/World/PointLight_{i}",
                prim_type="SphereLight",
                position=np.array([random.uniform(-10, 10), 
                                 random.uniform(-10, 10), 
                                 random.uniform(3, 6)]),
                attributes={
                    "color": (1.0, 1.0, 1.0),
                    "intensity": random.uniform(500, 1500),
                    "radius": 0.1
                }
            )
    
    def _generate_room(self, index, size_range):
        """Generate a single room"""
        # Random room dimensions
        width = random.uniform(size_range[0], size_range[1])
        depth = random.uniform(size_range[0], size_range[1])
        height = 3.0
        
        # Position rooms in a grid pattern
        x_pos = (index % 3) * (size_range[1] + 2)
        y_pos = (index // 3) * (size_range[1] + 2)
        
        # Create room walls
        self._create_room_walls(x_pos, y_pos, width, depth, height)
        
        # Create room floor
        self._create_room_floor(x_pos, y_pos, width, depth)
    
    def _create_room_walls(self, x, y, width, depth, height):
        """Create walls for a room"""
        wall_thickness = 0.2
        wall_height = height
        
        # Define wall positions and sizes
        walls = [
            # North wall
            {"position": [x, y + depth/2, wall_height/2], 
             "size": [width + 2*wall_thickness, wall_thickness, wall_height]},
            # South wall  
            {"position": [x, y - depth/2, wall_height/2], 
             "size": [width + 2*wall_thickness, wall_thickness, wall_height]},
            # East wall
            {"position": [x + width/2, y, wall_height/2], 
             "size": [wall_thickness, depth, wall_height]},
            # West wall
            {"position": [x - width/2, y, wall_height/2], 
             "size": [wall_thickness, depth, wall_height]}
        ]
        
        # Create wall materials
        wall_material = OmniPBR(
            prim_path="/World/Looks/wall_material",
            color=(0.8, 0.8, 0.8),
            roughness=0.7,
            metallic=0.0
        )
        
        # Create walls
        for i, wall in enumerate(walls):
            wall_prim = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Wall_{x}_{y}_{i}",
                    name=f"wall_{x}_{y}_{i}",
                    position=np.array(wall["position"]),
                    size=np.array(wall["size"]),
                    color=np.array([0.8, 0.8, 0.8]),
                    mass=1000.0  # Make walls static by giving high mass
                )
            )
            
            # Make wall static
            wall_prim.set_world_poses(
                positions=np.array(wall["position"]),
                orientations=np.array([0, 0, 0, 1])
            )
    
    def _create_room_floor(self, x, y, width, depth):
        """Create floor for a room"""
        floor = self.world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Floor_{x}_{y}",
                name=f"floor_{x}_{y}",
                position=np.array([x, y, -0.05]),
                size=np.array([width, depth, 0.1]),
                color=np.array([0.6, 0.6, 0.6]),
                mass=1000.0  # Static floor
            )
        )
    
    def _add_furniture_and_objects(self):
        """Add furniture and objects to rooms"""
        furniture_types = [
            {"name": "chair", "size": [0.5, 0.5, 0.8], "color": [0.6, 0.4, 0.2]},
            {"name": "table", "size": [1.0, 0.6, 0.7], "color": [0.4, 0.2, 0.1]},
            {"name": "box", "size": [0.3, 0.3, 0.3], "color": [0.9, 0.1, 0.1]}
        ]
        
        for room_idx in range(5):
            x = (room_idx % 3) * 10
            y = (room_idx // 3) * 10
            
            # Add random furniture to each room
            for _ in range(random.randint(2, 5)):
                furn_type = random.choice(furniture_types)
                
                # Random position within room bounds
                furn_x = x + random.uniform(-4, 4)
                furn_y = y + random.uniform(-4, 4)
                
                furniture = self.world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/{furn_type['name']}_{room_idx}_{furn_x}_{furn_y}",
                        name=f"{furn_type['name']}_{room_idx}",
                        position=np.array([furn_x, furn_y, furn_type['size'][2]/2]),
                        size=np.array(furn_type['size']),
                        color=np.array(furn_type['color']),
                        mass=random.uniform(1, 10)
                    )
                )

def setup_complex_environment():
    """Setup a complex procedural environment"""
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0
    )
    
    # Create procedural environment
    env = ProceduralEnvironment(world)
    env.generate_indoor_environment(room_count=8)
    
    # Add a robot to the environment
    robot = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Robot",
            name="robot",
            position=np.array([0, 0, 1.0]),
            size=0.3,
            color=np.array([0.1, 0.8, 0.2])
        )
    )
    
    return world, robot

# Usage example
world, robot = setup_complex_environment()
world.reset()

# Run simulation
for i in range(1000):
    world.step(render=True)
    
    if i % 100 == 0:
        robot_pos, robot_ori = robot.get_world_pose()
        print(f"Step {i}: Robot position: {robot_pos}")
```

## Domain Randomization

Domain randomization is crucial for improving sim-to-real transfer by exposing models to a wide variety of visual and physical conditions during training.

### Visual Domain Randomization

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.materials import OmniPBR
from pxr import UsdShade, Gf
import random
import numpy as np

class VisualDomainRandomizer:
    def __init__(self, world: World):
        self.world = world
        self.stage = omni.usd.get_context().get_stage()
        
    def randomize_materials(self, material_paths):
        """Randomize materials in the scene"""
        for path in material_paths:
            material_prim = get_prim_at_path(path)
            if material_prim:
                self._randomize_material(material_prim)
    
    def _randomize_material(self, material_prim):
        """Apply random material properties"""
        # Randomize base color
        base_color = (
            random.uniform(0.1, 1.0),
            random.uniform(0.1, 1.0), 
            random.uniform(0.1, 1.0)
        )
        
        # Randomize roughness
        roughness = random.uniform(0.1, 0.9)
        
        # Randomize metallic
        metallic = random.uniform(0.0, 0.3)
        
        # Apply randomization to material
        material = OmniPBR(
            prim_path=material_prim.GetPath().pathString,
            color=base_color,
            roughness=roughness,
            metallic=metallic
        )
    
    def randomize_lighting(self):
        """Randomize lighting conditions"""
        # Get all lights in the scene
        light_paths = ["/World/DomeLight", "/World/DirectionalLight"]
        
        for path in light_paths:
            light_prim = get_prim_at_path(path)
            if light_prim:
                # Randomize light intensity
                intensity = random.uniform(500, 3000)
                
                # Randomize light color
                color = (
                    random.uniform(0.8, 1.0),
                    random.uniform(0.8, 1.0),
                    random.uniform(0.8, 1.0)
                )
                
                # Apply changes
                light_prim.GetAttribute("inputs:intensity").Set(intensity)
                light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*color))
    
    def randomize_environment_textures(self):
        """Randomize environment textures"""
        # This would typically involve changing USD stage textures
        # For this example, we'll randomize material properties
        material_paths = [
            "/World/Looks/floor_material",
            "/World/Looks/wall_material"
        ]
        
        self.randomize_materials(material_paths)
    
    def randomize_camera_parameters(self, camera_prim_path):
        """Randomize camera parameters"""
        camera_prim = get_prim_at_path(camera_prim_path)
        
        if camera_prim:
            # Randomize exposure
            exposure = random.uniform(-2.0, 2.0)
            camera_prim.GetAttribute("inputs:exposure").Set(exposure)
            
            # Randomize focal length (if applicable)
            # This would depend on the specific camera implementation

class PhysicalDomainRandomizer:
    def __init__(self, world: World):
        self.world = world
    
    def randomize_friction(self, object_names):
        """Randomize friction coefficients"""
        for obj_name in object_names:
            obj = self.world.scene.get_object(obj_name)
            if obj:
                static_friction = random.uniform(0.1, 1.0)
                dynamic_friction = random.uniform(0.1, 1.0)
                
                # Apply friction changes
                # This would use Isaac's physics API
    
    def randomize_mass(self, object_names):
        """Randomize object masses"""
        for obj_name in object_names:
            obj = self.world.scene.get_object(obj_name)
            if obj:
                # Apply random mass variation
                base_mass = obj.get_mass()
                variation = random.uniform(0.8, 1.2)
                new_mass = base_mass * variation
                
                obj.set_mass(new_mass)
    
    def randomize_dynamics(self, object_names):
        """Randomize dynamic properties"""
        for obj_name in object_names:
            obj = self.world.scene.get_object(obj_name)
            if obj:
                # Randomize restitution (bounciness)
                restitution = random.uniform(0.0, 0.8)
                
                # Apply restitution
                # This would use Isaac's physics API

def apply_domain_randomization(world, step_interval=100):
    """Apply domain randomization during simulation"""
    visual_randomizer = VisualDomainRandomizer(world)
    physical_randomizer = PhysicalDomainRandomizer(world)
    
    # Get object names for physical randomization
    object_names = ["robot", "box_0", "chair_0"]  # Example object names
    
    for step in range(10000):
        world.step(render=True)
        
        # Apply domain randomization periodically
        if step % step_interval == 0:
            print(f"Applying domain randomization at step {step}")
            
            # Visual randomization
            visual_randomizer.randomize_lighting()
            visual_randomizer.randomize_environment_textures()
            
            # Physical randomization
            physical_randomizer.randomize_mass(object_names)
            physical_randomizer.randomize_friction(object_names)
            physical_randomizer.randomize_dynamics(object_names)
```

## Synthetic Data Generation

Isaac Sim excels at generating synthetic training data for machine learning models:

### RGB-D Data Generation

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import create_prim
from omni.replicator.core import random_colormap
import omni.replicator.core as rep
import numpy as np
import cv2
import os
from PIL import Image

class SyntheticDataGenerator:
    def __init__(self, world: World, output_dir="synthetic_data"):
        self.world = world
        self.output_dir = output_dir
        self.step_count = 0
        
        # Create output directories
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/segmentation", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        
        # Create camera
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([2.0, 2.0, 2.0]),
            look_at=np.array([0, 0, 0])
        )
        
        # Add camera to world
        self.world.scene.add(self.camera)
        
        # Initialize replicator for synthetic data
        self.setup_replicator()
    
    def setup_replicator(self):
        """Setup Omniverse Replicator for synthetic data generation"""
        # Create RGB provider
        rgb = rep.create.render_product("/World/Camera", (640, 480))
        
        # Create various outputs
        with rep.trigger.on_frame(num_frames=1000):
            # RGB output
            rgb_data = rep.randomizer.augmentations.rgb(rgb=rgb)
            
            # Depth output
            depth_data = rep.randomizer.augmentations.linear_depth(rgb=rgb)
            
            # Semantic segmentation output
            seg_data = rep.randomizer.augmentations.semantic_segmentation(rgb=rgb)
            
            # Write outputs
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=self.output_dir,
                rgb=rgb_data,
                depth=depth_data,
                segmentation=seg_data
            )
            writer.attach([rgb])
    
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data"""
        for i in range(num_samples):
            # Move camera to random position
            self.move_camera_randomly()
            
            # Move objects randomly
            self.move_objects_randomly()
            
            # Apply domain randomization
            self.apply_visual_randomization()
            
            # Step simulation
            self.world.step(render=True)
            
            # Capture data
            self.capture_data_frame(i)
            
            self.step_count += 1
    
    def move_camera_randomly(self):
        """Move camera to random position"""
        # Random spherical coordinates
        radius = np.random.uniform(1.5, 3.0)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(np.pi/6, np.pi/2)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # Set new camera pose
        self.camera.set_world_positions(np.array([x, y, z]))
        self.camera.look_at(np.array([0, 0, 0]))
    
    def move_objects_randomly(self):
        """Move objects to random positions"""
        # Get all objects in the scene
        objects = self.world.scene.objects
        for obj_name, obj in objects.items():
            if obj_name != "camera":  # Don't move camera
                # Random position offset
                offset = np.random.uniform(-1.0, 1.0, size=3)
                offset[2] = abs(offset[2])  # Keep above ground
                
                current_pos, current_ori = obj.get_world_pose()
                new_pos = current_pos + offset
                
                # Set new position
                obj.set_world_poses(positions=new_pos, orientations=current_ori)
    
    def apply_visual_randomization(self):
        """Apply visual domain randomization"""
        # Randomize lighting
        dome_light = get_prim_at_path("/World/DomeLight")
        if dome_light:
            intensity = np.random.uniform(500, 2500)
            dome_light.GetAttribute("inputs:intensity").Set(intensity)
    
    def capture_data_frame(self, frame_id):
        """Capture a single frame of data"""
        # Get camera data
        rgb_image = self.camera.get_rgb()
        depth_image = self.camera.get_depth()
        segmentation = self.camera.get_semantic_segmentation()
        
        # Save RGB image
        if rgb_image is not None:
            rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
            rgb_pil.save(f"{self.output_dir}/rgb/frame_{frame_id:06d}.png")
        
        # Save depth image
        if depth_image is not None:
            depth_pil = Image.fromarray((depth_image * 255).astype(np.uint8))
            depth_pil.save(f"{self.output_dir}/depth/frame_{frame_id:06d}.png")
        
        # Save segmentation
        if segmentation is not None:
            seg_pil = Image.fromarray((segmentation * 255).astype(np.uint8))
            seg_pil.save(f"{self.output_dir}/segmentation/frame_{frame_id:06d}.png")

def generate_synthetic_dataset():
    """Generate a complete synthetic dataset"""
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0
    )
    
    # Create a simple scene
    world.scene.add_default_ground_plane()
    
    # Add objects for segmentation
    for i in range(5):
        world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Object_{i}",
                name=f"object_{i}",
                position=np.array([np.random.uniform(-2, 2), 
                                 np.random.uniform(-2, 2), 
                                 0.5]),
                size=0.3,
                color=np.array([np.random.uniform(0, 1),
                               np.random.uniform(0, 1), 
                               np.random.uniform(0, 1)])
            )
        )
    
    # Initialize data generator
    data_gen = SyntheticDataGenerator(world, output_dir="isaac_synthetic_data")
    
    # Reset world
    world.reset()
    
    # Generate training data
    data_gen.generate_training_data(num_samples=500)
    
    print("Synthetic dataset generation completed!")

# Usage
generate_synthetic_dataset()
```

## Isaac Sim with ROS Integration

### ROS Bridge Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
from scipy.spatial.transform import Rotation as R

class IsaacSimROSBridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_bridge')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # ROS publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/rgb/camera_info', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.status_pub = self.create_publisher(String, '/sim_status', 10)
        
        # ROS subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # Isaac Sim components
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/60.0,
            rendering_dt=1.0/60.0
        )
        
        # Robot and camera setup
        self.setup_robot_and_camera()
        
        # Simulation parameters
        self.simulation_active = True
        self.publish_timer = self.create_timer(1.0/30.0, self.publish_sensor_data)  # 30 Hz
        self.cmd_vel = Twist()
        
    def setup_robot_and_camera(self):
        """Setup robot and camera in Isaac Sim"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Create robot (simplified as a cuboid)
        self.robot = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Robot",
                name="robot",
                position=np.array([0, 0, 0.5]),
                size=0.5,
                color=np.array([0.1, 0.8, 0.2])
            )
        )
        
        # Create camera attached to robot
        self.camera = Camera(
            prim_path="/World/Robot/Camera",
            position=np.array([0.2, 0, 0.1]),  # Position relative to robot
            look_at=np.array([1, 0, 0])  # Look forward
        )
        self.world.scene.add(self.camera)
        
        # Initialize camera parameters
        self.camera_resolution = (640, 480)
        self.camera_intrinsics = np.array([
            [320, 0, 320],    # fx, 0, cx
            [0, 320, 240],    # 0, fy, cy  
            [0, 0, 1]         # 0, 0, 1
        ])
    
    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS"""
        self.cmd_vel = msg
    
    def update_robot_motion(self):
        """Update robot motion based on velocity commands"""
        if self.world.current_time_step_index % 2 == 0:  # Update every 2 steps for stability
            # Get current pose
            current_pos, current_ori = self.robot.get_world_pose()
            
            # Convert orientation quaternion to rotation matrix
            rot = R.from_quat([current_ori[0], current_ori[1], current_ori[2], current_ori[3]])
            
            # Apply linear velocity in robot's forward direction
            linear_vel_world = rot.apply([self.cmd_vel.linear.x, 0, 0])
            
            # Calculate new position
            new_pos = current_pos + linear_vel_world * (1.0/60.0)  # Assuming 60 Hz physics
            
            # Apply angular velocity for rotation
            angular_vel = np.array([0, 0, self.cmd_vel.angular.z])
            angular_quat = R.from_rotvec(angular_vel * (1.0/60.0)).as_quat()
            
            # Combine rotations
            new_rot = R.from_quat([current_ori[0], current_ori[1], current_ori[2], current_ori[3]])
            new_rot = new_rot * R.from_quat(angular_quat)
            
            # Update robot pose
            self.robot.set_world_poses(
                positions=new_pos,
                orientations=new_rot.as_quat()
            )
    
    def publish_sensor_data(self):
        """Publish sensor data to ROS topics"""
        if not self.simulation_active:
            return
        
        # Update robot motion
        self.update_robot_motion()
        
        # Get sensor data
        rgb_image = self.camera.get_rgb()
        depth_image = self.camera.get_depth()
        
        # Get robot pose for odometry
        robot_pos, robot_ori = self.robot.get_world_pose()
        robot_lin_vel, robot_ang_vel = self.robot.get_linear_velocity(), self.robot.get_angular_velocity()
        
        # Publish RGB image
        if rgb_image is not None:
            rgb_msg = self.bridge.cv2_to_imgmsg((rgb_image * 255).astype(np.uint8), encoding='rgb8')
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = 'camera_rgb_optical_frame'
            self.rgb_pub.publish(rgb_msg)
        
        # Publish depth image
        if depth_image is not None:
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image.astype(np.float32), encoding='32FC1')
            depth_msg.header.stamp = self.get_clock().now().to_img()
            depth_msg.header.frame_id = 'camera_depth_optical_frame'
            self.depth_pub.publish(depth_msg)
        
        # Publish camera info
        camera_info_msg = self.create_camera_info_msg()
        self.camera_info_pub.publish(camera_info_msg)
        
        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Position
        odom_msg.pose.pose.position.x = float(robot_pos[0])
        odom_msg.pose.pose.position.y = float(robot_pos[1])
        odom_msg.pose.pose.position.z = float(robot_pos[2])
        
        # Orientation
        odom_msg.pose.pose.orientation.x = float(robot_ori[0])
        odom_msg.pose.pose.orientation.y = float(robot_ori[1])
        odom_msg.pose.pose.orientation.z = float(robot_ori[2])
        odom_msg.pose.pose.orientation.w = float(robot_ori[3])
        
        # Velocity
        odom_msg.twist.twist.linear.x = float(robot_lin_vel[0])
        odom_msg.twist.twist.linear.y = float(robot_lin_vel[1])
        odom_msg.twist.twist.linear.z = float(robot_lin_vel[2])
        odom_msg.twist.twist.angular.x = float(robot_ang_vel[0])
        odom_msg.twist.twist.angular.y = float(robot_ang_vel[1])
        odom_msg.twist.twist.angular.z = float(robot_ang_vel[2])
        
        self.odom_pub.publish(odom_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Simulation running - Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})"
        self.status_pub.publish(status_msg)
    
    def create_camera_info_msg(self):
        """Create camera info message"""
        camera_info = CameraInfo()
        camera_info.header.stamp = self.get_clock().now().to_msg()
        camera_info.header.frame_id = 'camera_rgb_optical_frame'
        camera_info.height = self.camera_resolution[1]
        camera_info.width = self.camera_resolution[0]
        camera_info.distortion_model = 'plumb_bob'
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        
        # Intrinsic matrix
        camera_info.k = [
            float(self.camera_intrinsics[0, 0]), 0.0, float(self.camera_intrinsics[0, 2]),  # fx, 0, cx
            0.0, float(self.camera_intrinsics[1, 1]), float(self.camera_intrinsics[1, 2]),  # 0, fy, cy
            0.0, 0.0, 1.0  # 0, 0, 1
        ]
        
        # Projection matrix
        camera_info.p = [
            float(self.camera_intrinsics[0, 0]), 0.0, float(self.camera_intrinsics[0, 2]), 0.0,  # fx', 0, cx', Tx
            0.0, float(self.camera_intrinsics[1, 1]), float(self.camera_intrinsics[1, 2]), 0.0,  # 0, fy', cy', Ty
            0.0, 0.0, 1.0, 0.0  # 0, 0, 1, Tz
        ]
        
        return camera_info

def main(args=None):
    rclpy.init(args=args)
    
    # Initialize Isaac Sim
    omni.usd.get_context().new_stage()
    
    # Create ROS bridge node
    ros_bridge = IsaacSimROSBridge()
    
    # Reset Isaac Sim world
    ros_bridge.world.reset()
    
    try:
        rclpy.spin(ros_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        ros_bridge.simulation_active = False
        ros_bridge.world.clear()
        ros_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Simulation Optimization Techniques

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.settings import set_physics_solver_type
from omni.physx.scripts import physicsUtils
import carb

class IsaacSimOptimizer:
    def __init__(self, world: World):
        self.world = world
        self.stage = omni.usd.get_context().get_stage()
    
    def optimize_physics_settings(self):
        """Optimize physics simulation settings"""
        # Set solver type to TGS for better stability
        set_physics_solver_type(1)  # TGS solver
        
        # Configure solver parameters
        physicsUtils.set_physics_scene_parameters(
            self.world.scene,
            gravity=-981.0,  # cm/s^2
            solver_type=1,   # TGS solver
            num_position_iterations=4,
            num_velocity_iterations=1,
            max_depenetration_velocity=10.0
        )
    
    def optimize_rendering_settings(self):
        """Optimize rendering settings for performance"""
        # Set rendering parameters
        carb.settings.get_settings().set("/app/window/drawMouse", False)
        carb.settings.get_settings().set("/app/viewport/renderMode", "Material")
        
        # Reduce shadow quality if needed
        carb.settings.get_settings().set("/rtx/transmission/enabled", False)
        carb.settings.get_settings().set("/rtx/reflections/enabled", False)
    
    def optimize_object_properties(self):
        """Optimize object properties for better performance"""
        # Reduce collision mesh complexity for distant objects
        # Use simpler shapes for collision detection
        # Implement LOD (Level of Detail) for complex objects
        
        # Example: Simplify collision for objects far from camera
        objects = self.world.scene.objects
        for obj_name, obj in objects.items():
            # Simplify collision based on distance or importance
            pass
    
    def batch_operations(self):
        """Batch operations for better performance"""
        # Batch physics updates
        # Batch rendering operations
        # Batch data transfers
        
        # Example: Batch pose updates
        all_poses = []
        all_orientations = []
        
        for obj_name, obj in self.world.scene.objects.items():
            pos, ori = obj.get_world_pose()
            all_poses.append(pos)
            all_orientations.append(ori)
        
        # Update all poses at once if possible
        # This depends on the specific Isaac Sim API

def setup_optimized_simulation():
    """Setup an optimized Isaac Sim instance"""
    # Configure simulation parameters for optimal performance
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,  # 60 Hz physics
        rendering_dt=1.0/30.0,  # 30 Hz rendering (every 2 physics steps)
        sim_params={
            "use_gpu": True,
            "use_gpu_dynamics": True,
            "solver_type": 1,  # TGS solver
            "num_position_iterations": 4,
            "num_velocity_iterations": 1,
            "max_depenetration_velocity": 10.0
        }
    )
    
    # Apply optimizations
    optimizer = IsaacSimOptimizer(world)
    optimizer.optimize_physics_settings()
    optimizer.optimize_rendering_settings()
    
    return world
```

## Exercises

1. Create an Isaac Sim environment with multiple rooms and populate it with furniture. Implement domain randomization for materials, lighting, and object positions.

2. Design a synthetic data generation pipeline that creates RGB-D datasets for training a semantic segmentation model. Include variation in lighting, materials, and object configurations.

3. Implement a ROS bridge that connects Isaac Sim to a navigation stack. Test path planning and obstacle avoidance in the simulated environment.

4. Create a complex manipulation scene with articulated robots and objects. Implement physics optimization techniques to maintain stable simulation performance.

## References

NVIDIA. (2023). *NVIDIA Isaac Sim Documentation*. Retrieved from https://docs.omniverse.nvidia.com/isaacsim/

NVIDIA. (2023). *Omniverse Replicator Documentation*. Retrieved from https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html

NVIDIA. (2023). *NVIDIA Isaac Lab Synthetic Data Generation*. Retrieved from https://isaac-sim.github.io/IsaacLab/

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 23-30.