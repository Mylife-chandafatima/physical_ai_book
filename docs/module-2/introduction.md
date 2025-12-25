---
title: Digital Twin Overview - Gazebo & Unity
sidebar_label: Digital Twin Overview
---

# Save as docs/module-2/introduction.md

# Digital Twin: Physics Simulation and High-Fidelity Rendering

## Learning Outcomes
By the end of this module, you will be able to:
- Understand the concept of digital twins in robotics
- Compare Gazebo and Unity for robotics simulation
- Implement physics-based simulation with realistic sensors
- Create high-fidelity visual environments for human-robot interaction
- Evaluate the trade-offs between different simulation platforms

## Introduction to Digital Twins in Robotics

A digital twin is a virtual replica of a physical system that serves as a real-time digital counterpart. In robotics, digital twins enable:
- Pre-deployment testing of algorithms
- Safe development of complex behaviors
- Rapid prototyping of robot designs
- Training of AI models in simulated environments
- Validation of control strategies before real-world deployment

### Benefits of Digital Twins

1. **Safety**: Test dangerous or complex behaviors in simulation before real-world deployment
2. **Cost-effectiveness**: Reduce hardware costs and prototyping time
3. **Repeatability**: Conduct controlled experiments with consistent conditions
4. **Scalability**: Train multiple agents simultaneously in parallel simulations
5. **Debugging**: Visualize internal states and system behavior in detail

## Gazebo: Physics-Based Simulation

Gazebo is a 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics research and development.

### Core Features of Gazebo

- **Physics Engine**: ODE, Bullet, Simbody, and DART physics engines
- **Sensor Simulation**: Cameras, LIDAR, IMU, force/torque sensors
- **Realistic Rendering**: OpenSceneGraph-based graphics
- **ROS Integration**: Direct integration with ROS/ROS 2
- **Plugins System**: Extensible through C++ plugins

### Setting up a Gazebo Environment

Here's an example of creating a simple Gazebo world:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="simple_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Include the sun -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Add a box obstacle -->
    <model name="box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Add a robot -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Physics Simulation Concepts

Gazebo uses physics engines to simulate real-world physics:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class PhysicsTutorial : public WorldPlugin
  {
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      // Set the physics parameters
      physics::PhysicsEnginePtr physics = _world->Physics();
      physics->SetGravity(math::Vector3(0, 0, -9.8));
      physics->SetMaxStepSize(0.001);
      physics->SetRealTimeUpdateRate(1000);
      physics->SetRealTimeFactor(1);
    }
  };
  GZ_REGISTER_WORLD_PLUGIN(PhysicsTutorial)
}
```

### Sensor Simulation in Gazebo

Gazebo provides realistic sensor simulation through plugins:

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30.0</update_rate>
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>800</width>
      <height>600</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <camera_name>camera</camera_name>
    <image_topic_name>image_raw</image_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
    <frame_name>camera_frame</frame_name>
  </plugin>
</sensor>
```

## Unity: High-Fidelity Rendering and Interaction

Unity is a powerful game engine that provides high-fidelity rendering capabilities, making it suitable for simulating complex visual environments and human-robot interaction scenarios.

### Unity for Robotics

Unity Robotics provides:
- **High-fidelity graphics**: Photorealistic rendering capabilities
- **Physics simulation**: Built-in physics engine with realistic collision detection
- **VR/AR support**: For immersive human-robot interaction
- **Cross-platform deployment**: Runs on multiple platforms
- **Asset ecosystem**: Large library of 3D models and environments

### Unity Robotics Simulation (URS) Framework

The Unity Robotics Simulation framework enables:
- High-fidelity visual rendering
- Physics simulation with realistic materials
- Sensor simulation (cameras, LIDAR, etc.)
- Integration with ROS/ROS 2

### Unity Environment Setup

Here's a basic Unity script for ROS integration:

```csharp
using UnityEngine;
using RosSharp;

public class RobotController : MonoBehaviour
{
    public float linearVelocity = 1.0f;
    public float angularVelocity = 1.0f;
    
    private float cmdLinear = 0.0f;
    private float cmdAngular = 0.0f;
    
    void Start()
    {
        // Initialize ROS connections
        RosSocket rosSocket = new RosSocket("ws://localhost:9090");
        rosSocket.Subscribe<Twist>("/cmd_vel", ReceiveTwist);
    }
    
    void ReceiveTwist(Twist cmd)
    {
        cmdLinear = cmd.linear.x;
        cmdAngular = cmd.angular.z;
    }
    
    void Update()
    {
        // Apply movement based on ROS commands
        transform.Translate(Vector3.forward * cmdLinear * Time.deltaTime);
        transform.Rotate(Vector3.up, cmdAngular * Time.deltaTime);
    }
}
```

### Unity Sensor Simulation

Unity can simulate various sensors:

```csharp
using UnityEngine;
using System.Collections;

public class CameraSensor : MonoBehaviour
{
    public int imageWidth = 640;
    public int imageHeight = 480;
    public string topicName = "/camera/image_raw";
    
    private WebCamTexture webCamTexture;
    private Texture2D texture2D;
    
    void Start()
    {
        // Initialize camera texture
        webCamTexture = new WebCamTexture();
        webCamTexture.requestedWidth = imageWidth;
        webCamTexture.requestedHeight = imageHeight;
        webCamTexture.Play();
        
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }
    
    void Update()
    {
        if (webCamTexture.didUpdateThisFrame)
        {
            // Capture image and send to ROS
            texture2D.SetPixels(webCamTexture.GetPixels());
            byte[] imageBytes = texture2D.EncodeToJPG();
            
            // Publish to ROS topic
            PublishImage(imageBytes);
        }
    }
    
    void PublishImage(byte[] imageBytes)
    {
        // Implementation to publish image to ROS
        // This would use a ROS bridge to send the image data
    }
}
```

## Comparison: Gazebo vs Unity

| Feature | Gazebo | Unity |
|---------|--------|-------|
| Physics Accuracy | High | Medium-High |
| Graphics Quality | Medium | Very High |
| Sensor Simulation | Excellent | Good |
| Learning Curve | Medium | Steep (for complex scenes) |
| ROS Integration | Native | Requires plugins |
| Performance | Optimized for physics | Optimized for graphics |
| Use Cases | Physics, control, navigation | Visual perception, HRI |

### When to Use Gazebo

- Physics-based simulation is critical
- Testing control algorithms
- Simulating realistic sensor data
- Robotics research requiring accurate physics
- Integration with existing ROS ecosystem

### When to Use Unity

- High-fidelity visual rendering is needed
- Human-robot interaction studies
- VR/AR applications
- Photorealistic perception training
- Complex visual environments

## Practical Exercise: Implementing a Simple Gazebo Environment

Let's create a complete Gazebo simulation environment:

1. **Create a world file** (`my_world.world`):

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="tutorial_world">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Add obstacles -->
    <model name="wall_1">
      <pose>-5 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.2 10 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 10 2</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Add a robot -->
    <include>
      <uri>model://turtlebot3_burger</uri>
      <pose>0 0 0.1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

2. **Launch the simulation**:

```bash
gazebo my_world.world
```

3. **Connect with ROS**:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class GazeboRobotController(Node):
    def __init__(self):
        super().__init__('gazebo_robot_controller')
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for laser scan
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Robot state
        self.obstacle_distance = float('inf')
        
    def laser_callback(self, msg):
        # Process laser scan to detect obstacles
        if len(msg.ranges) > 0:
            # Get minimum distance in front of robot
            front_distances = msg.ranges[:30] + msg.ranges[-30:]
            self.obstacle_distance = min([d for d in front_distances if not math.isinf(d)])
    
    def control_loop(self):
        cmd = Twist()
        
        # Simple obstacle avoidance
        if self.obstacle_distance > 1.0:
            cmd.linear.x = 0.5  # Move forward
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right
            
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboRobotController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration Strategies

### 1. Hybrid Simulation Approach

Combine Gazebo and Unity strengths:

- Use Gazebo for physics and sensor simulation
- Use Unity for high-fidelity visual rendering
- Synchronize states between both simulators

### 2. Sensor Simulation Optimization

- Use Gazebo for physics-based sensor simulation (LIDAR, IMU)
- Use Unity for camera simulation with realistic lighting
- Combine both for comprehensive sensor fusion training

### 3. Transfer Learning Considerations

- Domain randomization to bridge sim-to-real gap
- Adversarial training to improve generalization
- Progressive domain adaptation techniques

## Exercises

1. Create a Gazebo world with multiple obstacles and implement a navigation algorithm that avoids collisions using laser scan data.

2. Design a Unity scene with realistic lighting and textures, and simulate a camera sensor that publishes images to a ROS topic.

3. Compare the performance of a simple navigation algorithm when trained exclusively in Gazebo vs Unity. What are the differences in sim-to-real transfer?

4. Implement a human-robot interaction scenario in Unity where a human avatar can guide a robot through an environment using gestures.

## References

Koenig, N., & Howard, A. (2004). Design and use paradigms for Gazebo, an open-source multi-robot simulator. *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2149-2154.

Unity Technologies. (2023). *Unity Robotics Simulation*. Retrieved from https://github.com/Unity-Technologies/Unity-Robotics-Hub

Open Robotics. (2023). *Gazebo Documentation*. Retrieved from http://gazebosim.org/

Paigwar, A., & Erkorkmaz, K. (2019). ROS#.NET and Unity 3D integration for robotics simulation and programming. *IEEE International Conference on Robotics and Biomimetics*, 1518-1523.