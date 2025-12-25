# Save this file as specify/specs/module-2.spec.md

# Module 2: Digital Twin (Gazebo & Unity)

## Learning Outcomes

Upon completion of this module, students will be able to:
- Understand the fundamental concepts of digital twin technology in robotics
- Set up and configure Gazebo simulation environments for robotic applications
- Implement physics simulations including gravity, collision detection, and sensor modeling
- Create high-fidelity visualizations using Unity for robot simulation
- Design human-robot interaction scenarios in simulated environments
- Integrate sensor models (LiDAR, Depth Cameras, IMUs) into simulation workflows
- Configure URDF/SDF models for Gazebo simulation
- Debug and validate simulation performance and accuracy

## 1. Introduction to Digital Twin Concepts

Digital twin technology represents a virtual replica of a physical system, enabling real-time monitoring, analysis, and optimization. In robotics, digital twins serve as powerful tools for testing algorithms, validating control systems, and predicting robot behavior in various environments without the risk and cost associated with physical testing.

The digital twin concept in robotics encompasses three key components:
- **Physical Twin**: The actual robot in the real world
- **Virtual Twin**: The digital simulation model
- **Connection**: Real-time data flow between physical and virtual twins

Digital twins enable:
- Pre-deployment testing and validation
- Scenario-based training without hardware wear
- Performance optimization through simulation
- Safe testing of new algorithms

### Practical Exercise 1.1
Research and compare three different digital twin implementations in robotics, highlighting their advantages and limitations.

## 2. Gazebo Simulation Environment Setup

Gazebo is a 3D dynamic simulator widely used in robotics for simulating robots in realistic indoor and outdoor environments. It provides accurate physics simulation, high-quality graphics, and convenient programmatic interfaces.

### Installation and Configuration

To set up Gazebo for robotics simulation:

```bash
# Install Gazebo (example for Ubuntu)
sudo apt-get install gazebo libgazebo-dev

# For ROS 2 integration
sudo apt-get install ros-humble-gazebo-ros-pkgs
```

### Basic Environment Configuration

Create a simple world file to define your simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 0.4 -0.8</direction>
    </light>
  </world>
</sdf>
```

### Practical Exercise 2.1
Set up a basic Gazebo environment with custom lighting and terrain. Add at least three different objects with varying physical properties.

## 3. Physics Simulation: Gravity, Collisions, Sensors

Physics simulation is crucial for realistic robot behavior in digital twin environments. Gazebo uses the Open Dynamics Engine (ODE), Bullet, or DART physics engines to simulate realistic physical interactions.

### Gravity Configuration

Gravity is defined in the world file and affects all objects in the simulation:

```xml
<gravity>0 0 -9.8</gravity>  <!-- Standard Earth gravity -->
```

### Collision Detection

Collision detection in Gazebo is handled through collision elements in model definitions:

```xml
<collision name="collision">
  <geometry>
    <box>
      <size>1.0 1.0 1.0</size>
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <contact>
      <ode>
        <kp>1e+16</kp>
        <kd>1e+13</kd>
      </ode>
    </contact>
  </surface>
</collision>
```

### Physics Parameters

Important physics parameters include:
- **ERP (Error Reduction Parameter)**: Controls how much of the joint error is corrected per step
- **CFM (Constraint Force Mixing)**: Adds a small stabilizing force to constraints
- **Max Vel**: Maximum velocity of contacts
- **Min Depth**: Depth of penetration considered in contact calculations

### Practical Exercise 3.1
Create a simulation with different materials and friction coefficients. Test how different objects behave when interacting with various surfaces.

## 4. High-Fidelity Rendering in Unity

Unity provides high-fidelity rendering capabilities that can be integrated with robotics simulation for enhanced visualization and human-robot interaction studies.

### Unity-ROS Integration

Unity Robotics provides packages for ROS integration:

```csharp
// Example Unity script for ROS communication
using UnityEngine;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotName = "my_robot";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>("/joint_states");
    }

    void Update()
    {
        // Publish joint states to ROS
        JointStateMsg jointState = new JointStateMsg();
        jointState.name = new string[] { "joint1", "joint2" };
        jointState.position = new double[] { 0.0, 0.0 };
        ros.Publish("/joint_states", jointState);
    }
}
```

### Rendering Quality Settings

For high-fidelity rendering in Unity:
- Enable HDR rendering
- Use physically-based materials
- Implement realistic lighting models
- Apply post-processing effects
- Optimize for real-time performance

### Practical Exercise 4.1
Create a Unity scene with realistic lighting and materials that mirrors a Gazebo environment. Implement basic robot visualization with accurate joint positions.

## 5. Human-Robot Interaction Simulations

Human-robot interaction (HRI) simulations are essential for developing and testing interfaces, communication protocols, and collaborative behaviors in a safe environment.

### Interaction Scenarios

Common HRI simulation scenarios include:
- Gesture recognition and response
- Voice command processing
- Collaborative task execution
- Safety protocols and emergency responses
- Social interaction and navigation

### Simulation Components

HRI simulations typically include:
- Avatar models for human representation
- Sensor simulation for perception
- Behavior trees for decision making
- Communication protocols
- Safety validation mechanisms

### Practical Exercise 5.1
Design and implement a simple HRI scenario where a robot responds to human gestures. Include safety checks and validation procedures.

## 6. Sensor Simulation: LiDAR, Depth Cameras, IMUs

Accurate sensor simulation is critical for developing robust perception and navigation algorithms that transfer from simulation to reality.

### LiDAR Simulation

LiDAR sensors in Gazebo can be configured as ray sensors:

```xml
<sensor name="lidar" type="ray">
  <pose>0 0 0.3 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>
        <max_angle>1.570796</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>lidar</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

### Depth Camera Simulation

Depth cameras provide RGB-D data for 3D perception:

```xml
<sensor name="depth_camera" type="depth">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <camera_name>depth_camera</camera_name>
    <image_topic_name>rgb/image_raw</image_topic_name>
    <depth_image_topic_name>depth/image_raw</depth_image_topic_name>
    <point_cloud_topic_name>depth/points</point_cloud_topic_name>
    <camera_info_topic_name>rgb/camera_info</camera_info_topic_name>
  </plugin>
</sensor>
```

### IMU Simulation

Inertial Measurement Units (IMUs) provide orientation and acceleration data:

```xml
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <topicName>imu</topicName>
    <serviceName>imu_service</serviceName>
    <gaussianNoise>0.01</gaussianNoise>
    <xyzOffset>0 0 0</xyzOffset>
    <rpyOffset>0 0 0</rpyOffset>
  </plugin>
</sensor>
```

### Practical Exercise 6.1
Implement a multi-sensor fusion system that combines LiDAR, depth camera, and IMU data in a simulated environment. Validate the sensor data consistency and accuracy.

## 7. URDF/SDF Integration with Gazebo

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are essential for defining robot models in ROS and Gazebo respectively.

### URDF to SDF Conversion

A basic URDF model:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint1" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

### Gazebo-Specific Extensions

Gazebo-specific elements in URDF:

```xml
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.35</wheel_separation>
    <wheel_diameter>0.15</wheel_diameter>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>
```

### Practical Exercise 7.1
Create a complete URDF model of a simple mobile robot with differential drive, sensors, and Gazebo plugins. Load it in Gazebo and verify proper functionality.

## 8. Debugging and Simulation Testing

Effective debugging and testing strategies are essential for reliable simulation results.

### Common Simulation Issues

- **Physics instability**: Adjust ERP, CFM, and solver parameters
- **Sensor noise**: Validate sensor models against real hardware
- **Timing issues**: Ensure proper update rates and synchronization

### Testing Strategies

- Unit testing for individual components
- Integration testing for complete systems
- Performance benchmarking

## References

Digital Twin Consortium. (2021). *Digital Twin Vocabulary*. Digital Twin Consortium. https://www.digitaltwinconsortium.org/

Kazi, S., Datta, S., & De, P. (2020). Digital twin in manufacturing: A categorical literature review and classification. *ACM International Conference Proceeding Series*, 123-130.

Open Robotics. (2023). *Gazebo Documentation*. http://gazebosim.org/

Unity Technologies. (2023). *Unity Robotics Hub Documentation*. https://docs.unity3d.com/Packages/com.unity.robotics.ros-tcp-connector@latest

## Acceptance Criteria

Students will successfully complete this module when they can:
- Demonstrate proficiency in Gazebo simulation setup and configuration
- Successfully integrate a robot model with accurate physics properties
- Implement and validate sensor simulation for LiDAR, cameras, and IMUs
- Create a Unity visualization that accurately reflects the Gazebo simulation
- Design and execute a human-robot interaction scenario
- Debug and validate simulation performance against specified requirements
- Document simulation results with appropriate analysis and conclusions