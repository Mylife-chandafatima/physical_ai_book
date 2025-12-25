---
title: URDF for Humanoids
sidebar_label: URDF for Humanoids
---

# Save as docs/module-1/urdf-humanoids.md

# URDF for Humanoids: Modeling Humanoid Robots

## Learning Outcomes
By the end of this module, you will be able to:
- Understand the structure and components of URDF (Unified Robot Description Format)
- Create URDF models for humanoid robots with proper kinematic chains
- Implement joint constraints and limits for realistic humanoid movement
- Integrate sensors and actuators into URDF models
- Validate and debug URDF models for simulation and real robots

## Introduction to URDF

Unified Robot Description Format (URDF) is an XML-based format used to describe robots in ROS. It contains information about joints, links, inertial properties, visual and collision models, and other physical properties of a robot. For humanoid robots, URDF is essential for simulation, kinematic analysis, and control.

### Key Components of URDF

- **Links**: Rigid bodies with mass, visual representation, and collision properties
- **Joints**: Connections between links with specific degrees of freedom
- **Materials**: Visual appearance properties
- **Transmissions**: Mapping between actuators and joints
- **Gazebo plugins**: Simulation-specific extensions

## URDF Structure for Humanoid Robots

Humanoid robots have a specific kinematic structure that typically includes:
- Torso (trunk)
- Head with neck joints
- Two arms with shoulders, elbows, and wrists
- Two legs with hips, knees, and ankles
- Sometimes hands with fingers

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Joints -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>
  
  <!-- More links and joints... -->
</robot>
```

## Detailed Humanoid URDF Example

Here's a more complete example of a humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
  
  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.2"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Torso -->
  <link name="torso">
    <inertial>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>
  
  <!-- Head -->
  <link name="head">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="orange"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>
  
  <!-- Left Arm -->
  <link name="left_shoulder">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="green"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.2 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>
  
  <link name="left_upper_arm">
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="green"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="0.5" effort="50" velocity="2.0"/>
  </joint>
  
  <!-- Right Arm (similar to left) -->
  <link name="right_shoulder">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="red"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="-0.2 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>
  
  <link name="right_upper_arm">
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="red"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="0.5" effort="50" velocity="2.0"/>
  </joint>
  
  <!-- Left Leg -->
  <link name="left_hip">
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.3"/>
      </geometry>
      <material name="green"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.3"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_hip"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1.0"/>
  </joint>
  
  <link name="left_lower_leg">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="green"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="100" velocity="1.0"/>
  </joint>
  
  <!-- Right Leg -->
  <link name="right_hip">
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.3"/>
      </geometry>
      <material name="red"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.3"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_hip"/>
    <origin xyz="-0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1.0"/>
  </joint>
  
  <link name="right_lower_leg">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="right_knee_joint" type="revolute">
    <parent link="right_hip"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="100" velocity="1.0"/>
  </joint>
  
  <!-- Gazebo plugin for simulation -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
  </gazebo>
  
  <gazebo reference="torso">
    <material>Gazebo/Blue</material>
  </gazebo>
  
  <gazebo reference="head">
    <material>Gazebo/Orange</material>
  </gazebo>
  
  <gazebo reference="left_shoulder">
    <material>Gazebo/Green</material>
  </gazebo>
  
  <gazebo reference="right_shoulder">
    <material>Gazebo/Red</material>
  </gazebo>
  
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <joint_name>torso_joint</joint_name>
      <joint_name>neck_joint</joint_name>
      <joint_name>left_shoulder_joint</joint_name>
      <joint_name>left_elbow_joint</joint_name>
      <joint_name>right_shoulder_joint</joint_name>
      <joint_name>right_elbow_joint</joint_name>
      <joint_name>left_hip_joint</joint_name>
      <joint_name>left_knee_joint</joint_name>
      <joint_name>right_hip_joint</joint_name>
      <joint_name>right_knee_joint</joint_name>
    </plugin>
  </gazebo>
  
</robot>
```

## Xacro for Complex Humanoid Models

Xacro (XML Macros) is a macro language that extends URDF to make complex models more manageable:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">
  
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_mass" value="10.0" />
  <xacro:property name="torso_mass" value="8.0" />
  <xacro:property name="arm_mass" value="1.5" />
  <xacro:property name="leg_mass" value="2.0" />
  
  <!-- Inertial macro -->
  <xacro:macro name="default_inertial" params="mass">
    <inertial>
      <mass value="${mass}" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
  </xacro:macro>
  
  <!-- Link macro -->
  <xacro:macro name="simple_link" params="name mass xyz size material">
    <link name="${name}">
      <xacro:default_inertial mass="${mass}"/>
      <visual>
        <origin xyz="${xyz}" rpy="0 0 0"/>
        <geometry>
          <box size="${size}"/>
        </geometry>
        <material name="${material}"/>
      </visual>
      <collision>
        <origin xyz="${xyz}" rpy="0 0 0"/>
        <geometry>
          <box size="${size}"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>
  
  <!-- Base link -->
  <xacro:simple_link name="base_link" mass="${base_mass}" 
                     xyz="0 0 0.1" size="0.3 0.3 0.2" material="white"/>
  
  <!-- Torso -->
  <xacro:simple_link name="torso" mass="${torso_mass}" 
                     xyz="0 0 0.3" size="0.3 0.2 0.6" material="blue"/>
  
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>
  
  <!-- Arms using macros -->
  <xacro:macro name="arm_chain" params="side parent_link parent_xyz">
    <!-- Shoulder -->
    <link name="${side}_shoulder">
      <inertial>
        <mass value="1.0" />
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001" />
      </inertial>
      <visual>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <material name="green"/>
      </visual>
      <collision>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
      </collision>
    </link>
    
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${parent_xyz}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
    </joint>
    
    <!-- Upper arm -->
    <link name="${side}_upper_arm">
      <inertial>
        <mass value="${arm_mass}" />
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01" />
      </inertial>
      <visual>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.3"/>
        </geometry>
        <material name="green"/>
      </visual>
      <collision>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.3"/>
        </geometry>
      </collision>
    </link>
    
    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-2.0" upper="0.5" effort="50" velocity="2.0"/>
    </joint>
  </xacro:macro>
  
  <!-- Instantiate arms -->
  <xacro:arm_chain side="left" parent_link="torso" parent_xyz="0.2 0 0.3"/>
  <xacro:arm_chain side="right" parent_link="torso" parent_xyz="-0.2 0 0.3"/>
  
</robot>
```

## Joint Types for Humanoid Robots

Humanoid robots require various joint types to achieve human-like movement:

### 1. Revolute Joints
```xml
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="lower_arm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="0.5" effort="50" velocity="2.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### 2. Continuous Joints
```xml
<joint name="waist_joint" type="continuous">
  <parent link="torso"/>
  <child link="pelvis"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.1"/>
</joint>
```

### 3. Fixed Joints
```xml
<joint name="head_joint" type="fixed">
  <parent link="neck"/>
  <child link="head"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>
```

### 4. Prismatic Joints
```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="base"/>
  <child link="slider"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1.0"/>
</joint>
```

## Sensors in URDF

Sensors can be integrated into URDF models for simulation:

```xml
<!-- IMU Sensor -->
<link name="imu_link">
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>

<!-- Camera Sensor -->
<link name="camera_link">
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov>
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
      <frame_name>camera_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Validation and Debugging

### Checking URDF Validity
```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Parse and display robot structure
urdf_to_graphiz /path/to/robot.urdf
```

### Common URDF Issues
1. **Missing parent links**: Ensure all joints have valid parent links
2. **Inconsistent units**: Use consistent units (typically meters for distance, kg for mass)
3. **Invalid joint limits**: Ensure joint limits are physically plausible
4. **Inertia tensor errors**: Verify that inertia tensors are physically valid

## Exercises

1. Create a simplified URDF model of a humanoid robot with at least 12 degrees of freedom (DOF). Include the torso, head, two arms, and two legs.

2. Implement a URDF model for a humanoid robot with a camera sensor mounted on the head and an IMU in the torso. Include the Gazebo plugins for these sensors.

3. Design a URDF model for a humanoid robot hand with at least 4 DOF per hand. Use Xacro macros to reduce code duplication.

4. Research and implement a more realistic URDF model for a humanoid robot, considering actual human joint ranges of motion and realistic inertial properties.

## References

Chitta, S., Marder-Eppstein, E., & Prats, M. (2010). Automatic generation of collision geometries for ROS packages. *IEEE International Conference on Robotics and Automation*, 1541-1546.

Koenig, N., & Howard, A. (2004). Design and use paradigms for Gazebo, an open-source multi-robot simulator. *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2149-2154.

Quigley, M., Gerkey, B., & Smart, W. D. (2015). Programming robots with ROS: a practical introduction to the Robot Operating System. *O'Reilly Media*.

Wheeler, R. M. (2008). URDF: Unified Robot Description Format. *ROS Wiki*. Retrieved from http://wiki.ros.org/urdf