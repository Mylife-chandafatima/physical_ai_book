# Save this file as specify/implement/module-2/gazebo-simulation-setup.md

# Gazebo Simulation Environment Setup

## Overview

Gazebo is a 3D dynamic simulator widely used in robotics for simulating robots in realistic indoor and outdoor environments. It provides accurate physics simulation, high-quality graphics, and convenient programmatic interfaces.

## Installation and Configuration

To set up Gazebo for robotics simulation:

```bash
# Install Gazebo (example for Ubuntu with ROS 2 Humble)
sudo apt-get update
sudo apt-get install gazebo libgazebo-dev
sudo apt-get install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-gazebo-ros

# Verify installation
gazebo --version
```

## Basic Environment Configuration

Create a simple world file to define your simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sun light source -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Custom directional light -->
    <light name="main_light" type="directional">
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
    
    <!-- Optional: Add objects to the environment -->
    <model name="box1">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 0 0 1</specular>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Launching Gazebo with Custom World

Create a launch file to start Gazebo with your custom world:

```python
# launch/gazebo_world.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    world_path = PathJoinSubstitution([
        FindPackageShare("your_robot_description"),
        "worlds",
        "your_custom_world.sdf"
    ])

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_init.so', world_path],
            output='screen'
        )
    ])
```

## Practical Exercise 2.1: Custom Environment Creation

Create a Gazebo environment with the following specifications:
1. Custom terrain with at least 3 different objects
2. Multiple light sources with different properties
3. Different materials applied to objects
4. Verify the environment loads correctly in Gazebo

Document your world file and explain the design choices you made for each element.

## References

Open Robotics. (2023). *Gazebo Documentation*. http://gazebosim.org/

ROS.org. (2023). *Gazebo with ROS 2 Tutorials*. https://classic.gazebosim.org/tutorials?cat=connect_ros