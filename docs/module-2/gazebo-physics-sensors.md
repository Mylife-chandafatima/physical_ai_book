---
title: Gazebo Physics Simulation and Sensors
sidebar_label: Gazebo Physics & Sensors
---

# Save as docs/module-2\gazebo-physics-sensors.md

# Gazebo: Physics Simulation and Sensor Integration

## Learning Outcomes
By the end of this module, you will be able to:
- Configure and optimize Gazebo physics engines for different robotic applications
- Implement realistic sensor models with appropriate noise characteristics
- Design complex simulation environments with multiple robots and objects
- Validate sensor data accuracy and adjust parameters for realism
- Troubleshoot common physics and sensor simulation issues

## Gazebo Physics Engine Fundamentals

Gazebo's physics simulation is powered by several backend engines, each with different strengths and characteristics. Understanding these engines is crucial for creating accurate simulations.

### Physics Engine Options

Gazebo supports multiple physics engines:

1. **ODE (Open Dynamics Engine)**: Default engine, good balance of speed and accuracy
2. **Bullet**: Good for complex collision detection
3. **Simbody**: High-accuracy simulation for complex articulated systems
4. **DART**: Dynamic Animation and Robotics Toolkit, good for complex contacts

### Physics Configuration

Here's how to configure physics parameters in a Gazebo world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="physics_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      
      <!-- ODE-specific parameters -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Models and environments -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Gravity and Environmental Forces

```xml
<world name="gravity_test">
  <gravity>0 0 -9.8</gravity>  <!-- Earth gravity -->
  
  <!-- For different environments -->
  <!-- Moon gravity: <gravity>0 0 -1.62</gravity> -->
  <!-- Mars gravity: <gravity>0 0 -3.71</gravity> -->
  
  <magnetic_field>6e-6 2.3e-5 -4.2e-5</magnetic_field>
  
  <atmosphere type="adiabatic">
    <temperature>288.15</temperature>
    <pressure>101325</pressure>
  </atmosphere>
</world>
```

## Collision Detection and Contact Modeling

### Collision Properties

```xml
<model name="robot_with_collisions">
  <link name="base_link">
    <collision name="collision">
      <pose>0 0 0 0 0 0</pose>
      <geometry>
        <mesh>
          <uri>model://my_robot/meshes/base_link.stl</uri>
        </mesh>
      </geometry>
      
      <!-- Surface properties -->
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
            <fdir1>0 0 0</fdir1>
            <slip1>0</slip1>
            <slip2>0</slip2>
          </ode>
          <torsional>
            <coefficient>1.0</coefficient>
            <use_patch_radius>1</use_patch_radius>
            <surface_radius>0.01</surface_radius>
            <patch_radius>0.01</patch_radius>
            <constraint>
              <stiffness>1e8</stiffness>
              <dissipation>1.0</dissipation>
            </constraint>
          </torsional>
        </friction>
        
        <bounce>
          <restitution_coefficient>0.1</restitution_coefficient>
          <threshold>100000</threshold>
        </bounce>
        
        <contact>
          <collide_without_contact>0</collide_without_contact>
          <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
          <collide_bitmask>15</collide_bitmask>
          <ode>
            <soft_cfm>0</soft_cfm>
            <soft_erp>0.2</soft_erp>
            <kp>1e10</kp>
            <kd>1</kd>
            <max_vel>100.0</max_vel>
            <min_depth>0.001</min_depth>
          </ode>
          <bullet>
            <kp>1e6</kp>
            <kd>1</kd>
            <max_vel>100.0</max_vel>
            <min_depth>0.001</min_depth>
            <split_impulse>1</split_impulse>
            <split_impulse_penetration_threshold>-0.01</split_impulse_penetration_threshold>
          </bullet>
        </contact>
      </surface>
    </collision>
  </link>
</model>
```

### Custom Physics Plugins

You can extend Gazebo's physics capabilities with custom plugins:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class CustomPhysicsPlugin : public WorldPlugin
  {
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      this->world = _world;
      
      // Connect to the pre-update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&CustomPhysicsPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Custom physics calculations
      // Example: Apply custom forces to objects
      for (auto &model : this->world->Models())
      {
        if (model->GetName() == "object_to_affect")
        {
          math::Vector3 force(0.1, 0, 0); // Apply small force in X direction
          model->GetLink("link")->AddForce(force);
        }
      }
    }

    private: physics::WorldPtr world;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_WORLD_PLUGIN(CustomPhysicsPlugin)
}
```

## Sensor Simulation in Gazebo

### Camera Sensors

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
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
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <camera_name>camera</camera_name>
    <image_topic_name>image_raw</image_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
    <frame_name>camera_frame</frame_name>
    <hack_baseline>0.07</hack_baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
  </plugin>
</sensor>
```

### LIDAR Sensors (Ray Sensor)

```xml
<sensor name="laser" type="ray">
  <always_on>true</always_on>
  <update_rate>40</update_rate>
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
      <min>0.10</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </ray>
  <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
    <topic_name>scan</topic_name>
    <frame_name>laser_frame</frame_name>
  </plugin>
</sensor>
```

### IMU Sensors

```xml
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.017</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.017</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.017</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <topicName>imu</topicName>
    <bodyName>imu_link</bodyName>
    <frameName>imu_frame</frameName>
    <serviceName>imu_service</serviceName>
    <gaussianNoise>0.0017</gaussianNoise>
    <updateRateHZ>100.0</updateRateHZ>
  </plugin>
</sensor>
```

### Force/Torque Sensors

```xml
<sensor name="ft_sensor" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin name="ft_sensor_controller" filename="libgazebo_ros_ft_sensor.so">
    <topicName>ft_sensor</topicName>
    <jointName>sensor_joint</jointName>
  </plugin>
</sensor>
```

## Advanced Sensor Modeling

### Multi-Camera Systems

```xml
<!-- Stereo camera setup -->
<sensor name="left_camera" type="camera">
  <pose>0.05 0.05 0 0 0 0</pose>
  <camera name="left">
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
  <plugin name="left_camera_controller" filename="libgazebo_ros_camera.so">
    <camera_name>stereo/left</camera_name>
    <image_topic_name>image_raw</image_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
    <frame_name>left_camera_frame</frame_name>
  </plugin>
</sensor>

<sensor name="right_camera" type="camera">
  <pose>0.05 -0.05 0 0 0 0</pose>
  <camera name="right">
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
  <plugin name="right_camera_controller" filename="libgazebo_ros_camera.so">
    <camera_name>stereo/right</camera_name>
    <image_topic_name>image_raw</image_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
    <frame_name>right_camera_frame</frame_name>
  </plugin>
</sensor>
```

### Custom Sensor Plugins

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <sensor_msgs/Range.h>

namespace gazebo
{
  class CustomRangeSensor : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Get the range sensor
      this->parentSensor = 
        std::dynamic_pointer_cast<sensors::RaySensor>(_sensor);
      
      if (!this->parentSensor)
      {
        gzerr << "CustomRangeSensor requires a Ray sensor as its parent\n";
        return;
      }
      
      // Connect to the sensor update event
      this->updateConnection = this->parentSensor->ConnectUpdated(
          std::bind(&CustomRangeSensor::OnUpdate, this));
      
      // Make sure the parent sensor is active
      this->parentSensor->SetActive(true);
    }

    public: void OnUpdate()
    {
      // Get range data from the ray sensor
      double range = this->parentSensor->Range(0);
      
      // Publish custom range message
      sensor_msgs::Range msg;
      msg.header.stamp = ros::Time::now();
      msg.header.frame_id = "range_sensor_frame";
      msg.radiation_type = sensor_msgs::Range::ULTRASOUND;
      msg.field_of_view = 0.1;  // Example field of view
      msg.min_range = 0.01;
      msg.max_range = 10.0;
      msg.range = range;
      
      // Publish the message (implementation depends on ROS setup)
      // range_pub_.publish(msg);
    }

    private: sensors::RaySensorPtr parentSensor;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomRangeSensor)
}
```

## Environment Modeling

### Creating Complex Environments

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="complex_environment">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
    
    <!-- Ground plane with texture -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Add sun -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Indoor environment -->
    <model name="room">
      <static>true</static>
      <link name="room_link">
        <!-- Walls -->
        <collision name="wall_1">
          <geometry>
            <box><size>0.2 10 3</size></box>
          </geometry>
        </collision>
        <visual name="wall_1_visual">
          <geometry>
            <box><size>0.2 10 3</size></box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/White</name>
            </script>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Furniture -->
    <model name="table">
      <pose>3 0 0 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box><size>1.5 0.8 0.75</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1.5 0.8 0.75</size></box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/Wood</name>
            </script>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1.0</iyy>
            <iyz>0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Objects for manipulation -->
    <model name="object_1">
      <pose>3.2 0.2 0.8 0 0 0</pose>
      <link name="object_link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>0.05</radius><length>0.2</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.05</radius><length>0.2</length></cylinder>
          </geometry>
          <material>
            <script>
              <name>Gazebo/Red</name>
            </script>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.001</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.001</iyy>
            <iyz>0</iyz>
            <izz>0.00125</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Performance Optimization

### Physics Optimization

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  
  <!-- Optimize solver parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>  <!-- Balance between accuracy and speed -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Sensor Optimization

For better performance with multiple sensors:

1. **Reduce update rates** for sensors that don't need high frequency
2. **Limit sensor resolution** where appropriate
3. **Use sensor noise models** that match real sensors
4. **Optimize rendering quality** settings

## Troubleshooting Common Issues

### Physics Issues

1. **Objects falling through surfaces**: Check collision geometries and surface parameters
2. **Unstable simulation**: Adjust solver parameters and time step
3. **Objects jittering**: Increase constraint parameters or adjust surface properties

### Sensor Issues

1. **No sensor data**: Verify plugin configuration and topic names
2. **Incorrect sensor readings**: Check coordinate frames and sensor placement
3. **Performance issues**: Reduce sensor resolution or update rates

## Exercises

1. Create a Gazebo simulation of a mobile robot navigating through a room with furniture. Implement realistic collision detection and sensor models.

2. Design a manipulation environment with multiple objects and implement a gripper with force/torque sensing capabilities.

3. Implement a custom physics plugin that simulates wind effects on lightweight objects in the environment.

4. Create a multi-robot simulation with realistic sensor models and communication between robots.

## References

Koenig, N., & Howard, A. (2004). Design and use paradigms for Gazebo, an open-source multi-robot simulator. *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2149-2154.

Open Robotics. (2023). *Gazebo Sensors Documentation*. Retrieved from http://gazebosim.org/tutorials?tut=sensor_noise

Coumans, E. (2013). Bullet physics engine. *GitHub Repository*. Retrieved from https://github.com/bulletphysics/bullet3

Smith, R., & Ekeberg, Ö. (2010). Simbody: Multibody simulation for biomedical research. *Proceedings of the ASME 2010 International Design Engineering Technical Conferences*.