# Save this file as specify/implement/module-3/nav2-path-planning.md

# Nav2 Path Planning for Bipedal Humanoids

## Overview

Navigation2 (Nav2) is the state-of-the-art navigation stack for ROS 2, designed for autonomous navigation of mobile robots. For bipedal humanoids, special considerations are needed for balance, step planning, and dynamic stability.

## Nav2 Architecture for Humanoids

Nav2 consists of several key components that need to be adapted for bipedal humanoids:

```
Goal → Global Planner → Local Planner → Controller → Robot
                ↓
        Behavior Tree Executor
```

For bipedal robots, these components require specific modifications to account for:
- Balance constraints
- Step planning
- Dynamic stability
- Terrain analysis

## Nav2 Configuration for Humanoids

### Basic Configuration File

```yaml
# nav2_params.yaml for humanoid robot
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 10.0
    laser_min_range: -1.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_footprint
    odom_topic: /odom
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]
    
    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.3
      vx_max: 0.4
      vx_min: -0.2
      vy_max: 0.2
      wz_max: 0.4
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      rot_stopped_velocity: 0.25
      control_duration: 0.1
      prediction_horizon: 2.5
      regularization_lambda: 0.0
      trajector_replacer: "traj_sampler"
      trajectory_visualizer: "traj_visualizer"
      costmap_topic: "local_costmap/costmap_raw"
      footprint_topic: "local_costmap/published_footprint"
      velocity_topic: "cmd_vel"
      transform_tolerance: 0.1
      use_astar: false
      force_initial_rotation: false

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_footprint
      use_sim_time: True
      rolling_window: true
      width: 10
      height: 10
      resolution: 0.05
      # Humanoid-specific footprint
      footprint: "[[-0.3, -0.15], [-0.3, 0.15], [0.3, 0.15], [0.3, -0.15]]"
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
```

## Bipedal-Specific Considerations

### Step Planning
For bipedal robots, navigation must account for foot placement:

```python
# Example step planning implementation
import numpy as np
from geometry_msgs.msg import PoseStamped

class StepPlanner:
    def __init__(self):
        self.step_size = 0.3  # meters
        self.max_step_height = 0.1  # meters
        self.support_polygon = np.array([[-0.1, -0.05], [-0.1, 0.05], [0.1, 0.05], [0.1, -0.05]])
    
    def plan_steps(self, path):
        """Convert continuous path to discrete steps for bipedal robot"""
        steps = []
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            # Calculate distance between points
            dist = np.sqrt((end_point[0] - start_point[0])**2 + 
                          (end_point[1] - start_point[1])**2)
            
            # Calculate number of steps needed
            num_steps = int(dist / self.step_size)
            
            # Generate intermediate steps
            for j in range(num_steps):
                step_x = start_point[0] + (end_point[0] - start_point[0]) * j / num_steps
                step_y = start_point[1] + (end_point[1] - start_point[1]) * j / num_steps
                steps.append([step_x, step_y])
        
        return steps
```

### Balance Constraints
```python
# Balance constraint implementation
class BalanceConstraint:
    def __init__(self):
        self.com_height = 0.8  # Center of mass height
        self.max_com_offset = 0.05  # Max CoM offset from support polygon
    
    def is_balanced(self, com_position, support_polygon):
        """Check if the robot is balanced based on CoM position"""
        # Simple check if CoM is within support polygon bounds
        min_x = min([point[0] for point in support_polygon])
        max_x = max([point[0] for point in support_polygon])
        min_y = min([point[1] for point in support_polygon])
        max_y = max([point[1] for point in support_polygon])
        
        return (min_x - self.max_com_offset <= com_position[0] <= max_x + self.max_com_offset and
                min_y - self.max_com_offset <= com_position[1] <= max_y + self.max_com_offset)
```

## Custom Plugins for Humanoid Navigation

### Creating a Humanoid-Specific Controller
```cpp
// Example C++ controller plugin for bipedal robots
#include <nav2_core/controller.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/msg/twist.hpp>

class HumanoidController : public nav2_core::Controller
{
public:
    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros,
        const std::shared_ptr<nav2_core::LifecycleNode> &) override
    {
        node_ = parent.lock();
        name_ = name;
        costmap_ros_ = costmap_ros;
        tf_ = costmap_ros_->getTfBuffer();
        
        // Initialize humanoid-specific parameters
        node_->get_parameter_or(name_ + ".step_size", step_size_, 0.3);
        node_->get_parameter_or(name_ + ".max_step_height", max_step_height_, 0.1);
    }
    
    geometry_msgs::msg::Twist computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity,
        nav2_core::GoalChecker * goal_checker) override
    {
        geometry_msgs::msg::Twist cmd_vel;
        
        // Implement humanoid-specific velocity computation
        // Consider balance constraints and step planning
        
        return cmd_vel;
    }
    
private:
    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    tf2_ros::Buffer * tf_;
    
    double step_size_;
    double max_step_height_;
};
```

## Practical Exercise 4.1: Humanoid Navigation Implementation

Configure Nav2 for a bipedal humanoid robot in simulation:

1. Set up the Nav2 configuration file with humanoid-specific parameters
2. Implement custom plugins for step planning and balance-aware navigation
3. Test the system in various environments (flat terrain, stairs, uneven surfaces)
4. Evaluate performance metrics:
   - Navigation success rate
   - Balance maintenance
   - Path efficiency
   - Computation time
5. Document your implementation approach and results

Create a navigation test course with obstacles and evaluate your humanoid navigation system's performance.

## References

ROS.org. (2023). *Navigation2 Documentation*. https://navigation.ros.org/

Kuffner, J., & LaValle, S. M. (2000). RRT-connect: An efficient approach to single-query path planning. *Proceedings of ICRA*, 995-1001.

Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance. *IEEE Robotics & Automation Magazine*, 4(1), 23-33.