---
description: "Implementation guide for ROS 2 Nodes in the Robotic Nervous System module"
---

# ROS 2 Nodes Implementation

**Save this file as**: `specify/implement/module-1/ros2-nodes.md`

## Overview

ROS 2 nodes are the fundamental building blocks of ROS 2 applications, representing individual processes that perform specific functions within the robotic system. A humanoid robot typically consists of numerous nodes managing different aspects such as sensor processing, motion planning, control algorithms, and user interfaces.

Each node operates independently but communicates with other nodes through topics, services, and actions. This modular approach enhances system reliability since the failure of one node doesn't necessarily affect others. In humanoid robotics, this means that if the vision processing node fails, the walking controller can continue functioning.

## Node Structure and Lifecycle

Nodes in ROS 2 have a well-defined structure using the rclpy library. Here's the basic structure of a ROS 2 node:

```python
import rclpy
from rclpy.node import Node

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.get_logger().info('Joint Controller initialized')

        # Initialize joint position variables
        self.joint_positions = {
            'left_hip': 0.0,
            'right_hip': 0.0,
            'left_knee': 0.0,
            'right_knee': 0.0
        }

        # Create timer for periodic control updates
        self.timer = self.create_timer(0.1, self.control_callback)

    def control_callback(self):
        # Control loop implementation
        self.get_logger().debug(f'Current positions: {self.joint_positions}')

def main(args=None):
    rclpy.init(args=args)
    joint_controller = JointController()

    try:
        rclpy.spin(joint_controller)
    except KeyboardInterrupt:
        pass
    finally:
        joint_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Publishers and Subscribers

Nodes can create publishers to send messages and subscribers to receive messages:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()
    
    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Node Parameters and Configuration

Nodes can use parameters for configuration:

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        
        # Declare parameters with default values and descriptions
        self.declare_parameter('control_frequency', 10, 
                             ParameterDescriptor(description='Control loop frequency in Hz'))
        self.declare_parameter('default_position', 0.0,
                             ParameterDescriptor(description='Default joint position'))
        
        # Access parameters
        freq = self.get_parameter('control_frequency').value
        default_pos = self.get_parameter('default_position').value
        
        self.get_logger().info(f'Frequency: {freq}, Default position: {default_pos}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 1: Basic Node Implementation

Create a joint controller node that simulates controlling a humanoid robot's hip joints. The node should maintain internal state for left and right hip positions, accept commands via a service call, and publish updated joint states.

**Steps:**
1. Create a new Python file called `joint_controller.py`
2. Implement a node that maintains joint position state
3. Add a publisher for joint states
4. Include proper logging and shutdown procedures

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.get_logger().info('Joint Controller initialized')
        
        # Initialize joint position variables with default values
        self.joint_positions = {
            'left_hip': 0.0,
            'right_hip': 0.0,
            'left_knee': 0.0,
            'right_knee': 0.0
        }
        
        # Create publishers for each joint
        self.joint_publishers = {}
        for joint_name in self.joint_positions.keys():
            topic_name = f'/{joint_name}_position'
            self.joint_publishers[joint_name] = self.create_publisher(Float64, topic_name, 10)
        
        # Create timer for periodic control updates
        self.timer = self.create_timer(0.1, self.control_callback)
        
    def control_callback(self):
        # Control loop implementation
        self.get_logger().debug(f'Current positions: {self.joint_positions}')
        
        # Publish current joint positions
        for joint_name, position in self.joint_positions.items():
            msg = Float64()
            msg.data = position
            self.joint_publishers[joint_name].publish(msg)

def main(args=None):
    rclpy.init(args=args)
    joint_controller = JointController()

    try:
        rclpy.spin(joint_controller)
    except KeyboardInterrupt:
        pass
    finally:
        joint_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Node Design Principles for Humanoid Robotics

When designing nodes for humanoid robotics, consider these principles:

- **Single responsibility**: Each node should handle one specific aspect of robot functionality
- **State management**: Nodes maintain internal state relevant to their function
- **Resource efficiency**: Minimize computational overhead for real-time applications
- **Error handling**: Implement robust error recovery mechanisms
- **Logging**: Provide informative logging for debugging and monitoring

## Summary

ROS 2 nodes provide the foundation for distributed robotic systems. Understanding node structure, lifecycle, and communication patterns is essential for building humanoid robot control systems. The modular nature of nodes allows for robust and maintainable robot software architectures.