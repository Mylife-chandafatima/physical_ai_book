---
description: "Implementation guide for Topics and Messaging in the Robotic Nervous System module"
---

# Topics and Messaging Implementation

**Save this file as**: `specify/implement/module-1/topics-messaging.md`

## Overview

Topics in ROS 2 implement a publish-subscribe communication pattern where nodes publish messages to topics and other nodes subscribe to receive those messages. This asynchronous communication model is ideal for sensor data streams, where multiple consumers need access to the same information simultaneously.

In humanoid robotics, topics commonly carry:
- Sensor data (IMU readings, camera images, force/torque measurements)
- Joint states (positions, velocities, efforts)
- Robot states (battery level, temperature, operational status)
- Control commands (desired joint positions, walking gaits)
- Environmental information (obstacle locations, map data)

## Publisher Implementation

Messages are defined using IDL (Interface Definition Language) and come in predefined types or custom definitions. Common message types for humanoid robots include:

- sensor_msgs/JointState: Contains joint positions, velocities, and efforts
- sensor_msgs/Image: Camera image data
- geometry_msgs/Twist: Velocity commands
- std_msgs/Float64: Scalar values
- Custom messages for humanoid-specific data

Here's an example publisher implementation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # Timer for publishing at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_joint_states)
        self.i = 0

    def publish_joint_states(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['left_hip', 'right_hip', 'left_knee', 'right_knee']

        # Generate oscillating joint positions
        msg.position = [
            math.sin(self.i * 0.1),
            math.cos(self.i * 0.1),
            math.sin(self.i * 0.1 + math.pi/2),
            math.cos(self.i * 0.1 + math.pi/2)
        ]

        # Add velocity and effort values
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)

        self.publisher.publish(msg)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    publisher = JointStatePublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Subscriber Implementation

Subscribers process incoming messages asynchronously, allowing for real-time response to sensor data or control commands. The quality of service (QoS) settings allow fine-tuning of reliability and performance characteristics for different types of data.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        
        # Create subscription to joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10  # QoS profile
        )
        self.subscription  # prevent unused variable warning

    def joint_state_callback(self, msg):
        # Process the received joint state message
        self.get_logger().info(f'Received joint states: {len(msg.name)} joints')
        for i, name in enumerate(msg.name):
            self.get_logger().info(f'  {name}: pos={msg.position[i]:.2f}, vel={msg.velocity[i]:.2f}')

def main(args=None):
    rclpy.init(args=args)
    subscriber = JointStateSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Settings

Quality of Service settings allow customization of message delivery guarantees:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

class QoSPublisher(Node):
    def __init__(self):
        super().__init__('qos_publisher')
        
        # Create a custom QoS profile for reliable delivery
        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        self.publisher = self.create_publisher(String, 'qos_topic', qos_profile)
        
        # Create another publisher with best-effort delivery for sensor data
        sensor_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        self.sensor_publisher = self.create_publisher(SensorMsg, 'sensor_topic', sensor_qos)
```

## Custom Message Types

You can define custom message types in ROS 2. Create a file called `CustomJointState.msg`:

```
# Custom extended joint state message
float64[] position
float64[] velocity
float64[] effort
string[] name
float64[] temperature  # Additional field for temperature
float64[] health_status  # Additional field for health status
```

Then use it in your nodes:

```python
from module1_pkg.msg import CustomJointState  # Assuming you created this message type

class CustomPublisher(Node):
    def __init__(self):
        super().__init__('custom_publisher')
        self.publisher = self.create_publisher(CustomJointState, 'custom_joint_states', 10)

    def publish_custom_state(self, positions, velocities, efforts, names, temperatures, health_status):
        msg = CustomJointState()
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts
        msg.name = names
        msg.temperature = temperatures
        msg.health_status = health_status
        
        self.publisher.publish(msg)
```

## Exercise 2: Sensor Fusion Node

Implement a sensor fusion node that subscribes to multiple sensor topics (IMU and joint states) and publishes a combined robot state message. Students will learn to synchronize data from different sources and handle timing considerations in distributed systems.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu_data',
            self.imu_callback,
            10
        )
        
        # Publisher for combined robot state
        self.state_pub = self.create_publisher(Float64MultiArray, '/robot_state', 10)
        
        # Store latest sensor data
        self.latest_joint_state = None
        self.latest_imu_data = None
        
        # Timer to publish combined state
        self.timer = self.create_timer(0.02, self.publish_combined_state)  # 50 Hz

    def joint_callback(self, msg):
        self.latest_joint_state = msg

    def imu_callback(self, msg):
        self.latest_imu_data = msg

    def publish_combined_state(self):
        if self.latest_joint_state and self.latest_imu_data:
            # Combine joint and IMU data into a single message
            combined_msg = Float64MultiArray()
            
            # Add joint positions
            combined_msg.data.extend(self.latest_joint_state.position)
            
            # Add IMU orientation
            combined_msg.data.extend([
                self.latest_imu_data.orientation.x,
                self.latest_imu_data.orientation.y,
                self.latest_imu_data.orientation.z,
                self.latest_imu_data.orientation.w
            ])
            
            # Add IMU angular velocity
            combined_msg.data.extend([
                self.latest_imu_data.angular_velocity.x,
                self.latest_imu_data.angular_velocity.y,
                self.latest_imu_data.angular_velocity.z
            ])
            
            self.state_pub.publish(combined_msg)
            self.get_logger().debug(f'Published combined state with {len(combined_msg.data)} values')

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

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

## Message Synchronization

For applications requiring synchronized data from multiple sources, you can use message filters:

```python
from message_filters import ApproximateTimeSynchronizer, Subscriber

class SynchronizedNode(Node):
    def __init__(self):
        super().__init__('synchronized_node')
        
        # Create subscribers
        joint_sub = Subscriber(self, JointState, '/joint_states')
        imu_sub = Subscriber(self, Imu, '/imu_data')
        
        # Synchronize messages with a time tolerance of 0.1 seconds
        ats = ApproximateTimeSynchronizer(
            [joint_sub, imu_sub], 
            queue_size=10, 
            slop=0.1
        )
        ats.registerCallback(self.sync_callback)

    def sync_callback(self, joint_msg, imu_msg):
        # Process synchronized joint and IMU data
        self.get_logger().info(f'Synchronized data: {len(joint_msg.position)} joints, IMU orientation: {imu_msg.orientation}')
```

## Summary

Topics and messaging form the backbone of ROS 2 communication systems. Understanding how to properly implement publishers, subscribers, and manage message synchronization is crucial for building responsive and reliable humanoid robot systems. Quality of Service settings allow tuning the communication behavior based on the specific requirements of different data types.