---
title: ROS 2 Nodes, Topics, Services
sidebar_label: Nodes, Topics, Services
---

# Save as docs/module-1/nodes-topics-services.md

# ROS 2 Nodes, Topics, Services

## Learning Outcomes
By the end of this module, you will be able to:
- Explain the fundamental concepts of ROS 2 architecture
- Create and implement ROS 2 nodes
- Design communication patterns using topics and services
- Implement message passing between different components
- Debug and monitor ROS 2 communication

## Introduction to ROS 2 Architecture

Robot Operating System 2 (ROS 2) provides a flexible framework for writing robot software. Unlike traditional operating systems, ROS 2 is middleware that provides services designed for a heterogeneous computer cluster. It includes hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

### Core Concepts

**Nodes**: A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are processes that perform computation. ROS 2 is designed to be used by multiple nodes in a single system, potentially on multiple machines connected together.

**Topics**: Topics are named buses over which nodes exchange messages. A node can publish messages to a topic or subscribe to messages from a topic. Topic-based communication is asynchronous and follows a publish/subscribe pattern.

**Services**: Services provide a request/response communication pattern. A service client sends a request message to a service server, which responds with a result message.

## Creating ROS 2 Nodes

A ROS 2 node is a program that uses the ROS 2 client library. Here's a basic example of a ROS 2 node in Python:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalNode()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle

Each ROS 2 node goes through several lifecycle states:
1. **Unconfigured**: Initial state after creation
2. **Inactive**: Configured but not active
3. **Active**: Fully operational
4. **Finalized**: Cleaned up and ready for destruction

## Topics and Publishers/Subscribers

### Topic Communication

Topics enable asynchronous communication between nodes using a publish/subscribe pattern. This pattern allows for loose coupling between publishers and subscribers.

### Creating Publishers

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Creating Subscribers

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

## Services

Services provide synchronous request/response communication. Here's how to implement a service server:

### Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Quality of Service (QoS) Settings

QoS settings allow you to specify how messages are delivered between publishers and subscribers:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Create a QoS profile with custom settings
qos_profile = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Use the QoS profile when creating a publisher
publisher = node.create_publisher(String, 'topic', qos_profile)
```

## Practical Exercise: Implement a Sensor Node

Create a ROS 2 node that simulates a distance sensor publishing readings to a topic:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class DistanceSensorNode(Node):

    def __init__(self):
        super().__init__('distance_sensor')
        self.publisher_ = self.create_publisher(Float32, 'distance', 10)
        self.timer = self.create_timer(0.1, self.publish_distance)
        
    def publish_distance(self):
        msg = Float32()
        # Simulate sensor reading with some noise
        msg.data = random.uniform(0.1, 10.0)  # Distance in meters
        self.publisher_.publish(msg)
        self.get_logger().info(f'Distance: {msg.data:.2f} m')

def main(args=None):
    rclpy.init(args=args)
    sensor_node = DistanceSensorNode()
    
    try:
        rclpy.spin(sensor_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. Create a ROS 2 package called `robot_controller` with a node that publishes joint positions to a topic and another node that subscribes to these positions and prints them to the console.

2. Implement a service that calculates the Euclidean distance between two 3D points. Create both the service server and client.

3. Design a communication architecture for a humanoid robot with the following components: vision system, motion controller, and path planner. Specify the topics and services each component would use.

4. Research and explain the difference between ROS 1 and ROS 2 in terms of communication architecture. What are the advantages of ROS 2's DDS-based approach?

## References

Faust, J., Tola, D., Ichnowski, J., Derenick, J., Macenski, S., Marzouk, M., ... & Satici, A. C. (2018). The design and implementation of ROS 2. *IEEE International Conference on Simulation, Modeling, and Programming for Autonomous Robots*, 1-8.

Quigley, M., Conley, K., & Gerkey, B. (2009). ROS: an open-source Robot Operating System. *ICRA Workshop on Open Source Software*, 3(2), 5.

ROS.org. (2023). *ROS 2 Documentation*. Retrieved from https://docs.ros.org/en/humble/