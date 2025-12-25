---
title: Bridging Python Agents to ROS Controllers
sidebar_label: Bridging Python Agents
---

# Save as docs/module-1/bridging-python-ros.md

# Bridging Python Agents to ROS Controllers using rclpy

## Learning Outcomes
By the end of this module, you will be able to:
- Integrate Python-based AI agents with ROS 2 controllers
- Create custom message types for agent-controller communication
- Implement bidirectional communication between agents and controllers
- Design robust interfaces for real-time robot control
- Debug and optimize agent-controller communication

## Introduction

Modern robotics increasingly relies on sophisticated AI agents for high-level decision making, planning, and control. These agents are often implemented in Python using popular machine learning libraries like TensorFlow, PyTorch, or scikit-learn. However, robots typically use ROS (Robot Operating System) for low-level control and coordination of hardware components.

This module focuses on bridging Python-based AI agents with ROS controllers using rclpy, the Python client library for ROS 2. This integration enables AI agents to interact with robotic hardware while leveraging ROS's communication infrastructure.

## Architecture of Agent-Controller Integration

The integration between AI agents and ROS controllers involves several key components:

1. **AI Agent**: Implements high-level reasoning, planning, and decision-making
2. **ROS Bridge**: Facilitates communication between the agent and ROS ecosystem
3. **ROS Controllers**: Handle low-level hardware control and real-time execution
4. **Robot Hardware**: Physical actuators, sensors, and other components

### Communication Patterns

The communication between agents and controllers typically follows these patterns:

- **Request-Response**: Agent requests an action, controller executes and reports results
- **Asynchronous Updates**: Controller continuously publishes sensor data to agent
- **State Synchronization**: Agent and controller maintain consistent state representations

## Implementing Python AI Agents with rclpy

### Basic Integration Pattern

Here's a basic pattern for integrating a Python AI agent with ROS controllers:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import JointState
import numpy as np
import tensorflow as tf  # Example AI library

class AIControllerBridge(Node):
    def __init__(self):
        super().__init__('ai_controller_bridge')
        
        # Publishers for sending commands to controllers
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Subscribers for receiving sensor data from controllers
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        
        # Initialize AI agent
        self.ai_agent = self.initialize_agent()
        
        # Timer for periodic AI decision making
        self.timer = self.create_timer(0.1, self.ai_decision_loop)
        
        # Robot state
        self.current_joint_states = None
        
    def initialize_agent(self):
        """Initialize and return the AI agent"""
        # This could be a neural network, rule-based system, etc.
        # For example, a simple neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(7, activation='linear')  # 7 joint commands
        ])
        return model
    
    def joint_state_callback(self, msg):
        """Callback for receiving joint state updates"""
        self.current_joint_states = msg
        self.get_logger().info(f'Received joint states: {msg.position}')
    
    def ai_decision_loop(self):
        """Main AI decision-making loop"""
        if self.current_joint_states is not None:
            # Prepare state for AI agent
            state_vector = np.array(self.current_joint_states.position)
            
            # Get action from AI agent
            action = self.ai_agent.predict(np.expand_dims(state_vector, axis=0))
            
            # Send command to robot
            self.send_joint_commands(action[0])
    
    def send_joint_commands(self, joint_commands):
        """Send joint commands to the robot controller"""
        msg = JointState()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        msg.position = joint_commands.tolist()
        self.joint_command_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ai_bridge = AIControllerBridge()
    
    try:
        rclpy.spin(ai_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        ai_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Custom Message Types

For complex agent-controller communication, custom message types may be necessary. Here's how to define and use custom messages:

### Creating Custom Message Definition

Create a file `ActionCommand.msg`:

```
# Custom message for AI agent commands
string action_type  # "move", "grasp", "rotate", etc.
float64[] parameters  # Action parameters
float64 priority  # Execution priority
string target_object  # Target for the action
```

### Using Custom Messages in Python

```python
from your_package.msg import ActionCommand  # Generated from .msg file
import rclpy
from rclpy.node import Node

class CustomMessageBridge(Node):
    def __init__(self):
        super().__init__('custom_message_bridge')
        self.action_publisher = self.create_publisher(ActionCommand, 'action_commands', 10)
        
    def send_action_command(self, action_type, parameters, priority=1.0, target_object=""):
        """Send a custom action command to the robot"""
        msg = ActionCommand()
        msg.action_type = action_type
        msg.parameters = parameters
        msg.priority = priority
        msg.target_object = target_object
        
        self.action_publisher.publish(msg)
        self.get_logger().info(f'Sent action: {action_type} with params: {parameters}')
```

## Advanced Integration Patterns

### Asynchronous Agent Processing

For computationally expensive AI operations, it's important to avoid blocking the ROS communication:

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class AsyncAIBridge(Node):
    def __init__(self):
        super().__init__('async_ai_bridge')
        
        # Publishers and subscribers
        self.result_publisher = self.create_publisher(String, 'ai_results', 10)
        
        # Thread pool for AI processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Store latest sensor data
        self.latest_sensor_data = None
        
        # Timer for checking AI results
        self.result_timer = self.create_timer(0.05, self.check_ai_results)
        
        # Future for ongoing AI computation
        self.ai_future = None
        
    def process_sensor_data_async(self, sensor_data):
        """Process sensor data asynchronously"""
        if self.ai_future is None or self.ai_future.done():
            self.latest_sensor_data = sensor_data
            self.ai_future = self.executor.submit(self.ai_computation, sensor_data)
    
    def ai_computation(self, sensor_data):
        """Heavy AI computation (runs in separate thread)"""
        # Simulate heavy computation
        import time
        time.sleep(0.1)  # Simulated AI processing time
        
        # Return AI result
        return f"Processed data: {sensor_data}"
    
    def check_ai_results(self):
        """Check if AI computation is complete"""
        if self.ai_future and self.ai_future.done():
            result = self.ai_future.result()
            self.publish_result(result)
            self.ai_future = None
    
    def publish_result(self, result):
        """Publish AI result to ROS topic"""
        msg = String()
        msg.data = result
        self.result_publisher.publish(msg)
```

### State Management and Synchronization

Proper state management is crucial for reliable agent-controller interaction:

```python
from collections import deque
import time

class StateSynchronizedBridge(Node):
    def __init__(self):
        super().__init__('state_sync_bridge')
        
        # State history for temporal consistency
        self.state_history = deque(maxlen=10)
        
        # Timestamp synchronization
        self.last_update_time = time.time()
        
        # State validation
        self.state_valid = True
        
        # Publishers and subscribers
        self.joint_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        
    def joint_state_callback(self, msg):
        """Process joint state with validation and timestamping"""
        current_time = time.time()
        
        # Validate state
        if not self.validate_joint_state(msg):
            self.get_logger().warn('Invalid joint state received')
            self.state_valid = False
            return
        
        # Add to history with timestamp
        state_entry = {
            'state': msg,
            'timestamp': current_time,
            'valid': True
        }
        
        self.state_history.append(state_entry)
        self.last_update_time = current_time
        self.state_valid = True
        
        # Trigger AI processing
        self.process_state_for_ai(state_entry)
    
    def validate_joint_state(self, joint_state):
        """Validate joint state for physical plausibility"""
        # Check for NaN values
        if any(np.isnan(pos) for pos in joint_state.position):
            return False
        
        # Check position limits (example: -2π to 2π)
        if any(abs(pos) > 2 * np.pi for pos in joint_state.position):
            return False
        
        return True
    
    def process_state_for_ai(self, state_entry):
        """Process validated state for AI agent"""
        # Convert ROS message to AI-friendly format
        ai_state = self.convert_to_ai_format(state_entry['state'])
        
        # Process with AI agent
        ai_action = self.run_ai_agent(ai_state)
        
        # Send to controller
        self.send_to_controller(ai_action)
    
    def convert_to_ai_format(self, joint_state):
        """Convert ROS joint state to AI-friendly format"""
        return {
            'positions': list(joint_state.position),
            'velocities': list(joint_state.velocity),
            'effort': list(joint_state.effort),
            'timestamp': time.time()
        }
    
    def run_ai_agent(self, state):
        """Run AI agent on the state"""
        # Placeholder for actual AI agent
        return {'action': 'move', 'parameters': [0.1, 0.2, 0.3]}
    
    def send_to_controller(self, action):
        """Send action to robot controller"""
        # Implementation depends on controller interface
        pass
```

## Practical Example: Reinforcement Learning Agent

Here's a complete example of integrating a reinforcement learning agent with ROS:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import numpy as np
import torch
import torch.nn as nn

class RLAgentBridge(Node):
    def __init__(self):
        super().__init__('rl_agent_bridge')
        
        # Subscribers for robot state
        self.joint_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.image_subscriber = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.laser_subscriber = self.create_subscription(
            Float32, 'laser_distance', self.laser_callback, 10)
        
        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Robot state variables
        self.joint_positions = np.zeros(7)
        self.camera_image = None
        self.distance = 1.0  # Default distance
        
        # Initialize RL agent
        self.rl_agent = self.initialize_rl_agent()
        
        # Timer for RL decision making
        self.rl_timer = self.create_timer(0.2, self.rl_decision_step)
        
    def initialize_rl_agent(self):
        """Initialize a simple neural network for RL"""
        class SimpleActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 64)  # Input: joint positions (7) + distance (1) + other (2)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 2)  # Output: linear and angular velocity
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.tanh(self.fc3(x))  # Output between -1 and 1
                return x
        
        return SimpleActor()
    
    def joint_callback(self, msg):
        """Update joint state"""
        self.joint_positions = np.array(msg.position[:7])  # Take first 7 joints
    
    def image_callback(self, msg):
        """Process camera image (simplified)"""
        # In a real implementation, you'd convert ROS image to numpy array
        # For now, we'll just note that we received an image
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')
    
    def laser_callback(self, msg):
        """Update distance measurement"""
        self.distance = msg.data
    
    def rl_decision_step(self):
        """Main RL decision step"""
        if self.camera_image is not None:
            # Prepare state vector for RL agent
            state_vector = np.concatenate([
                self.joint_positions,
                [self.distance],
                [0.0, 0.0]  # Placeholder for other state variables
            ])
            
            # Convert to tensor and get action
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            with torch.no_grad():
                action_tensor = self.rl_agent(state_tensor)
                action = action_tensor.numpy()[0]
            
            # Convert action to robot command
            cmd_msg = Twist()
            cmd_msg.linear.x = float(action[0])  # Linear velocity
            cmd_msg.angular.z = float(action[1])  # Angular velocity
            
            # Publish command
            self.cmd_publisher.publish(cmd_msg)
            
            self.get_logger().info(f'RL Action: linear={action[0]:.2f}, angular={action[1]:.2f}')

def main(args=None):
    rclpy.init(args=args)
    rl_bridge = RLAgentBridge()
    
    try:
        rclpy.spin(rl_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        rl_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Agent-Controller Integration

### 1. Error Handling and Fallbacks

```python
class RobustAIBridge(Node):
    def __init__(self):
        super().__init__('robust_ai_bridge')
        
        # Set up error handling
        self.fallback_active = False
        self.last_known_good_state = None
        
    def safe_ai_execution(self, state):
        """Execute AI with error handling"""
        try:
            # Validate input state
            if not self.validate_state(state):
                raise ValueError("Invalid state for AI")
            
            # Execute AI agent
            action = self.ai_agent.predict(state)
            
            # Validate output action
            action = self.validate_action(action)
            
            return action
        except Exception as e:
            self.get_logger().error(f'AI execution error: {e}')
            return self.get_fallback_action()
    
    def validate_state(self, state):
        """Validate state for AI processing"""
        # Implementation of state validation
        return True
    
    def validate_action(self, action):
        """Validate action before sending to controller"""
        # Implementation of action validation
        return action
    
    def get_fallback_action(self):
        """Return safe fallback action"""
        # Implementation of safe fallback behavior
        return np.zeros(7)  # Zero joint commands as fallback
```

### 2. Performance Optimization

```python
class OptimizedAIBridge(Node):
    def __init__(self):
        super().__init__('optimized_ai_bridge')
        
        # Pre-allocate arrays to avoid memory allocation during execution
        self.pre_allocated_state = np.zeros(20)
        self.pre_allocated_action = np.zeros(7)
        
        # Use efficient data structures
        self.state_buffer = deque(maxlen=5)
        
        # Timer for optimized execution
        self.ai_timer = self.create_timer(0.05, self.optimized_ai_step)
    
    def optimized_ai_step(self):
        """Optimized AI decision step"""
        # Use pre-allocated arrays
        np.copyto(self.pre_allocated_state, self.get_current_state())
        
        # Efficient AI processing
        self.ai_agent.predict_inplace(self.pre_allocated_state, self.pre_allocated_action)
        
        # Publish result
        self.publish_action(self.pre_allocated_action)
```

## Exercises

1. Implement a Python AI agent that learns to control a simulated robot arm to reach target positions. Use the bridge pattern to connect your agent to ROS controllers.

2. Create a custom message type for representing robot goals and implement a bridge that converts high-level goals (e.g., "move to kitchen") to low-level joint commands.

3. Design and implement a fault-tolerant agent-controller interface that can handle sensor failures and maintain safe robot operation.

4. Research and implement a method for real-time performance monitoring of the agent-controller communication. What metrics would you track?

## References

Macenski, S., Woodall, W., & Faust, J. (2019). The design and implementation of ROS 2. *IEEE International Conference on Simulation, Modeling, and Programming for Autonomous Robots*.

Quigley, M., Gerkey, B., & Smart, W. D. (2015). Programming robots with ROS: a practical introduction to the Robot Operating System. *O'Reilly Media*.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

Zhang, T., Yang, H., Liu, C., Wei, W., Liu, Y., & Wang, Q. (2020). Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. *IEEE International Conference on Robotics and Automation*.