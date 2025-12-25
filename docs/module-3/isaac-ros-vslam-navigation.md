---
title: Isaac ROS - VSLAM & Navigation
sidebar_label: Isaac ROS VSLAM & Navigation
---

# Save as docs/module-3/isaac-ros-vslam-navigation.md

# Isaac ROS: Visual SLAM and Navigation

## Learning Outcomes
By the end of this module, you will be able to:
- Implement Visual SLAM systems using Isaac ROS packages
- Configure and optimize navigation stacks for robot autonomy
- Integrate perception and navigation systems for seamless operation
- Deploy advanced perception pipelines using Isaac ROS
- Troubleshoot common issues in SLAM and navigation systems

## Introduction to Isaac ROS for Perception and Navigation

Isaac ROS provides a suite of hardware-accelerated packages specifically designed for robotics perception and navigation. These packages leverage NVIDIA's GPU computing capabilities to deliver real-time performance for complex perception tasks like Visual SLAM, object detection, and navigation.

### Key Isaac ROS Packages for Perception and Navigation

1. **Isaac ROS Visual SLAM**: GPU-accelerated visual SLAM
2. **Isaac ROS Stereo DNN**: Deep neural network-based stereo vision
3. **Isaac ROS Apriltag**: High-performance AprilTag detection
4. **Isaac ROS Point Cloud**: GPU-accelerated point cloud processing
5. **Isaac ROS Image Pipeline**: Optimized image processing pipeline

## Isaac ROS Visual SLAM

Visual SLAM (Simultaneous Localization and Mapping) is crucial for robot autonomy, allowing robots to understand their environment and navigate without prior maps.

### Visual SLAM Architecture

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from collections import deque

class IsaacVisualSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam_node')
        
        self.bridge = CvBridge()
        
        # SLAM state variables
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.keyframes = deque(maxlen=100)  # Store recent keyframes
        self.map_points = {}  # 3D map points
        
        # Feature detection parameters
        self.feature_detector = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31
        )
        
        # Feature matching parameters
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.knn_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.image_size = (640, 480)
        
        # Previous frame for tracking
        self.prev_gray = None
        self.prev_features = None
        
        # Subscribers
        self.left_image_sub = self.create_subscription(
            Image, '/camera/left/image_rect_color', 
            self.left_image_callback, 10)
        self.right_image_sub = self.create_subscription(
            Image, '/camera/right/image_rect_color', 
            self.right_image_callback, 10)
        self.left_info_sub = self.create_subscription(
            CameraInfo, '/camera/left/camera_info', 
            self.camera_info_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)
        self.keyframe_pub = self.create_publisher(Image, '/visual_slam/keyframe', 10)
        
        # SLAM parameters
        self.keyframe_threshold = 20  # Minimum distance for keyframe
        self.max_features = 1000
        self.min_matches = 20
        
        # Initialize
        self.initialized = False
        self.frame_count = 0
        
    def camera_info_callback(self, msg):
        """Store camera intrinsics"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)
            self.image_size = (msg.width, msg.height)
            self.get_logger().info(f'Camera intrinsics set: {self.camera_matrix}')
    
    def imu_callback(self, msg):
        """Handle IMU data for sensor fusion"""
        # Store IMU data for potential sensor fusion
        self.imu_data = {
            'angular_velocity': [msg.angular_velocity.x, 
                                msg.angular_velocity.y, 
                                msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x,
                                   msg.linear_acceleration.y,
                                   msg.linear_acceleration.z],
            'orientation': [msg.orientation.x, msg.orientation.y,
                           msg.orientation.z, msg.orientation.w],
            'timestamp': msg.header.stamp
        }
    
    def left_image_callback(self, msg):
        """Process left camera image for visual SLAM"""
        if self.camera_matrix is None:
            return
            
        # Convert ROS image to OpenCV
        current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            # Initialize first frame
            self.prev_gray = current_gray
            self.prev_features = self.extract_features(current_gray)
            self.initialized = True
            return
        
        # Track features between frames
        transformation = self.track_features(self.prev_gray, current_gray)
        
        if transformation is not None:
            # Update pose
            self.update_pose(transformation)
            
            # Check if we should create a keyframe
            if self.should_create_keyframe(transformation):
                self.create_keyframe(current_image, current_gray)
        
        # Update previous frame
        self.prev_gray = current_gray
        self.prev_features = self.extract_features(current_gray)
        
        self.frame_count += 1
        
        # Publish pose and odometry
        self.publish_pose_and_odometry(msg.header.stamp)
    
    def right_image_callback(self, msg):
        """Process right camera image for stereo processing"""
        # This would be used for stereo depth estimation
        # in a complete implementation
        pass
    
    def extract_features(self, gray_image):
        """Extract features from grayscale image"""
        keypoints = self.feature_detector.detect(gray_image, None)
        if keypoints:
            keypoints, descriptors = self.feature_detector.compute(gray_image, keypoints)
        else:
            descriptors = None
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
    
    def track_features(self, prev_gray, curr_gray):
        """Track features between consecutive frames"""
        # Detect features in current frame
        curr_features = self.extract_features(curr_gray)
        
        if self.prev_features['descriptors'] is None or curr_features['descriptors'] is None:
            return None
        
        # Match features between frames
        matches = self.knn_matcher.knnMatch(
            self.prev_features['descriptors'], 
            curr_features['descriptors'], 
            k=2
        )
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_matches:
            return None
        
        # Extract matched points
        prev_points = np.float32([self.prev_features['keypoints'][m.queryIdx].pt 
                                 for m in good_matches]).reshape(-1, 1, 2)
        curr_points = np.float32([curr_features['keypoints'][m.trainIdx].pt 
                                 for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate transformation using RANSAC
        transformation_matrix, mask = cv2.estimateAffine2D(
            prev_points, curr_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        if transformation_matrix is not None:
            # Convert 2D transformation to 3D pose update
            # This is a simplified approach - real implementation would use
            # full 3D reconstruction
            pose_update = np.eye(4)
            pose_update[0:2, 0:2] = transformation_matrix[0:2, 0:2]
            pose_update[0:2, 3] = transformation_matrix[0:2, 2]
            
            return pose_update
        
        return None
    
    def update_pose(self, transformation):
        """Update robot pose using transformation matrix"""
        self.current_pose = self.current_pose @ transformation
    
    def should_create_keyframe(self, transformation):
        """Determine if a new keyframe should be created"""
        # Calculate movement magnitude
        translation = np.linalg.norm(transformation[0:3, 3])
        
        # Create keyframe if movement is significant
        return translation > self.keyframe_threshold
    
    def create_keyframe(self, image, gray_image):
        """Create a keyframe for map building"""
        # Store keyframe with pose
        keyframe_data = {
            'image': image,
            'gray_image': gray_image,
            'pose': self.current_pose.copy(),
            'features': self.extract_features(gray_image),
            'timestamp': self.get_clock().now()
        }
        
        self.keyframes.append(keyframe_data)
        
        # Publish keyframe for visualization
        keyframe_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        keyframe_msg.header.stamp = self.get_clock().now().to_msg()
        keyframe_msg.header.frame_id = 'keyframe'
        self.keyframe_pub.publish(keyframe_msg)
    
    def publish_pose_and_odometry(self, stamp):
        """Publish pose and odometry messages"""
        # Create pose message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        
        # Extract position and orientation from transformation matrix
        pose_msg.pose.position.x = float(self.current_pose[0, 3])
        pose_msg.pose.position.y = float(self.current_pose[1, 3])
        pose_msg.pose.position.z = float(self.current_pose[2, 3])
        
        # Convert rotation matrix to quaternion
        rotation_matrix = self.current_pose[0:3, 0:3]
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()
        
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        
        self.pose_pub.publish(pose_msg)
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'
        
        # Position
        odom_msg.pose.pose = pose_msg.pose
        
        # Velocity (approximate from pose changes)
        if len(self.keyframes) > 1:
            # Calculate velocity from pose changes
            prev_pose = self.keyframes[-2]['pose']
            dt = 0.1  # Assume 10Hz for simplicity
            vel_x = (self.current_pose[0, 3] - prev_pose[0, 3]) / dt
            vel_y = (self.current_pose[1, 3] - prev_pose[1, 3]) / dt
            vel_z = (self.current_pose[2, 3] - prev_pose[2, 3]) / dt
            
            odom_msg.twist.twist.linear.x = float(vel_x)
            odom_msg.twist.twist.linear.y = float(vel_y)
            odom_msg.twist.twist.linear.z = float(vel_z)
        
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacVisualSLAMNode()
    
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

## Isaac ROS Navigation Integration

Navigation systems are critical for autonomous robot operation. Isaac ROS provides optimized navigation capabilities that work seamlessly with its perception packages.

### Navigation Stack Configuration

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
import numpy as np
import math

class IsaacNavigationNode(Node):
    def __init__(self):
        super().__init__('isaac_navigation_node')
        
        # Navigation parameters
        self.planner_frequency = 1.0  # Hz
        self.controller_frequency = 10.0  # Hz
        self.transform_tolerance = 0.1  # seconds
        self.global_frame = 'map'
        self.robot_base_frame = 'base_link'
        
        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.global_plan = []
        self.local_plan = []
        
        # Goal state
        self.goal_pose = None
        self.is_active = False
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.global_plan_pub = self.create_publisher(Path, '/plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.path_marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)
        
        # Services
        self.make_plan_client = self.create_client(GetPlan, '/plan')
        
        # Timers
        self.planner_timer = self.create_timer(1.0/self.planner_frequency, self.plan_callback)
        self.controller_timer = self.create_timer(1.0/self.controller_frequency, self.control_callback)
        
        # Navigation state
        self.nav_state = 'IDLE'  # IDLE, PLANNING, EXECUTING, STOPPED
        self.recovery_behavior = 'clear_costmap'  # Current recovery behavior
        
    def initial_pose_callback(self, msg):
        """Handle initial pose estimation"""
        self.current_pose = msg.pose.pose
        self.get_logger().info(f'Set initial pose: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})')
    
    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist
    
    def scan_callback(self, msg):
        """Handle laser scan data for obstacle detection"""
        # Process laser scan for local obstacle detection
        # This would integrate with local planner for obstacle avoidance
        pass
    
    def set_goal(self, x, y, theta=0.0):
        """Set navigation goal"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.global_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        
        # Convert theta to quaternion
        goal_pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_pose.pose.orientation.w = math.cos(theta / 2.0)
        
        self.goal_pose = goal_pose
        self.nav_state = 'PLANNING'
        
        self.get_logger().info(f'Set navigation goal: ({x:.2f}, {y:.2f})')
        return self.plan_path()
    
    def plan_path(self):
        """Plan global path to goal"""
        if self.goal_pose is None or self.current_pose is None:
            return False
        
        # Create plan request
        plan_request = GetPlan.Request()
        plan_request.start.header.frame_id = self.global_frame
        plan_request.start.header.stamp = self.get_clock().now().to_msg()
        plan_request.start.pose = self.current_pose
        
        plan_request.goal = self.goal_pose
        plan_request.tolerance = 0.5  # 0.5m tolerance
        
        # Wait for plan service
        if not self.make_plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Plan service not available')
            return False
        
        # Call plan service
        future = self.make_plan_client.call_async(plan_request)
        future.add_done_callback(self.plan_response_callback)
        
        return True
    
    def plan_response_callback(self, future):
        """Handle plan response"""
        try:
            response = future.result()
            if response.plan.poses:
                self.global_plan = response.plan.poses
                self.nav_state = 'EXECUTING'
                self.get_logger().info(f'Plan received with {len(self.global_plan)} waypoints')
                
                # Publish global plan
                path_msg = Path()
                path_msg.header.frame_id = self.global_frame
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.poses = self.global_plan
                self.global_plan_pub.publish(path_msg)
                
                # Visualize path
                self.visualize_path()
            else:
                self.get_logger().warn('No valid plan found')
                self.nav_state = 'IDLE'
        except Exception as e:
            self.get_logger().error(f'Plan service call failed: {e}')
            self.nav_state = 'IDLE'
    
    def plan_callback(self):
        """Global planning callback"""
        if self.nav_state == 'PLANNING' and self.goal_pose is not None:
            self.plan_path()
    
    def control_callback(self):
        """Local control and navigation execution"""
        if self.nav_state != 'EXECUTING':
            return
        
        if not self.global_plan:
            self.nav_state = 'IDLE'
            self.get_logger().info('Reached goal or plan empty')
            return
        
        if self.current_pose is None:
            return
        
        # Get current robot position
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        
        # Calculate distance to goal
        goal_x = self.global_plan[-1].pose.position.x
        goal_y = self.global_plan[-1].pose.position.y
        distance_to_goal = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        
        # Check if goal reached
        if distance_to_goal < 0.5:  # 0.5m tolerance
            self.nav_state = 'IDLE'
            self.get_logger().info('Goal reached!')
            self.stop_robot()
            return
        
        # Follow the plan - simple proportional controller
        cmd_vel = self.follow_global_plan()
        self.cmd_vel_pub.publish(cmd_vel)
    
    def follow_global_plan(self):
        """Follow the global plan with local obstacle avoidance"""
        if not self.global_plan or self.current_pose is None:
            cmd = Twist()
            return cmd
        
        # Get robot position
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        
        # Find closest point on path
        closest_idx = 0
        min_dist = float('inf')
        
        for i, pose in enumerate(self.global_plan):
            dist = math.sqrt(
                (pose.pose.position.x - robot_x)**2 + 
                (pose.pose.position.y - robot_y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Target point ahead of closest point
        target_idx = min(closest_idx + 5, len(self.global_plan) - 1)
        target_pose = self.global_plan[target_idx].pose
        
        # Calculate desired direction
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        
        # Calculate angle to target
        angle_to_target = math.atan2(target_y - robot_y, target_x - robot_x)
        
        # Get robot orientation
        robot_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        
        # Calculate angle difference
        angle_diff = self.normalize_angle(angle_to_target - robot_yaw)
        
        # Create velocity command
        cmd = Twist()
        
        # Proportional control for forward speed
        cmd.linear.x = min(0.5, max(0.1, 0.5 * (1.0 - abs(angle_diff))))
        
        # Proportional control for angular speed
        cmd.angular.z = 1.0 * angle_diff
        
        # Simple obstacle avoidance (simplified)
        if hasattr(self, 'last_scan') and self.last_scan:
            # Check for obstacles in front
            front_range = self.last_scan.ranges[len(self.last_scan.ranges)//2]
            if front_range < 0.8:  # Obstacle within 0.8m
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn right to avoid
        
        return cmd
    
    def get_yaw_from_quaternion(self, orientation):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def visualize_path(self):
        """Visualize the navigation path with markers"""
        marker_array = MarkerArray()
        
        # Create markers for path points
        for i, pose in enumerate(self.global_plan):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "path"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose = pose.pose
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        # Create line strip for path
        line_marker = Marker()
        line_marker.header.frame_id = self.global_frame
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "path_line"
        line_marker.id = len(self.global_plan)
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        for pose in self.global_plan:
            point = pose.pose.position
            line_marker.points.append(point)
        
        line_marker.scale.x = 0.05
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 0.8
        
        marker_array.markers.append(line_marker)
        
        self.path_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacNavigationNode()
    
    # Example: Set a goal after 2 seconds
    node.create_timer(2.0, lambda: node.set_goal(2.0, 2.0))
    
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

## Advanced Perception Pipelines

### Multi-Modal Perception Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, LaserScan
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
import tf2_ros
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

class IsaacMultiModalPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_multi_modal_perception')
        
        self.bridge = CvBridge()
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Sensor data storage
        self.rgb_image = None
        self.depth_image = None
        self.camera_intrinsics = None
        self.laser_scan = None
        self.point_cloud = None
        
        # Object detection results
        self.detected_objects = []
        self.tracked_objects = {}
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.pc_sub = self.create_subscription(
            PointCloud2, '/points', self.pointcloud_callback, 10)
        
        # Publishers
        self.detection_pub = self.create_publisher(
            MarkerArray, '/object_detections', 10)
        self.fused_detection_pub = self.create_publisher(
            MarkerArray, '/fused_detections', 10)
        self.semantic_map_pub = self.create_publisher(
            PointCloud2, '/semantic_map', 10)
        
        # Processing parameters
        self.detection_threshold = 0.5
        self.fusion_distance_threshold = 0.5
        self.tracking_timeout = 5.0  # seconds
        
        # Initialize processing timer
        self.process_timer = self.create_timer(0.1, self.process_fused_perception)
    
    def camera_info_callback(self, msg):
        """Store camera intrinsics"""
        self.camera_intrinsics = np.array(msg.k).reshape(3, 3)
    
    def rgb_callback(self, msg):
        """Process RGB image"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')
    
    def depth_callback(self, msg):
        """Process depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_scan = msg
    
    def pointcloud_callback(self, msg):
        """Process point cloud data"""
        self.point_cloud = msg
    
    def process_fused_perception(self):
        """Process multi-modal perception"""
        if (self.rgb_image is None or self.depth_image is None or 
            self.camera_intrinsics is None):
            return
        
        # Perform 2D object detection on RGB
        detections_2d = self.detect_objects_2d(self.rgb_image)
        
        # Fuse with depth information
        detections_3d = self.fuse_2d_with_depth(detections_2d)
        
        # Integrate with laser scan data
        fused_detections = self.fuse_with_laser(detections_3d)
        
        # Update object tracking
        self.update_object_tracking(fused_detections)
        
        # Publish results
        self.publish_detections(fused_detections)
    
    def detect_objects_2d(self, image):
        """Perform 2D object detection (simplified implementation)"""
        # This would typically use a deep learning model
        # For this example, we'll use a simple color-based detection
        
        detections = []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different objects
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }
        
        for obj_class, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    detection = {
                        'class': obj_class,
                        'confidence': 0.8,  # Placeholder confidence
                        'bbox': [x, y, x+w, y+h],
                        'center': [x + w//2, y + h//2]
                    }
                    detections.append(detection)
        
        return detections
    
    def fuse_2d_with_depth(self, detections_2d):
        """Fuse 2D detections with depth information to get 3D positions"""
        detections_3d = []
        
        for detection in detections_2d:
            bbox = detection['bbox']
            center_x, center_y = detection['center']
            
            # Get depth at object center
            if (center_y < self.depth_image.shape[0] and 
                center_x < self.depth_image.shape[1]):
                
                depth_value = self.depth_image[center_y, center_x]
                
                if depth_value > 0 and depth_value < 10:  # Valid depth range
                    # Convert pixel coordinates to 3D world coordinates
                    world_x = (center_x - self.camera_intrinsics[0, 2]) * depth_value / self.camera_intrinsics[0, 0]
                    world_y = (center_y - self.camera_intrinsics[1, 2]) * depth_value / self.camera_intrinsics[1, 1]
                    world_z = depth_value
                    
                    detection_3d = {
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'position': (world_x, world_y, world_z),
                        'bbox': detection['bbox'],
                        'pixel_center': detection['center']
                    }
                    detections_3d.append(detection_3d)
        
        return detections_3d
    
    def fuse_with_laser(self, detections_3d):
        """Fuse 3D detections with laser scan data"""
        if self.laser_scan is None:
            return detections_3d
        
        fused_detections = []
        
        for detection in detections_3d:
            pos_3d = detection['position']
            
            # Convert 3D position to laser frame if needed
            # For simplicity, assume same frame
            laser_x, laser_y, _ = pos_3d
            
            # Check if this position corresponds to laser scan reading
            # Calculate corresponding angle for laser scan
            angle = math.atan2(laser_y, laser_x)
            
            # Find closest laser reading
            angle_increment = self.laser_scan.angle_increment
            angle_min = self.laser_scan.angle_min
            angle_index = int((angle - angle_min) / angle_increment)
            
            if 0 <= angle_index < len(self.laser_scan.ranges):
                laser_range = self.laser_scan.ranges[angle_index]
                expected_range = math.sqrt(laser_x**2 + laser_y**2)
                
                # Check consistency between camera and laser
                if abs(laser_range - expected_range) < self.fusion_distance_threshold:
                    detection['fused'] = True
                    detection['confidence'] = min(1.0, detection['confidence'] + 0.2)
            
            fused_detections.append(detection)
        
        return fused_detections
    
    def update_object_tracking(self, detections):
        """Update object tracking"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Update existing tracks
        for detection in detections:
            pos_3d = detection['position']
            pos_key = (round(pos_3d[0], 1), round(pos_3d[1], 1))
            
            if pos_key in self.tracked_objects:
                # Update existing track
                track = self.tracked_objects[pos_key]
                track['last_seen'] = current_time
                track['position'] = pos_3d
                track['class'] = detection['class']
                track['confidence'] = detection['confidence']
            else:
                # Create new track
                self.tracked_objects[pos_key] = {
                    'position': pos_3d,
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'id': len(self.tracked_objects) + 1
                }
        
        # Remove old tracks
        tracks_to_remove = []
        for pos_key, track in self.tracked_objects.items():
            if current_time - track['last_seen'] > self.tracking_timeout:
                tracks_to_remove.append(pos_key)
        
        for pos_key in tracks_to_remove:
            del self.tracked_objects[pos_key]
    
    def publish_detections(self, detections):
        """Publish object detections as markers"""
        marker_array = MarkerArray()
        
        # Publish individual detections
        for i, detection in enumerate(detections):
            marker = Marker()
            marker.header.frame_id = 'camera_link'  # or appropriate frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'object_detections'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position
            pos = detection['position']
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            # Color based on class
            if detection['class'] == 'red':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif detection['class'] == 'blue':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            
            marker.color.a = 0.8
            
            # Add label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = 'object_labels'
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose = marker.pose
            text_marker.pose.position.z += 0.3  # Offset label above object
            text_marker.scale.z = 0.1
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.text = f"{detection['class']}: {detection['confidence']:.2f}"
            
            marker_array.markers.extend([marker, text_marker])
        
        self.detection_pub.publish(marker_array)
        
        # Publish tracked objects
        tracked_marker_array = MarkerArray()
        for i, (pos_key, track) in enumerate(self.tracked_objects.items()):
            marker = Marker()
            marker.header.frame_id = 'map'  # or appropriate frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'tracked_objects'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            pos = track['position']
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            marker.color.r = 1.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        self.fused_detection_pub.publish(tracked_marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacMultiModalPerceptionNode()
    
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

## Isaac ROS Navigation Configuration

### Navigation Parameters and Tuning

```yaml
# navigation_params.yaml
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
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "differential"
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
    scan_topic: scan

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
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
    - nav2_single_trigger_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      progress_checker:
        plugin: "nav2_controller::SimpleProgressChecker"
        required_movement_radius: 0.5
        movement_time_allowance: 10.0
      goal_checker:
        plugin: "nav2_controller::SimpleGoalChecker"
        xy_goal_tolerance: 0.25
        yaw_goal_tolerance: 0.25
        stateful: True
      RotateShim:
        plugin: "nav2_controller::SimpleGoalChecker"
        xy_goal_tolerance: 0.25
        yaw_goal_tolerance: 0.25
        stateful: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
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
        z_resolution: 0.05
        z_voxels: 16
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
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: True
      robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
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
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

smoother_server:
  ros__parameters:
    use_sim_time: True
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True
```

## Exercises

1. Implement a Visual SLAM system using Isaac ROS packages and test it in a simulated environment. Evaluate the accuracy of pose estimation and map building.

2. Configure and tune the navigation stack for a mobile robot in Isaac Sim. Test path planning and execution in various environments.

3. Create a multi-modal perception system that combines camera, LIDAR, and IMU data for robust object detection and tracking.

4. Design and implement a recovery behavior for the navigation system that handles common failure scenarios like getting stuck or local minima.

## References

NVIDIA. (2023). *Isaac ROS Documentation*. Retrieved from https://nvidia-isaac-ros.github.io/

ROS.org. (2023). *Navigation Stack (Nav2) Documentation*. Retrieved from https://navigation.ros.org/

Mur-Artal, R., & Tardós, J. D. (2017). ORB-SLAM2: an open-source SLAM system for monocular, stereo, and RGB-D cameras. *IEEE Transactions on Robotics*, 33(5), 1255-1262.

Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance. *IEEE Robotics & Automation Magazine*, 4(1), 23-33.