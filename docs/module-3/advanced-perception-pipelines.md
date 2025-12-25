# Save this file as specify/implement/module-3/advanced-perception-pipelines.md

# Advanced Perception Pipelines

## Overview

Advanced perception pipelines leverage NVIDIA's AI capabilities for complex robotic perception tasks, including object detection, segmentation, and scene understanding. Isaac ROS provides GPU-accelerated perception nodes that integrate seamlessly with ROS 2.

## Isaac ROS Perception Architecture

The Isaac ROS perception pipeline architecture:

```
Raw Sensor Data → Isaac ROS Nodes → Processed Perception Data → Applications
                        ↓
                GPU Acceleration Layer
```

## Isaac ROS Perception Nodes

### DetectNet for Object Detection

```python
# Example perception pipeline using Isaac ROS
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_detectnet',
                plugin='nvidia::isaac_ros::detection::DetectNetNode',
                name='detectnet',
                parameters=[{
                    'model_name': 'ssd_mobilenet_v2_coco',
                    'input_topic': '/camera/image_raw',
                    'camera_info_topic': '/camera/camera_info',
                    'max_batch_size': 1,
                    'max_workspace_size': 1073741824,  # 1GB
                    'tensorrt_precision': 'FP16',
                    'threshold': 0.5,
                    'enable_padding': True
                }],
                remappings=[
                    ('image', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info'),
                    ('detections', '/detectnet/detections')
                ]
            )
        ]
    )

    return LaunchDescription([perception_container])
```

### Segmentation Node

```python
# Semantic segmentation node
def generate_segmentation_launch():
    segmentation_container = ComposableNodeContainer(
        name='segmentation_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_segmentation',
                plugin='nvidia::isaac_ros::segmentation::SegmentationNode',
                name='segmentation',
                parameters=[{
                    'model_name': 'unet_coral',
                    'input_topic': '/camera/image_raw',
                    'output_topic': '/segmentation',
                    'colormap_topic': '/segmentation_colormap',
                    'max_batch_size': 1,
                    'max_workspace_size': 536870912,  # 512MB
                    'tensorrt_precision': 'FP16'
                }],
                remappings=[
                    ('image', '/camera/image_raw'),
                    ('segmentation', '/segmentation'),
                    ('colormap', '/segmentation_colormap')
                ]
            )
        ]
    )

    return LaunchDescription([segmentation_container])
```

### Stereo Disparity Node

```python
# Stereo disparity for depth estimation
def generate_stereo_launch():
    stereo_container = ComposableNodeContainer(
        name='stereo_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
                name='disparity_node',
                parameters=[{
                    'min_disparity': 0.0,
                    'max_disparity': 64.0,
                    'delta_d': 1.0,
                    'kernel_size': 5
                }],
                remappings=[
                    ('left/image_rect', '/camera/left/image_rect'),
                    ('left/camera_info', '/camera/left/camera_info'),
                    ('right/image_rect', '/camera/right/image_rect'),
                    ('right/camera_info', '/camera/right/camera_info'),
                    ('disparity', '/disparity')
                ]
            )
        ]
    )

    return LaunchDescription([stereo_container])
```

## Perception Pipeline Integration

### Multi-Modal Perception Pipeline

```python
# Complete multi-modal perception pipeline
def generate_multi_modal_pipeline():
    perception_container = ComposableNodeContainer(
        name='multi_modal_perception',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Image rectification
            ComposableNode(
                package='isaac_ros_diff_rectify',
                plugin='nvidia::isaac_ros::differential_rectification::RectifyNode',
                name='rectify_left',
                parameters=[{
                    'output_width': 640,
                    'output_height': 480
                }],
                remappings=[
                    ('image_raw', '/camera/left/image_raw'),
                    ('camera_info', '/camera/left/camera_info'),
                    ('image_rect', '/camera/left/image_rect'),
                    ('camera_info_rect', '/camera/left/camera_info_rect')
                ]
            ),
            ComposableNode(
                package='isaac_ros_diff_rectify',
                plugin='nvidia::isaac_ros::differential_rectification::RectifyNode',
                name='rectify_right',
                parameters=[{
                    'output_width': 640,
                    'output_height': 480
                }],
                remappings=[
                    ('image_raw', '/camera/right/image_raw'),
                    ('camera_info', '/camera/right/camera_info'),
                    ('image_rect', '/camera/right/image_rect'),
                    ('camera_info_rect', '/camera/right/camera_info_rect')
                ]
            ),
            # Object detection
            ComposableNode(
                package='isaac_ros_detectnet',
                plugin='nvidia::isaac_ros::detection::DetectNetNode',
                name='detectnet',
                parameters=[{
                    'model_name': 'ssd_mobilenet_v2_coco',
                    'input_topic': '/camera/left/image_rect',
                    'max_batch_size': 1,
                    'max_workspace_size': 1073741824,
                    'tensorrt_precision': 'FP16',
                    'threshold': 0.5
                }],
                remappings=[
                    ('image', '/camera/left/image_rect'),
                    ('camera_info', '/camera/left/camera_info_rect'),
                    ('detections', '/detectnet/detections')
                ]
            ),
            # Stereo disparity
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
                name='disparity_node',
                parameters=[{
                    'min_disparity': 0.0,
                    'max_disparity': 64.0,
                    'delta_d': 1.0,
                    'kernel_size': 5
                }],
                remappings=[
                    ('left/image_rect', '/camera/left/image_rect'),
                    ('left/camera_info', '/camera/left/camera_info_rect'),
                    ('right/image_rect', '/camera/right/image_rect'),
                    ('right/camera_info', '/camera/right/camera_info_rect'),
                    ('disparity', '/disparity')
                ]
            ),
            # Point cloud generation
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
                name='pointcloud_node',
                parameters=[{
                    'use_color': True
                }],
                remappings=[
                    ('left/image_rect_color', '/camera/left/image_rect'),
                    ('left/camera_info', '/camera/left/camera_info_rect'),
                    ('right/camera_info', '/camera/right/camera_info_rect'),
                    ('disparity', '/disparity'),
                    ('points', '/points')
                ]
            )
        ]
    )

    return LaunchDescription([perception_container])
```

## Custom Perception Node Development

### Creating a Custom Perception Node

```cpp
// Example custom perception node
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class CustomPerceptionNode : public rclcpp::Node
{
public:
    CustomPerceptionNode() : Node("custom_perception_node")
    {
        // Create subscriber and publisher
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "input_image", 10,
            std::bind(&CustomPerceptionNode::image_callback, this, std::placeholders::_1));
            
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "output_image", 10);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        
        // Perform custom perception processing
        cv::Mat processed_image = process_image(cv_ptr->image);
        
        // Convert back to ROS image
        sensor_msgs::msg::Image::SharedPtr output_msg = 
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", processed_image).toImageMsg();
            
        // Publish processed image
        publisher_->publish(*output_msg);
    }
    
    cv::Mat process_image(const cv::Mat& input)
    {
        // Implement custom perception algorithm
        cv::Mat output;
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
        cv::Canny(output, output, 50, 150);
        return output;
    }
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CustomPerceptionNode>());
    rclcpp::shutdown();
    return 0;
}
```

## Performance Optimization

### TensorRT Optimization for Perception

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTOptimizer:
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def build_engine(self, onnx_model_path, precision='fp16'):
        """Build TensorRT engine from ONNX model"""
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        
        # Create runtime
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        return engine
```

## Perception Pipeline Evaluation

### Performance Metrics

```python
import time
import numpy as np

class PerceptionEvaluator:
    def __init__(self):
        self.frame_times = []
        self.detection_rates = []
        
    def measure_performance(self, perception_pipeline, test_data):
        """Measure perception pipeline performance"""
        for frame in test_data:
            start_time = time.time()
            
            # Process frame through pipeline
            results = perception_pipeline.process(frame)
            
            end_time = time.time()
            frame_time = end_time - start_time
            
            self.frame_times.append(frame_time)
            self.detection_rates.append(len(results))
        
        # Calculate metrics
        avg_frame_time = np.mean(self.frame_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        avg_detections = np.mean(self.detection_rates)
        
        return {
            'avg_frame_time': avg_frame_time,
            'avg_fps': avg_fps,
            'avg_detections': avg_detections,
            'latency_percentiles': np.percentile(self.frame_times, [50, 90, 95, 99])
        }
```

## Practical Exercise 7.1: Multi-Modal Perception System

Build an advanced perception pipeline that combines multiple perception capabilities:

1. Implement a pipeline that combines object detection, semantic segmentation, and depth estimation
2. Integrate Isaac ROS nodes for each perception task
3. Test the pipeline on both synthetic and real data
4. Compare performance metrics between synthetic and real-world data
5. Optimize the pipeline using TensorRT
6. Document the implementation approach and results

Evaluate:
- Detection accuracy
- Processing speed (FPS)
- Memory usage
- GPU utilization
- Robustness to different lighting conditions

Create a comprehensive evaluation report with performance benchmarks and recommendations.

## References

Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. *arXiv preprint arXiv:1804.02767*.

Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 3431-3440.

Bouget, F., Lepetit, V., & Fua, P. (2017). Monocular pose estimation of articulated objects using point pair features. *International Journal of Computer Vision*, 122(3), 434-456.