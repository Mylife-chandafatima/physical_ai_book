---
title: Unity for High-Fidelity Rendering and HRI
sidebar_label: Unity Rendering & HRI
---

# Save as docs/module-2/unity-rendering-hri.md

# Unity: High-Fidelity Rendering and Human-Robot Interaction

## Learning Outcomes
By the end of this module, you will be able to:
- Set up Unity for robotics simulation with realistic rendering
- Implement photorealistic visual environments for perception training
- Design human-robot interaction scenarios in Unity
- Integrate Unity with ROS/ROS 2 for bidirectional communication
- Optimize Unity scenes for real-time performance in robotics applications

## Unity for Robotics Overview

Unity has emerged as a powerful platform for robotics simulation, particularly for applications requiring high-fidelity visual rendering. Unlike Gazebo, which focuses on physics accuracy, Unity excels in creating photorealistic environments that are essential for training perception systems and studying human-robot interaction.

### Unity Robotics Features

- **Photorealistic rendering**: High-quality graphics with realistic lighting and materials
- **Real-time ray tracing**: Advanced lighting simulation
- **Large environment support**: Ability to create expansive, detailed worlds
- **VR/AR integration**: Support for immersive human-robot interaction
- **Cross-platform deployment**: Runs on multiple platforms with consistent results

## Setting Up Unity for Robotics

### Unity Installation and Configuration

1. **Install Unity Hub** and Unity 2021.3 LTS or later
2. **Install required packages**:
   - Unity Robotics Package
   - Unity ML-Agents (for reinforcement learning)
   - Unity Perception Package (for synthetic data generation)

### Basic Robotics Scene Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class RobotController : MonoBehaviour
{
    public string robotName = "my_robot";
    private RosConnection ros;
    
    void Start()
    {
        // Initialize ROS connection
        ros = RosConnection.GetOrCreateInstance();
        ros.RegisterPublisher<UInt64Msg>("/unity_robot_status");
        
        // Subscribe to robot commands
        ros.Subscribe<Unity.RosMessageTypes.Geometry.TwistMsg>(
            "/cmd_vel", ReceiveTwistCommand);
    }
    
    void ReceiveTwistCommand(Unity.RosMessageTypes.Geometry.TwistMsg cmd)
    {
        // Process velocity commands
        Vector3 linear = new Vector3((float)cmd.linear.x, 
                                    (float)cmd.linear.y, 
                                    (float)cmd.linear.z);
        Vector3 angular = new Vector3((float)cmd.angular.x, 
                                     (float)cmd.angular.y, 
                                     (float)cmd.angular.z);
        
        // Apply movement to robot
        transform.Translate(linear * Time.deltaTime);
        transform.Rotate(angular * Time.deltaTime);
    }
    
    void Update()
    {
        // Publish robot status
        ros.Publish("/unity_robot_status", 
                   new UInt64Msg((ulong)System.DateTime.Now.Ticks));
    }
}
```

## Creating Photorealistic Environments

### Material and Lighting Setup

```csharp
using UnityEngine;

public class EnvironmentSetup : MonoBehaviour
{
    [Header("Lighting Configuration")]
    public Light sunLight;
    public float sunIntensity = 1.0f;
    public Color sunColor = Color.white;
    
    [Header("Environment Materials")]
    public Material[] environmentMaterials;
    
    void Start()
    {
        SetupLighting();
        SetupMaterials();
    }
    
    void SetupLighting()
    {
        // Configure directional light to simulate sun
        sunLight.type = LightType.Directional;
        sunLight.intensity = sunIntensity;
        sunLight.color = sunColor;
        sunLight.shadows = LightShadows.Soft;
        sunLight.shadowResolution = ShadowResolution.High;
        
        // Add ambient lighting
        RenderSettings.ambientLight = new Color(0.2f, 0.2f, 0.2f, 1.0f);
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
    }
    
    void SetupMaterials()
    {
        // Apply physically-based materials
        foreach (Material mat in environmentMaterials)
        {
            mat.EnableKeyword("_METALLICGLOSSMAP");
            mat.SetFloat("_Metallic", 0.5f);
            mat.SetFloat("_Glossiness", 0.5f);
        }
    }
}
```

### Procedural Environment Generation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class ProceduralEnvironment : MonoBehaviour
{
    [Header("Environment Parameters")]
    public int roomCount = 10;
    public float roomSizeMin = 5f;
    public float roomSizeMax = 15f;
    public float roomHeight = 3f;
    
    [Header("Object Prefabs")]
    public GameObject[] furniturePrefabs;
    public GameObject[] obstaclePrefabs;
    
    private List<GameObject> generatedRooms = new List<GameObject>();
    
    void Start()
    {
        GenerateEnvironment();
    }
    
    void GenerateEnvironment()
    {
        for (int i = 0; i < roomCount; i++)
        {
            GenerateRoom(i);
        }
    }
    
    void GenerateRoom(int index)
    {
        // Create room dimensions
        float width = Random.Range(roomSizeMin, roomSizeMax);
        float depth = Random.Range(roomSizeMin, roomSizeMax);
        
        // Create room container
        GameObject room = new GameObject($"Room_{index}");
        room.transform.position = new Vector3(
            index * (width + 2), 0, 0); // Space rooms apart
        
        // Create floor
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floor.transform.SetParent(room.transform);
        floor.transform.localScale = new Vector3(width, 0.1f, depth);
        floor.transform.position = new Vector3(0, -0.05f, 0);
        floor.GetComponent<Renderer>().material.color = 
            Random.ColorHSV(0.1f, 0.2f, 0.5f, 1f, 0.8f, 1f);
        
        // Create walls
        CreateWalls(room, width, depth);
        
        // Add furniture
        AddFurniture(room, width, depth);
        
        generatedRooms.Add(room);
    }
    
    void CreateWalls(GameObject room, float width, float depth)
    {
        float wallHeight = roomHeight;
        float wallThickness = 0.2f;
        
        // Create 4 walls
        Vector3[] wallPositions = {
            new Vector3(0, wallHeight/2, depth/2 + wallThickness/2), // North
            new Vector3(0, wallHeight/2, -depth/2 - wallThickness/2), // South
            new Vector3(width/2 + wallThickness/2, wallHeight/2, 0), // East
            new Vector3(-width/2 - wallThickness/2, wallHeight/2, 0)  // West
        };
        
        foreach (Vector3 pos in wallPositions)
        {
            GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.transform.SetParent(room.transform);
            wall.transform.position = pos;
            
            if (pos.z > 0 || pos.z < 0) // North/South walls
            {
                wall.transform.localScale = new Vector3(width + 2*wallThickness, 
                                                      wallHeight, wallThickness);
            }
            else // East/West walls
            {
                wall.transform.localScale = new Vector3(wallThickness, 
                                                      wallHeight, depth + 2*wallThickness);
            }
            
            wall.GetComponent<Renderer>().material.color = 
                Random.ColorHSV(0.5f, 0.6f, 0.5f, 1f, 0.8f, 1f);
        }
    }
    
    void AddFurniture(GameObject room, float width, float depth)
    {
        int furnitureCount = Random.Range(2, 6);
        
        for (int i = 0; i < furnitureCount; i++)
        {
            if (furniturePrefabs.Length > 0)
            {
                GameObject prefab = furniturePrefabs[
                    Random.Range(0, furniturePrefabs.Length)];
                
                GameObject furniture = Instantiate(prefab, room.transform);
                
                // Random position within room bounds
                float x = Random.Range(-width/2 + 1, width/2 - 1);
                float z = Random.Range(-depth/2 + 1, depth/2 - 1);
                
                furniture.transform.localPosition = new Vector3(x, 0, z);
                
                // Random rotation
                furniture.transform.localRotation = 
                    Quaternion.Euler(0, Random.Range(0, 360), 0);
            }
        }
    }
}
```

## Sensor Simulation in Unity

### Camera Sensor Implementation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections;
using System.IO;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    public int imageWidth = 640;
    public int imageHeight = 480;
    public string cameraTopic = "/camera/image_raw";
    public string cameraInfoTopic = "/camera/camera_info";
    
    [Header("Noise Parameters")]
    public float noiseIntensity = 0.01f;
    
    private Camera cam;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private RosConnection ros;
    
    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        
        // Set up camera
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            cam = gameObject.AddComponent<Camera>();
        }
        
        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cam.targetTexture = renderTexture;
        
        // Create texture for reading
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        
        // Start publishing loop
        StartCoroutine(PublishImages());
    }
    
    IEnumerator PublishImages()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();
            PublishImage();
            yield return new WaitForSeconds(1f / 30f); // 30 FPS
        }
    }
    
    void PublishImage()
    {
        // Set active render texture
        RenderTexture.active = renderTexture;
        
        // Read pixels from render texture
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();
        
        // Add noise if enabled
        if (noiseIntensity > 0)
        {
            AddNoiseToTexture(texture2D);
        }
        
        // Convert to ROS message format
        byte[] imageBytes = texture2D.EncodeToJPG();
        
        // Create and publish ROS image message
        var imageMsg = new Unity.RosMessageTypes.Sensor.ImageMsg
        {
            header = new Unity.RosMessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.RosMessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_frame"
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(imageWidth * 3), // 3 bytes per pixel
            data = imageBytes
        };
        
        ros.Publish(cameraTopic, imageMsg);
        
        // Publish camera info
        PublishCameraInfo();
    }
    
    void AddNoiseToTexture(Texture2D tex)
    {
        Color[] pixels = tex.GetPixels();
        
        for (int i = 0; i < pixels.Length; i++)
        {
            Color original = pixels[i];
            pixels[i] = new Color(
                original.r + Random.Range(-noiseIntensity, noiseIntensity),
                original.g + Random.Range(-noiseIntensity, noiseIntensity),
                original.b + Random.Range(-noiseIntensity, noiseIntensity),
                original.a
            );
        }
        
        tex.SetPixels(pixels);
        tex.Apply();
    }
    
    void PublishCameraInfo()
    {
        var cameraInfo = new Unity.RosMessageTypes.Sensor.CameraInfoMsg
        {
            header = new Unity.RosMessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.RosMessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_frame"
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            distortion_model = "plumb_bob",
            D = new double[] { 0, 0, 0, 0, 0 }, // No distortion
            K = new double[] { 
                525, 0, imageWidth/2.0,  // fx, 0, cx
                0, 525, imageHeight/2.0,  // 0, fy, cy
                0, 0, 1                   // 0, 0, 1
            },
            R = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
            P = new double[] { 
                525, 0, imageWidth/2.0, 0,   // fx', 0, cx', Tx
                0, 525, imageHeight/2.0, 0,  // 0, fy', cy', Ty
                0, 0, 1, 0                   // 0, 0, 1, Tz
            }
        };
        
        ros.Publish(cameraInfoTopic, cameraInfo);
    }
}
```

### LIDAR Simulation in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class UnityLidarSensor : MonoBehaviour
{
    [Header("LIDAR Configuration")]
    public int horizontalSamples = 360;
    public int verticalSamples = 1;
    public float minRange = 0.1f;
    public float maxRange = 30.0f;
    public float angleMin = -Mathf.PI;
    public float angleMax = Mathf.PI;
    public string lidarTopic = "/scan";
    
    [Header("Raycast Parameters")]
    public LayerMask detectionMask = -1;
    public float noiseStdDev = 0.01f;
    
    private RosConnection ros;
    private float[] ranges;
    
    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        ranges = new float[horizontalSamples];
        
        // Start LIDAR scanning
        InvokeRepeating("ScanEnvironment", 0, 1f/10f); // 10 Hz
    }
    
    void ScanEnvironment()
    {
        for (int i = 0; i < horizontalSamples; i++)
        {
            float angle = angleMin + (angleMax - angleMin) * i / (horizontalSamples - 1);
            
            // Calculate ray direction
            Vector3 direction = new Vector3(
                Mathf.Cos(angle), 
                0, 
                Mathf.Sin(angle)
            );
            
            direction = transform.TransformDirection(direction);
            
            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange, detectionMask))
            {
                float distance = hit.distance;
                
                // Add noise
                if (noiseStdDev > 0)
                {
                    distance += RandomGaussian(0, noiseStdDev);
                }
                
                ranges[i] = Mathf.Clamp(distance, minRange, maxRange);
            }
            else
            {
                ranges[i] = float.PositiveInfinity;
            }
        }
        
        PublishLidarData();
    }
    
    void PublishLidarData()
    {
        var scanMsg = new Unity.RosMessageTypes.Sensor.LaserScanMsg
        {
            header = new Unity.RosMessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.RosMessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "lidar_frame"
            },
            angle_min = angleMin,
            angle_max = angleMax,
            angle_increment = (angleMax - angleMin) / (horizontalSamples - 1),
            time_increment = 0,
            scan_time = 0.1f, // 10 Hz
            range_min = minRange,
            range_max = maxRange,
            ranges = new float[horizontalSamples]
        };
        
        // Copy ranges with proper conversion
        for (int i = 0; i < ranges.Length; i++)
        {
            scanMsg.ranges[i] = ranges[i] < maxRange ? ranges[i] : float.PositiveInfinity;
        }
        
        // Fill intensities if needed
        scanMsg.intensities = new float[horizontalSamples];
        for (int i = 0; i < horizontalSamples; i++)
        {
            scanMsg.intensities[i] = 100.0f; // Placeholder intensity
        }
        
        ros.Publish(lidarTopic, scanMsg);
    }
    
    float RandomGaussian(float mean, float stdDev)
    {
        // Box-Muller transform
        float u1 = Random.value;
        float u2 = Random.value;
        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }
}
```

## Human-Robot Interaction (HRI) in Unity

### Human Avatar Controller

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class HumanAvatarController : MonoBehaviour
{
    [Header("Human Movement")]
    public float walkSpeed = 2.0f;
    public float runSpeed = 4.0f;
    public float jumpForce = 5.0f;
    
    [Header("Interaction")]
    public float interactionDistance = 2.0f;
    public LayerMask interactionLayer;
    
    private CharacterController controller;
    private Animator animator;
    private RosConnection ros;
    private float yVelocity = 0;
    private bool isGrounded = true;
    
    void Start()
    {
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
        ros = RosConnection.GetOrCreateInstance();
    }
    
    void Update()
    {
        HandleMovement();
        HandleInteraction();
        HandleAnimations();
    }
    
    void HandleMovement()
    {
        if (controller == null) return;
        
        // Get input
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        
        // Calculate movement direction
        Vector3 moveDirection = new Vector3(horizontal, 0, vertical);
        moveDirection = transform.TransformDirection(moveDirection);
        moveDirection.Normalize();
        
        // Apply gravity
        if (isGrounded && yVelocity < 0)
        {
            yVelocity = -2f;
        }
        else
        {
            yVelocity += Physics.gravity.y * Time.deltaTime;
        }
        
        // Apply movement
        Vector3 movement = moveDirection * (Input.GetKey(KeyCode.LeftShift) ? runSpeed : walkSpeed) * Time.deltaTime;
        movement.y = yVelocity * Time.deltaTime;
        
        controller.Move(movement);
        
        // Check if grounded
        isGrounded = controller.isGrounded;
        if (isGrounded && yVelocity < 0)
        {
            yVelocity = -2f;
        }
    }
    
    void HandleInteraction()
    {
        if (Input.GetKeyDown(KeyCode.E))
        {
            // Raycast to find interactive object
            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.forward, 
                              out hit, interactionDistance, interactionLayer))
            {
                // Publish interaction message to ROS
                var interactionMsg = new Unity.RosMessageTypes.Std.StringMsg
                {
                    data = $"Human interacted with {hit.collider.name}"
                };
                
                ros.Publish("/human_interaction", interactionMsg);
                
                // Handle specific interaction
                HandleSpecificInteraction(hit.collider);
            }
        }
    }
    
    void HandleSpecificInteraction(Collider collider)
    {
        // Example: Handle door interaction
        if (collider.CompareTag("Door"))
        {
            Door door = collider.GetComponent<Door>();
            if (door != null)
            {
                door.Toggle();
            }
        }
        // Add more interaction types as needed
    }
    
    void HandleAnimations()
    {
        if (animator == null) return;
        
        // Update animation parameters based on movement
        float speed = new Vector3(controller.velocity.x, 0, controller.velocity.z).magnitude;
        animator.SetFloat("Speed", speed);
        animator.SetBool("IsGrounded", isGrounded);
        animator.SetBool("IsRunning", Input.GetKey(KeyCode.LeftShift));
    }
}

// Door component for interaction example
public class Door : MonoBehaviour
{
    public float openAngle = 90f;
    public float speed = 2f;
    
    private bool isOpen = false;
    private Vector3 closedRotation;
    private Vector3 openRotation;
    
    void Start()
    {
        closedRotation = transform.localEulerAngles;
        openRotation = new Vector3(
            closedRotation.x,
            closedRotation.y + openAngle,
            closedRotation.z
        );
    }
    
    public void Toggle()
    {
        isOpen = !isOpen;
        StartCoroutine(MoveDoor(isOpen));
    }
    
    System.Collections.IEnumerator MoveDoor(bool open)
    {
        float targetAngle = open ? openAngle : 0;
        Vector3 targetRotation = open ? openRotation : closedRotation;
        
        float time = 0;
        Vector3 startRotation = transform.localEulerAngles;
        
        while (time < 1)
        {
            time += Time.deltaTime * speed;
            transform.localEulerAngles = Vector3.Lerp(startRotation, targetRotation, time);
            yield return null;
        }
    }
}
```

### Gesture Recognition System

```csharp
using UnityEngine;
using System.Collections.Generic;

public class GestureRecognition : MonoBehaviour
{
    [Header("Gesture Recognition")]
    public float gestureThreshold = 0.1f;
    public float gestureTimeout = 2.0f;
    
    private List<Vector3> gesturePoints = new List<Vector3>();
    private float gestureStartTime;
    private bool isRecordingGesture = false;
    
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            StartRecordingGesture();
        }
        else if (Input.GetMouseButton(0))
        {
            RecordGesturePoint();
        }
        else if (Input.GetMouseButtonUp(0) && isRecordingGesture)
        {
            ProcessGesture();
        }
        
        // Timeout check
        if (isRecordingGesture && Time.time - gestureStartTime > gestureTimeout)
        {
            ResetGesture();
        }
    }
    
    void StartRecordingGesture()
    {
        gesturePoints.Clear();
        gestureStartTime = Time.time;
        isRecordingGesture = true;
    }
    
    void RecordGesturePoint()
    {
        Vector3 mousePos = Input.mousePosition;
        // Convert to world position if needed
        gesturePoints.Add(mousePos);
    }
    
    void ProcessGesture()
    {
        if (gesturePoints.Count < 5)
        {
            ResetGesture();
            return;
        }
        
        string gestureType = RecognizeGesture();
        if (!string.IsNullOrEmpty(gestureType))
        {
            ExecuteGestureCommand(gestureType);
        }
        
        ResetGesture();
    }
    
    string RecognizeGesture()
    {
        // Simple gesture recognition based on movement pattern
        if (gesturePoints.Count < 10) return null;
        
        Vector3 start = gesturePoints[0];
        Vector3 end = gesturePoints[gesturePoints.Count - 1];
        
        Vector3 direction = (end - start).normalized;
        
        // Recognize basic gestures
        if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y) && direction.x > 0.5f)
        {
            return "RIGHT_SWIPE";
        }
        else if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y) && direction.x < -0.5f)
        {
            return "LEFT_SWIPE";
        }
        else if (Mathf.Abs(direction.y) > Mathf.Abs(direction.x) && direction.y > 0.5f)
        {
            return "UP_SWIPE";
        }
        else if (Mathf.Abs(direction.y) > Mathf.Abs(direction.x) && direction.y < -0.5f)
        {
            return "DOWN_SWIPE";
        }
        
        // More complex gesture recognition would go here
        return null;
    }
    
    void ExecuteGestureCommand(string gestureType)
    {
        // Publish gesture command to ROS
        var ros = Unity.Robotics.ROSTCPConnector.RosConnection.GetOrCreateInstance();
        var gestureMsg = new Unity.RosMessageTypes.Std.StringMsg
        {
            data = gestureType.ToLower()
        };
        
        ros.Publish("/gesture_command", gestureMsg);
        
        Debug.Log($"Gesture recognized: {gestureType}");
    }
    
    void ResetGesture()
    {
        isRecordingGesture = false;
        gesturePoints.Clear();
    }
}
```

## Unity-ROS Integration Patterns

### Publisher-Subscriber Pattern

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class UnityROSPublisher : MonoBehaviour
{
    [Header("ROS Topics")]
    public string positionTopic = "/robot_position";
    public string statusTopic = "/robot_status";
    
    private RosConnection ros;
    private float publishInterval = 0.1f;
    private float lastPublishTime;
    
    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PointMsg>(positionTopic);
        ros.RegisterPublisher<StringMsg>(statusTopic);
        
        // Subscribe to commands
        ros.Subscribe<TwistMsg>("/cmd_vel", OnCommandReceived);
    }
    
    void Update()
    {
        if (Time.time - lastPublishTime > publishInterval)
        {
            PublishRobotState();
            lastPublishTime = Time.time;
        }
    }
    
    void OnCommandReceived(TwistMsg cmd)
    {
        // Process velocity command
        Vector3 linear = new Vector3((float)cmd.linear.x, 
                                    (float)cmd.linear.y, 
                                    (float)cmd.linear.z);
        Vector3 angular = new Vector3((float)cmd.angular.x, 
                                     (float)cmd.angular.y, 
                                     (float)cmd.angular.z);
        
        // Apply to robot
        transform.Translate(linear * publishInterval);
        transform.Rotate(angular * publishInterval);
    }
    
    void PublishRobotState()
    {
        // Publish position
        var positionMsg = new PointMsg
        {
            x = transform.position.x,
            y = transform.position.y,
            z = transform.position.z
        };
        
        ros.Publish(positionTopic, positionMsg);
        
        // Publish status
        var statusMsg = new StringMsg
        {
            data = "RUNNING"
        };
        
        ros.Publish(statusTopic, statusMsg);
    }
}
```

### Service Client Implementation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class UnityROSServiceClient : MonoBehaviour
{
    public string serviceName = "/robot_control_service";
    
    private RosConnection ros;
    
    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
    }
    
    public void RequestRobotAction(string action)
    {
        // Create service request
        var request = new Unity.RosMessageTypes.Std.SetBoolRequestMsg
        {
            data = action == "activate"
        };
        
        // Send service request
        ros.CallService<Unity.RosMessageTypes.Std.SetBoolRequestMsg, 
                       Unity.RosMessageTypes.Std.SetBoolResponseMsg>(
            serviceName, 
            request, 
            OnServiceResponse);
    }
    
    void OnServiceResponse(Unity.RosMessageTypes.Std.SetBoolResponseMsg response)
    {
        if (response.success)
        {
            Debug.Log($"Service call successful: {response.message}");
        }
        else
        {
            Debug.LogError($"Service call failed: {response.message}");
        }
    }
}
```

## Performance Optimization

### Rendering Optimization

```csharp
using UnityEngine;

public class RenderingOptimizer : MonoBehaviour
{
    [Header("LOD Settings")]
    public int lodDistance1 = 10;
    public int lodDistance2 = 20;
    public int lodDistance3 = 50;
    
    [Header("Quality Settings")]
    public int targetFrameRate = 60;
    public bool enableDynamicBatching = true;
    public bool enableStaticBatching = true;
    
    void Start()
    {
        OptimizeRendering();
    }
    
    void OptimizeRendering()
    {
        // Set target frame rate
        Application.targetFrameRate = targetFrameRate;
        
        // Enable batching
        QualitySettings.blendWeights = BlendWeights.FourBones;
        
        // Optimize graphics settings
        QualitySettings.vSyncCount = 0; // Disable VSync for consistent frame rate
        QualitySettings.maxQueuedFrames = 2;
    }
    
    void Update()
    {
        // LOD management
        ManageLODs();
    }
    
    void ManageLODs()
    {
        float distanceToMainCamera = Vector3.Distance(
            Camera.main.transform.position, transform.position);
        
        // Implement LOD switching based on distance
        if (distanceToMainCamera > lodDistance3)
        {
            // Use lowest detail model
            SetLOD(3);
        }
        else if (distanceToMainCamera > lodDistance2)
        {
            SetLOD(2);
        }
        else if (distanceToMainCamera > lodDistance1)
        {
            SetLOD(1);
        }
        else
        {
            SetLOD(0); // Highest detail
        }
    }
    
    void SetLOD(int lodLevel)
    {
        // Implementation depends on your LOD system
        // This is a placeholder
    }
}
```

### Physics Optimization

```csharp
using UnityEngine;

public class PhysicsOptimizer : MonoBehaviour
{
    [Header("Physics Settings")]
    public int fixedTimestep = 50; // 50 FPS for physics
    public int maxSubSteps = 10;
    
    void Start()
    {
        OptimizePhysics();
    }
    
    void OptimizePhysics()
    {
        // Set physics timestep
        Time.fixedDeltaTime = 1.0f / fixedTimestep;
        Time.maximumDeltaTime = 1.0f / 10.0f; // Max time per frame
        Physics.defaultSolverIterations = 6; // Balance between accuracy and performance
        Physics.defaultSolverVelocityIterations = 1;
    }
}
```

## Exercises

1. Create a Unity scene with a photorealistic indoor environment and implement a camera sensor that publishes images to ROS with realistic noise characteristics.

2. Design a human-robot interaction scenario where a human avatar can guide a robot through an environment using voice commands (simulated) and gestures.

3. Implement a LIDAR sensor in Unity that accurately simulates distance measurements with appropriate noise models and publishes data in ROS LaserScan format.

4. Create a multi-modal perception system in Unity that combines camera and LIDAR data, and demonstrate how this data can be fused for improved environment understanding.

## References

Unity Technologies. (2023). *Unity Robotics Hub*. Retrieved from https://github.com/Unity-Technologies/Unity-Robotics-Hub

Unity Technologies. (2023). *Unity Perception*. Retrieved from https://github.com/Unity-Technologies/com.unity.perception

Paigwar, A., Erkorkmaz, K., & Droniou, A. (2019). ROS#.NET and Unity 3D integration for robotics simulation and programming. *IEEE International Conference on Robotics and Biomimetics*, 1518-1523.

Ghasemi, A., Arab, A., & Kosari, A. (2020). Unity3D as a development platform for virtual reality applications in robotics. *Robotics and Autonomous Systems*, 128, 103488.