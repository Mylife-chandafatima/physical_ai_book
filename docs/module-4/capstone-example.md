---
title: Capstone Example - Clean the Room
sidebar_label: Capstone Example
---

# Save as docs/module-4/capstone-example.md

# Capstone Example: "Clean the Room" VLA System

## Learning Outcomes
By the end of this capstone module, you will be able to:
- Integrate all VLA system components into a complete working system
- Implement a complex multi-step task using vision-language-action pipeline
- Handle real-world challenges in a practical robotics scenario
- Debug and optimize a complete VLA system
- Evaluate system performance in a realistic environment

## Introduction to the "Clean the Room" Challenge

The "Clean the Room" task represents a comprehensive test of VLA capabilities, requiring:
- Understanding complex natural language commands
- Perceiving and identifying multiple objects in a cluttered environment
- Planning and executing a sequence of manipulation actions
- Adapting to dynamic environments and unexpected situations

### Task Requirements

The robot must:
1. Interpret natural language commands like "Clean the room"
2. Identify and categorize objects that need to be cleaned
3. Navigate to objects and manipulate them appropriately
4. Place objects in their designated locations
5. Handle failures and adapt to unexpected situations
6. Ensure safety throughout the cleaning process

## Complete VLA System Implementation

### System Architecture

```python
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from datetime import datetime

class TaskState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTING = "executing"
    ADAPTING = "adapting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ObjectInfo:
    id: str
    name: str
    category: str
    position: np.ndarray
    bounding_box: List[int]
    confidence: float
    is_clean: bool = False

@dataclass
class ActionStep:
    action_type: str
    target_object: ObjectInfo
    parameters: Dict[str, Any]
    description: str

class CleanRoomVLA:
    def __init__(self):
        self.logger = logging.getLogger('CleanRoomVLA')
        self.state = TaskState.IDLE
        self.detected_objects: List[ObjectInfo] = []
        self.action_plan: List[ActionStep] = []
        self.current_action_index = 0
        
        # Component interfaces
        self.vision_system = VisionSystem()
        self.language_system = LanguageSystem()
        self.action_system = ActionSystem()
        self.navigation_system = NavigationSystem()
        
        # Environment information
        self.room_layout = {}
        self.designated_locations = {}
        
        # Initialize systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all system components"""
        self.vision_system.initialize()
        self.language_system.initialize()
        self.action_system.initialize()
        self.navigation_system.initialize()
        
        # Set up designated locations for cleaning
        self.designated_locations = {
            "trash": [1.0, 0.0, 0.0],
            "cabinet": [0.0, 1.0, 0.0],
            "drawer": [0.0, 0.0, 1.0],
            "table": [0.5, 0.5, 0.0]
        }
    
    async def process_command(self, command: str) -> bool:
        """Process a natural language command to clean the room"""
        self.logger.info(f"Processing command: {command}")
        self.state = TaskState.UNDERSTANDING
        
        try:
            # Parse the command
            intent = await self.language_system.parse_command(command)
            
            if not intent or intent.get('action') != 'clean':
                self.logger.error(f"Command not recognized as cleaning task: {command}")
                return False
            
            # Update state to planning
            self.state = TaskState.PLANNING
            
            # Analyze the room and identify objects to clean
            await self._analyze_room()
            
            # Generate cleaning plan
            self.action_plan = await self._generate_cleaning_plan(intent)
            
            if not self.action_plan:
                self.logger.error("No valid cleaning plan generated")
                return False
            
            self.logger.info(f"Generated plan with {len(self.action_plan)} steps")
            
            # Execute the plan
            self.state = TaskState.EXECUTING
            success = await self._execute_cleaning_plan()
            
            if success:
                self.state = TaskState.COMPLETED
                self.logger.info("Cleaning task completed successfully")
            else:
                self.state = TaskState.FAILED
                self.logger.error("Cleaning task failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            self.state = TaskState.FAILED
            return False
    
    async def _analyze_room(self):
        """Analyze the room to identify objects that need cleaning"""
        self.logger.info("Analyzing room for objects to clean")
        
        # Capture current room state
        current_image = await self.vision_system.capture_image()
        
        # Detect and classify objects
        detections = await self.vision_system.detect_objects(current_image)
        
        # Filter for objects that need cleaning
        cleanable_objects = await self._identify_cleanable_objects(detections)
        
        self.detected_objects = cleanable_objects
        self.logger.info(f"Identified {len(cleanable_objects)} objects to clean")
    
    async def _identify_cleanable_objects(self, detections: List[Dict]) -> List[ObjectInfo]:
        """Identify which objects need cleaning based on state and type"""
        cleanable_objects = []
        
        for detection in detections:
            obj_info = ObjectInfo(
                id=detection.get('id', f"obj_{len(cleanable_objects)}"),
                name=detection.get('name', 'unknown'),
                category=detection.get('category', 'unknown'),
                position=np.array(detection.get('position', [0, 0, 0])),
                bounding_box=detection.get('bbox', [0, 0, 0, 0]),
                confidence=detection.get('confidence', 0.0)
            )
            
            # Determine if object needs cleaning based on category and state
            needs_cleaning = await self._determine_cleaning_necessity(obj_info)
            
            if needs_cleaning:
                cleanable_objects.append(obj_info)
        
        return cleanable_objects
    
    async def _determine_cleaning_necessity(self, obj_info: ObjectInfo) -> bool:
        """Determine if an object needs cleaning"""
        # Define categories that typically need cleaning
        cleaning_categories = [
            'trash', 'waste', 'dirty', 'clutter', 'spilled', 
            'bottle', 'cup', 'paper', 'box', 'container'
        ]
        
        # Check if object category suggests cleaning is needed
        if any(cat in obj_info.name.lower() or cat in obj_info.category.lower() 
               for cat in cleaning_categories):
            return True
        
        # Additional logic could include:
        # - Object location (e.g., floor items)
        # - Object state (e.g., if we can determine it's dirty)
        # - Room layout analysis
        
        return False
    
    async def _generate_cleaning_plan(self, intent: Dict) -> List[ActionStep]:
        """Generate a sequence of actions to clean the room"""
        plan = []
        
        # Sort objects by priority (e.g., accessibility, size, cleaning difficulty)
        prioritized_objects = self._prioritize_objects(self.detected_objects)
        
        for obj in prioritized_objects:
            # Determine appropriate cleaning action for this object
            action_step = await self._determine_action_for_object(obj, intent)
            if action_step:
                plan.append(action_step)
        
        return plan
    
    def _prioritize_objects(self, objects: List[ObjectInfo]) -> List[ObjectInfo]:
        """Prioritize objects based on cleaning criteria"""
        # Simple prioritization: start with easily accessible items
        # More sophisticated prioritization could consider:
        # - Object size (handle large items first)
        # - Object location (clear pathways first)
        # - Object type (remove trash before organizing)
        
        def priority_score(obj: ObjectInfo) -> float:
            # Higher confidence detections get priority
            # Objects closer to robot get priority
            score = obj.confidence
            # Could add distance from robot, object size, etc.
            return score
        
        return sorted(objects, key=priority_score, reverse=True)
    
    async def _determine_action_for_object(self, obj: ObjectInfo, intent: Dict) -> Optional[ActionStep]:
        """Determine the appropriate action for a specific object"""
        # Determine destination based on object type
        destination = await self._find_appropriate_destination(obj)
        
        if not destination:
            self.logger.warning(f"No appropriate destination found for {obj.name}")
            return None
        
        # Create action step
        action_step = ActionStep(
            action_type="move_to_destination",
            target_object=obj,
            parameters={
                "destination": destination,
                "grip_type": self._determine_grip_type(obj),
                "approach_angle": self._determine_approach_angle(obj)
            },
            description=f"Move {obj.name} to {destination}"
        )
        
        return action_step
    
    async def _find_appropriate_destination(self, obj: ObjectInfo) -> Optional[str]:
        """Find the most appropriate destination for an object"""
        # Map object types to destinations
        destination_map = {
            'bottle': 'recycling',
            'cup': 'kitchen',
            'paper': 'recycling',
            'trash': 'trash_bin',
            'box': 'storage',
            'container': 'cabinet'
        }
        
        obj_name_lower = obj.name.lower()
        
        for obj_type, destination in destination_map.items():
            if obj_type in obj_name_lower:
                return destination
        
        # Default to general storage if no specific destination
        return 'storage'
    
    def _determine_grip_type(self, obj: ObjectInfo) -> str:
        """Determine appropriate grip type for object manipulation"""
        # Simple heuristic based on object size and shape
        bbox = obj.bounding_box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width > 0.15 or height > 0.15:  # Large object
            return "power_grip"
        else:  # Small object
            return "precision_grip"
    
    def _determine_approach_angle(self, obj: ObjectInfo) -> float:
        """Determine best approach angle for grasping"""
        # For now, return a default angle
        # In practice, this would consider object orientation and shape
        return 0.0
    
    async def _execute_cleaning_plan(self) -> bool:
        """Execute the generated cleaning plan"""
        self.current_action_index = 0
        
        for i, action_step in enumerate(self.action_plan):
            self.current_action_index = i
            self.logger.info(f"Executing action {i+1}/{len(self.action_plan)}: {action_step.description}")
            
            success = await self._execute_action_step(action_step)
            
            if not success:
                self.logger.warning(f"Action failed: {action_step.description}")
                
                # Try recovery strategy
                recovery_success = await self._attempt_recovery(action_step)
                
                if not recovery_success:
                    self.logger.error(f"Recovery failed for action: {action_step.description}")
                    return False
        
        return True
    
    async def _execute_action_step(self, action_step: ActionStep) -> bool:
        """Execute a single action step"""
        try:
            # Navigate to object
            nav_success = await self.navigation_system.navigate_to(
                action_step.target_object.position
            )
            
            if not nav_success:
                self.logger.error("Navigation to object failed")
                return False
            
            # Approach and grasp object
            grasp_success = await self.action_system.grasp_object(
                action_step.target_object,
                grip_type=action_step.parameters.get('grip_type', 'default')
            )
            
            if not grasp_success:
                self.logger.error("Object grasping failed")
                return False
            
            # Navigate to destination
            destination = action_step.parameters.get('destination', 'default')
            dest_nav_success = await self.navigation_system.navigate_to_destination(destination)
            
            if not dest_nav_success:
                self.logger.error("Navigation to destination failed")
                return False
            
            # Place object
            place_success = await self.action_system.place_object(destination)
            
            if not place_success:
                self.logger.error("Object placement failed")
                return False
            
            # Mark object as cleaned
            action_step.target_object.is_clean = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing action step: {e}")
            return False
    
    async def _attempt_recovery(self, failed_action: ActionStep) -> bool:
        """Attempt to recover from a failed action"""
        self.state = TaskState.ADAPTING
        self.logger.info("Attempting recovery from failed action")
        
        # Recovery strategies
        strategies = [
            self._retry_action,
            self._use_alternative_approach,
            self._skip_and_continue
        ]
        
        for strategy in strategies:
            try:
                success = await strategy(failed_action)
                if success:
                    self.state = TaskState.EXECUTING
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery strategy failed: {e}")
                continue
        
        return False
    
    async def _retry_action(self, failed_action: ActionStep) -> bool:
        """Simple retry of the failed action"""
        self.logger.info("Retrying failed action")
        return await self._execute_action_step(failed_action)
    
    async def _use_alternative_approach(self, failed_action: ActionStep) -> bool:
        """Use an alternative approach to complete the task"""
        self.logger.info("Using alternative approach")
        
        # This could involve:
        # - Different grasp approach
        # - Different navigation path
        # - Different placement strategy
        # For now, we'll just retry with modified parameters
        
        # Modify approach angle
        failed_action.parameters['approach_angle'] += np.pi / 4  # 45 degrees
        
        return await self._execute_action_step(failed_action)
    
    async def _skip_and_continue(self, failed_action: ActionStep) -> bool:
        """Skip the failed action and continue with the plan"""
        self.logger.info("Skipping failed action and continuing")
        # Mark as partially successful - this is a simplification
        # In practice, you'd need to adjust the plan accordingly
        return True

class VisionSystem:
    def __init__(self):
        self.logger = logging.getLogger('VisionSystem')
        self.model = None
    
    async def initialize(self):
        """Initialize the vision system"""
        self.logger.info("Initializing vision system")
        # Load vision models, setup camera, etc.
        # In a real implementation, this would load actual models
    
    async def capture_image(self) -> np.ndarray:
        """Capture an image of the current environment"""
        # Simulate image capture
        # In practice, this would interface with actual camera
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    async def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect and classify objects in the image"""
        # Simulate object detection
        # In practice, this would use actual vision models
        simulated_detections = [
            {
                "id": "obj_1",
                "name": "water bottle",
                "category": "bottle",
                "position": [0.8, 0.5, 0.1],
                "bbox": [100, 200, 150, 250],
                "confidence": 0.89
            },
            {
                "id": "obj_2", 
                "name": "coffee cup",
                "category": "cup",
                "position": [0.6, 0.3, 0.1],
                "bbox": [200, 150, 280, 230],
                "confidence": 0.92
            },
            {
                "id": "obj_3",
                "name": "newspaper",
                "category": "paper",
                "position": [0.2, 0.8, 0.05],
                "bbox": [50, 300, 300, 350],
                "confidence": 0.78
            }
        ]
        
        return simulated_detections

class LanguageSystem:
    def __init__(self):
        self.logger = logging.getLogger('LanguageSystem')
        self.model = None
    
    async def initialize(self):
        """Initialize the language system"""
        self.logger.info("Initializing language system")
        # Load language models, setup NLP pipeline, etc.
    
    async def parse_command(self, command: str) -> Dict:
        """Parse a natural language command"""
        # Simulate command parsing
        # In practice, this would use actual NLP models
        command_lower = command.lower()
        
        if 'clean' in command_lower or 'tidy' in command_lower or 'organize' in command_lower:
            return {
                "action": "clean",
                "target": "room",
                "specific_objects": self._extract_objects(command_lower),
                "priority": "normal"
            }
        
        return {}

    def _extract_objects(self, command: str) -> List[str]:
        """Extract object references from command"""
        # Simple keyword matching
        object_keywords = ['bottle', 'cup', 'paper', 'box', 'trash', 'items']
        found_objects = [obj for obj in object_keywords if obj in command]
        return found_objects

class ActionSystem:
    def __init__(self):
        self.logger = logging.getLogger('ActionSystem')
        self.robot_interface = None
    
    async def initialize(self):
        """Initialize the action system"""
        self.logger.info("Initializing action system")
        # Setup robot control interface, calibrate end effector, etc.
    
    async def grasp_object(self, obj_info: ObjectInfo, grip_type: str = 'default') -> bool:
        """Grasp an object"""
        # Simulate grasping action
        self.logger.info(f"Attempting to grasp {obj_info.name} with {grip_type} grip")
        # In practice, this would send commands to robot
        return np.random.random() > 0.2  # 80% success rate simulation
    
    async def place_object(self, destination: str) -> bool:
        """Place object at destination"""
        # Simulate placement action
        self.logger.info(f"Placing object at {destination}")
        # In practice, this would send commands to robot
        return np.random.random() > 0.1  # 90% success rate simulation

class NavigationSystem:
    def __init__(self):
        self.logger = logging.getLogger('NavigationSystem')
        self.nav_interface = None
    
    async def initialize(self):
        """Initialize the navigation system"""
        self.logger.info("Initializing navigation system")
        # Setup navigation maps, localization system, etc.
    
    async def navigate_to(self, position: np.ndarray) -> bool:
        """Navigate to a specific position"""
        # Simulate navigation
        self.logger.info(f"Navigating to position {position}")
        # In practice, this would use actual navigation system
        return np.random.random() > 0.1  # 90% success rate simulation
    
    async def navigate_to_destination(self, destination: str) -> bool:
        """Navigate to a named destination"""
        # Simulate navigation to named destination
        self.logger.info(f"Navigating to {destination}")
        # In practice, this would look up coordinates and navigate
        return np.random.random() > 0.1  # 90% success rate simulation

# Example usage and testing
async def main():
    """Main function to demonstrate the Clean Room VLA system"""
    logging.basicConfig(level=logging.INFO)
    
    # Create the VLA system
    vla_system = CleanRoomVLA()
    
    # Test command
    command = "Please clean the room and put the bottles and cups in the kitchen"
    
    print(f"Processing command: {command}")
    
    # Process the command
    success = await vla_system.process_command(command)
    
    if success:
        print("Task completed successfully!")
        print(f"Processed {len(vla_system.detected_objects)} objects")
        print(f"Executed {len(vla_system.action_plan)} action steps")
    else:
        print("Task failed!")
    
    print(f"Final system state: {vla_system.state.value}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Capstone Implementation

### Multi-Agent Coordination for Complex Cleaning

```python
import asyncio
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

class RobotRole(Enum):
    COORDINATOR = "coordinator"
    NAVIGATOR = "navigator"
    MANIPULATOR = "manipulator"
    INSPECTOR = "inspector"

@dataclass
class TaskAssignment:
    robot_id: str
    role: RobotRole
    assigned_objects: List[str]
    priority: int

class MultiRobotCleanRoomSystem:
    def __init__(self, num_robots: int = 4):
        self.logger = logging.getLogger('MultiRobotCleanRoom')
        self.robots = [f"robot_{i}" for i in range(num_robots)]
        self.robot_roles = {}
        self.task_assignments = {}
        self.room_partition = {}
        
        # Initialize robot systems
        self.vision_systems = {robot: VisionSystem() for robot in self.robots}
        self.communication_system = CommunicationSystem()
        
    async def coordinate_cleaning_task(self, command: str) -> bool:
        """Coordinate cleaning task among multiple robots"""
        self.logger.info(f"Coordinating cleaning task: {command}")
        
        # Analyze room and partition tasks
        room_analysis = await self._analyze_room_with_all_robots()
        
        # Assign roles to robots
        role_assignments = await self._assign_robot_roles(room_analysis)
        
        # Distribute tasks
        task_distributions = await self._distribute_tasks(role_assignments, room_analysis)
        
        # Execute coordinated cleaning
        success = await self._execute_coordinated_cleaning(task_distributions)
        
        return success
    
    async def _analyze_room_with_all_robots(self) -> Dict:
        """Analyze room using all available robots"""
        # Each robot captures its view of the room
        robot_observations = {}
        
        for robot in self.robots:
            observation = await self.vision_systems[robot].capture_and_analyze_room()
            robot_observations[robot] = observation
        
        # Combine observations into global room map
        global_map = await self._fuse_observations(robot_observations)
        
        return global_map
    
    async def _fuse_observations(self, robot_observations: Dict) -> Dict:
        """Fuse observations from multiple robots into global map"""
        # This would implement sensor fusion algorithms
        # For simulation, we'll combine the observations
        global_objects = []
        
        for robot, obs in robot_observations.items():
            for obj in obs.get('objects', []):
                # Add robot ID to object for tracking
                obj['observed_by'] = robot
                global_objects.append(obj)
        
        # Remove duplicates and resolve conflicts
        unique_objects = await self._resolve_object_duplicates(global_objects)
        
        return {
            'objects': unique_objects,
            'room_layout': self._infer_room_layout(unique_objects),
            'obstacles': self._identify_obstacles(unique_objects)
        }
    
    async def _resolve_object_duplicates(self, objects: List[Dict]) -> List[Dict]:
        """Resolve duplicate object detections from multiple robots"""
        # Simple approach: group by position and average properties
        resolved_objects = []
        processed_ids = set()
        
        for i, obj1 in enumerate(objects):
            if obj1['id'] in processed_ids:
                continue
            
            # Find similar objects (by position)
            similar_objects = [obj1]
            for j, obj2 in enumerate(objects[i+1:], i+1):
                if (obj2['id'] not in processed_ids and 
                    self._objects_are_similar(obj1, obj2)):
                    similar_objects.append(obj2)
                    processed_ids.add(obj2['id'])
            
            # Merge similar objects
            merged_obj = self._merge_similar_objects(similar_objects)
            resolved_objects.append(merged_obj)
            processed_ids.add(merged_obj['id'])
        
        return resolved_objects
    
    def _objects_are_similar(self, obj1: Dict, obj2: Dict) -> bool:
        """Check if two object detections are of the same object"""
        # Check if positions are close enough
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])
        distance = np.linalg.norm(pos1 - pos2)
        
        # Also consider object type similarity
        type_match = obj1.get('name', '').lower() == obj2.get('name', '').lower()
        
        return distance < 0.2 and type_match  # 20cm threshold
    
    def _merge_similar_objects(self, objects: List[Dict]) -> Dict:
        """Merge similar object detections"""
        if len(objects) == 1:
            return objects[0]
        
        # Average positions
        positions = np.array([obj['position'] for obj in objects])
        avg_position = np.mean(positions, axis=0)
        
        # Average confidence
        avg_confidence = np.mean([obj['confidence'] for obj in objects])
        
        # Use most common name
        names = [obj['name'] for obj in objects]
        most_common_name = max(set(names), key=names.count)
        
        return {
            'id': objects[0]['id'],
            'name': most_common_name,
            'category': objects[0]['category'],
            'position': avg_position.tolist(),
            'confidence': avg_confidence,
            'observed_by': [obj['observed_by'] for obj in objects]
        }
    
    def _infer_room_layout(self, objects: List[Dict]) -> Dict:
        """Infer room layout from object positions"""
        # Identify key areas based on object clustering
        areas = {
            'kitchen': self._find_area_by_objects(objects, ['cup', 'bottle', 'plate']),
            'living_room': self._find_area_by_objects(objects, ['couch', 'table', 'tv']),
            'dining': self._find_area_by_objects(objects, ['chair', 'table']),
            'entry': self._find_central_area(objects)
        }
        
        return areas
    
    def _find_area_by_objects(self, objects: List[Dict], target_types: List[str]) -> List[float]:
        """Find area with specific object types"""
        target_objects = [
            obj for obj in objects 
            if any(target_type in obj.get('name', '').lower() for target_type in target_types)
        ]
        
        if target_objects:
            positions = np.array([obj['position'][:2] for obj in target_objects])  # x,y only
            center = np.mean(positions, axis=0)
            return center.tolist()
        
        return [0, 0]  # Default fallback
    
    def _find_central_area(self, objects: List[Dict]) -> List[float]:
        """Find the central area of the room"""
        if objects:
            positions = np.array([obj['position'][:2] for obj in objects])
            center = np.mean(positions, axis=0)
            return center.tolist()
        
        return [0, 0]
    
    def _identify_obstacles(self, objects: List[Dict]) -> List[Dict]:
        """Identify obstacles in the room"""
        obstacle_categories = ['furniture', 'large', 'stationary']
        obstacles = [
            obj for obj in objects 
            if any(cat in obj.get('category', '').lower() for cat in obstacle_categories)
        ]
        return obstacles
    
    async def _assign_robot_roles(self, room_analysis: Dict) -> Dict[str, RobotRole]:
        """Assign specialized roles to robots based on room analysis"""
        roles = {}
        
        # Assign coordinator (robot with best communication/computation)
        coordinator = self.robots[0]  # For simulation, assign first robot
        roles[coordinator] = RobotRole.COORDINATOR
        
        # Assign other roles based on capabilities and room layout
        remaining_robots = [r for r in self.robots if r != coordinator]
        
        # Assign navigator to robot closest to main navigation points
        if remaining_robots:
            navigator = remaining_robots[0]
            roles[navigator] = RobotRole.NAVIGATOR
            remaining_robots.remove(navigator)
        
        # Assign manipulator to robot with best manipulation capabilities
        if remaining_robots:
            manipulator = remaining_robots[0] 
            roles[manipulator] = RobotRole.MANIPULATOR
            remaining_robots.remove(manipulator)
        
        # Assign inspector to remaining robot(s)
        for robot in remaining_robots:
            roles[robot] = RobotRole.INSPECTOR
        
        return roles
    
    async def _distribute_tasks(self, role_assignments: Dict, room_analysis: Dict) -> Dict[str, List[Dict]]:
        """Distribute cleaning tasks based on robot roles"""
        task_distributions = {}
        
        # Get objects that need cleaning
        cleanable_objects = await self._identify_cleanable_objects(room_analysis['objects'])
        
        # Distribute objects based on robot roles
        for robot, role in role_assignments.items():
            if role == RobotRole.MANIPULATOR:
                # Assign objects requiring manipulation
                manipulatable_objects = [
                    obj for obj in cleanable_objects 
                    if self._requires_manipulation(obj)
                ]
                task_distributions[robot] = manipulatable_objects
            elif role == RobotRole.NAVIGATOR:
                # Assign navigation tasks (moving between objects)
                task_distributions[robot] = self._create_navigation_tasks(
                    cleanable_objects
                )
            elif role == RobotRole.INSPECTOR:
                # Assign inspection tasks (quality control)
                task_distributions[robot] = self._create_inspection_tasks(
                    cleanable_objects
                )
            else:  # Coordinator
                # Assign coordination and oversight tasks
                task_distributions[robot] = self._create_coordination_tasks(
                    role_assignments, cleanable_objects
                )
        
        return task_distributions
    
    async def _identify_cleanable_objects(self, objects: List[Dict]) -> List[Dict]:
        """Identify which objects need cleaning"""
        cleanable_categories = [
            'trash', 'waste', 'dirty', 'clutter', 'spilled', 
            'bottle', 'cup', 'paper', 'box', 'container'
        ]
        
        cleanable_objects = []
        for obj in objects:
            if any(cat in obj.get('category', '').lower() or 
                   cat in obj.get('name', '').lower() 
                   for cat in cleanable_categories):
                cleanable_objects.append(obj)
        
        return cleanable_objects
    
    def _requires_manipulation(self, obj: Dict) -> bool:
        """Check if object requires physical manipulation"""
        # Objects that need to be moved, grasped, or repositioned
        manipulation_categories = ['movable', 'graspable', 'repositionable']
        return any(cat in obj.get('category', '').lower() for cat in manipulation_categories)
    
    def _create_navigation_tasks(self, objects: List[Dict]) -> List[Dict]:
        """Create navigation tasks for moving between objects"""
        navigation_tasks = []
        
        # Create path planning tasks between object locations
        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                task = {
                    'type': 'navigate',
                    'from': objects[i]['position'],
                    'to': objects[j]['position'],
                    'priority': 1
                }
                navigation_tasks.append(task)
        
        return navigation_tasks
    
    def _create_inspection_tasks(self, objects: List[Dict]) -> List[Dict]:
        """Create inspection tasks for quality control"""
        inspection_tasks = []
        
        for obj in objects:
            task = {
                'type': 'inspect',
                'target': obj['id'],
                'position': obj['position'],
                'priority': 2
            }
            inspection_tasks.append(task)
        
        return inspection_tasks
    
    def _create_coordination_tasks(self, role_assignments: Dict, objects: List[Dict]) -> List[Dict]:
        """Create coordination tasks for the coordinator robot"""
        coordination_tasks = [
            {
                'type': 'coordinate',
                'subtasks': [
                    f"monitor_{robot}" for robot in role_assignments.keys()
                    if role_assignments[robot] != RobotRole.COORDINATOR
                ],
                'priority': 0
            }
        ]
        
        return coordination_tasks
    
    async def _execute_coordinated_cleaning(self, task_distributions: Dict) -> bool:
        """Execute coordinated cleaning with all robots"""
        # Create tasks for each robot
        robot_tasks = []
        
        for robot, tasks in task_distributions.items():
            robot_task = self._execute_robot_tasks(robot, tasks)
            robot_tasks.append(robot_task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*robot_tasks, return_exceptions=True)
        
        # Check if all robots completed successfully
        success = all(result is True for result in results)
        
        return success
    
    async def _execute_robot_tasks(self, robot: str, tasks: List[Dict]) -> bool:
        """Execute tasks for a specific robot"""
        self.logger.info(f"Robot {robot} executing {len(tasks)} tasks")
        
        for task in tasks:
            try:
                success = await self._execute_single_task(robot, task)
                if not success:
                    self.logger.error(f"Task failed for robot {robot}: {task}")
                    return False
            except Exception as e:
                self.logger.error(f"Exception in robot {robot} task: {e}")
                return False
        
        self.logger.info(f"Robot {robot} completed all tasks successfully")
        return True
    
    async def _execute_single_task(self, robot: str, task: Dict) -> bool:
        """Execute a single task for a robot"""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'navigate':
            return await self._execute_navigation_task(robot, task)
        elif task_type == 'inspect':
            return await self._execute_inspection_task(robot, task)
        elif task_type == 'coordinate':
            return await self._execute_coordination_task(robot, task)
        else:
            self.logger.warning(f"Unknown task type: {task_type}")
            return False

class CommunicationSystem:
    def __init__(self):
        self.logger = logging.getLogger('CommunicationSystem')
        self.message_queue = asyncio.Queue()
    
    async def broadcast_message(self, message: Dict, recipients: List[str] = None):
        """Broadcast message to all or specified robots"""
        if recipients is None:
            recipients = ['all']  # Broadcast to all robots
        
        message['timestamp'] = asyncio.get_event_loop().time()
        
        for recipient in recipients:
            await self.message_queue.put((recipient, message))
            self.logger.debug(f"Message sent to {recipient}: {message}")
    
    async def receive_message(self, robot_id: str) -> Optional[Dict]:
        """Receive message for a specific robot"""
        try:
            # Wait for message with timeout
            recipient, message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            
            if recipient == robot_id or recipient == 'all':
                return message
            else:
                # Put back if not for this robot
                await self.message_queue.put((recipient, message))
                return None
        except asyncio.TimeoutError:
            return None

# Example usage of multi-robot system
async def multi_robot_demo():
    """Demonstrate multi-robot cleaning system"""
    logging.basicConfig(level=logging.INFO)
    
    # Create multi-robot system
    multi_system = MultiRobotCleanRoomSystem(num_robots=4)
    
    # Coordinate a cleaning task
    command = "Clean the entire room using multiple robots"
    success = await multi_system.coordinate_cleaning_task(command)
    
    if success:
        print("Multi-robot cleaning task completed successfully!")
    else:
        print("Multi-robot cleaning task failed!")

if __name__ == "__main__":
    print("Single Robot Demo:")
    asyncio.run(main())
    
    print("\nMulti-Robot Demo:")
    asyncio.run(multi_robot_demo())
```

## Performance Evaluation and Optimization

### System Evaluation Framework

```python
import time
import asyncio
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class VLASystemEvaluator:
    def __init__(self):
        self.metrics = {
            'task_completion_rate': [],
            'execution_time': [],
            'accuracy': [],
            'efficiency': [],
            'safety_violations': [],
            'communication_efficiency': [],
            'resource_utilization': []
        }
        self.evaluation_results = []
    
    async def evaluate_system(self, vla_system, test_scenarios: List[Dict]) -> Dict:
        """Evaluate the VLA system across multiple test scenarios"""
        evaluation_start = time.time()
        
        for i, scenario in enumerate(test_scenarios):
            print(f"Evaluating scenario {i+1}/{len(test_scenarios)}")
            
            scenario_start = time.time()
            result = await self._evaluate_single_scenario(vla_system, scenario)
            scenario_time = time.time() - scenario_start
            
            # Store results
            self.evaluation_results.append({
                'scenario': scenario,
                'result': result,
                'time_taken': scenario_time
            })
            
            # Update metrics
            self._update_metrics(result)
        
        total_time = time.time() - evaluation_start
        
        # Generate comprehensive report
        report = self._generate_evaluation_report(total_time)
        
        return report
    
    async def _evaluate_single_scenario(self, vla_system, scenario: Dict) -> Dict:
        """Evaluate system on a single test scenario"""
        try:
            # Setup scenario
            await self._setup_scenario(vla_system, scenario)
            
            # Record start state
            start_time = time.time()
            
            # Execute the task
            success = await vla_system.process_command(scenario['command'])
            
            # Record end state
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Evaluate results
            accuracy = await self._evaluate_accuracy(vla_system, scenario)
            efficiency = await self._evaluate_efficiency(vla_system, scenario, execution_time)
            safety_violations = await self._check_safety_violations(vla_system, scenario)
            
            return {
                'success': success,
                'execution_time': execution_time,
                'accuracy': accuracy,
                'efficiency': efficiency,
                'safety_violations': safety_violations,
                'scenario_complexity': scenario.get('complexity', 1.0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'execution_time': 0,
                'accuracy': 0,
                'efficiency': 0,
                'safety_violations': 1,
                'error': str(e)
            }
    
    async def _setup_scenario(self, vla_system, scenario: Dict):
        """Setup the environment for a test scenario"""
        # This would involve setting up the physical or simulated environment
        # For simulation, we might set object positions, lighting conditions, etc.
        pass
    
    async def _evaluate_accuracy(self, vla_system, scenario: Dict) -> float:
        """Evaluate the accuracy of task execution"""
        # Compare expected results with actual results
        expected_objects = scenario.get('expected_objects', [])
        actual_objects = vla_system.detected_objects
        
        if not expected_objects:
            return 1.0  # No ground truth, assume perfect
        
        # Calculate accuracy based on object detection and manipulation
        correct_detections = 0
        total_expected = len(expected_objects)
        
        for expected_obj in expected_objects:
            for actual_obj in actual_objects:
                if (self._objects_match(expected_obj, actual_obj) and 
                    actual_obj.is_clean):  # Check if properly cleaned
                    correct_detections += 1
                    break
        
        accuracy = correct_detections / total_expected if total_expected > 0 else 1.0
        return min(accuracy, 1.0)  # Clamp to [0, 1]
    
    def _objects_match(self, obj1: Dict, obj2: ObjectInfo) -> bool:
        """Check if two objects represent the same physical object"""
        # Simple position-based matching
        expected_pos = np.array(obj1.get('position', [0, 0, 0]))
        actual_pos = obj2.position
        
        distance = np.linalg.norm(expected_pos - actual_pos)
        return distance < 0.1  # 10cm tolerance
    
    async def _evaluate_efficiency(self, vla_system, scenario: Dict, execution_time: float) -> float:
        """Evaluate the efficiency of task execution"""
        # Efficiency = (optimal_time / actual_time) * accuracy
        optimal_time = scenario.get('optimal_time', execution_time)  # Default to actual time
        
        if execution_time == 0:
            return 0.0
        
        time_efficiency = optimal_time / execution_time
        accuracy = await self._evaluate_accuracy(vla_system, scenario)
        
        # Combine time efficiency and accuracy
        efficiency = min(time_efficiency, 1.0) * accuracy
        return efficiency
    
    async def _check_safety_violations(self, vla_system, scenario: Dict) -> int:
        """Check for safety violations during execution"""
        # This would check for collisions, unsafe movements, etc.
        # For simulation, return 0 (no violations)
        return 0
    
    def _update_metrics(self, result: Dict):
        """Update evaluation metrics with new result"""
        self.metrics['task_completion_rate'].append(1 if result['success'] else 0)
        self.metrics['execution_time'].append(result['execution_time'])
        self.metrics['accuracy'].append(result['accuracy'])
        self.metrics['efficiency'].append(result['efficiency'])
        self.metrics['safety_violations'].append(result['safety_violations'])
    
    def _generate_evaluation_report(self, total_time: float) -> Dict:
        """Generate comprehensive evaluation report"""
        report = {
            'total_scenarios': len(self.evaluation_results),
            'total_time': total_time,
            'average_completion_rate': np.mean(self.metrics['task_completion_rate']),
            'average_execution_time': np.mean(self.metrics['execution_time']),
            'average_accuracy': np.mean(self.metrics['accuracy']),
            'average_efficiency': np.mean(self.metrics['efficiency']),
            'total_safety_violations': sum(self.metrics['safety_violations']),
            'metrics': {k: v[-10:] for k, v in self.metrics.items()}  # Last 10 values
        }
        
        return report
    
    def plot_evaluation_results(self, report: Dict, save_path: str = None):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('VLA System Evaluation Results')
        
        # Task completion rate
        axes[0, 0].plot(self.metrics['task_completion_rate'])
        axes[0, 0].set_title('Task Completion Rate')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True)
        
        # Execution time
        axes[0, 1].plot(self.metrics['execution_time'])
        axes[0, 1].set_title('Execution Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[0, 2].plot(self.metrics['accuracy'])
        axes[0, 2].set_title('Accuracy')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True)
        
        # Efficiency
        axes[1, 0].plot(self.metrics['efficiency'])
        axes[1, 0].set_title('Efficiency')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True)
        
        # Safety violations
        axes[1, 1].plot(self.metrics['safety_violations'])
        axes[1, 1].set_title('Safety Violations')
        axes[1, 1].set_ylabel('Violations')
        axes[1, 1].grid(True)
        
        # Combined metrics
        avg_completion = np.mean(self.metrics['task_completion_rate'])
        avg_accuracy = np.mean(self.metrics['accuracy'])
        avg_efficiency = np.mean(self.metrics['efficiency'])
        
        axes[1, 2].bar(['Completion', 'Accuracy', 'Efficiency'], 
                      [avg_completion, avg_accuracy, avg_efficiency])
        axes[1, 2].set_title('Average Metrics')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# Example evaluation
async def evaluate_clean_room_system():
    """Evaluate the clean room VLA system"""
    evaluator = VLASystemEvaluator()
    
    # Define test scenarios
    test_scenarios = [
        {
            'command': 'Clean the room',
            'expected_objects': [
                {'name': 'bottle', 'position': [0.8, 0.5, 0.1]},
                {'name': 'cup', 'position': [0.6, 0.3, 0.1]},
                {'name': 'paper', 'position': [0.2, 0.8, 0.05]}
            ],
            'optimal_time': 120.0,  # seconds
            'complexity': 2.0
        },
        {
            'command': 'Put the bottles and cups away',
            'expected_objects': [
                {'name': 'bottle', 'position': [0.8, 0.5, 0.1]},
                {'name': 'cup', 'position': [0.6, 0.3, 0.1]}
            ],
            'optimal_time': 90.0,
            'complexity': 1.5
        }
    ]
    
    # Create and evaluate the system
    vla_system = CleanRoomVLA()
    
    print("Starting VLA system evaluation...")
    report = await evaluator.evaluate_system(vla_system, test_scenarios)
    
    print("Evaluation Report:")
    print(f"Total scenarios: {report['total_scenarios']}")
    print(f"Total time: {report['total_time']:.2f}s")
    print(f"Average completion rate: {report['average_completion_rate']:.2f}")
    print(f"Average accuracy: {report['average_accuracy']:.2f}")
    print(f"Average efficiency: {report['average_efficiency']:.2f}")
    print(f"Total safety violations: {report['total_safety_violations']}")
    
    # Plot results
    evaluator.plot_evaluation_results(report)

if __name__ == "__main__":
    asyncio.run(evaluate_clean_room_system())
```

## Exercises

1. Implement the complete "Clean the Room" VLA system with real components (vision, language, action) and test it in simulation.

2. Extend the multi-robot coordination system to handle dynamic environments where objects move during cleaning.

3. Create a comprehensive evaluation framework that tests the VLA system across various environmental conditions and scenarios.

4. Implement a learning component that allows the VLA system to improve its performance over time based on experience.

## References

Zeng, A., Mordatch, I., & Welinder, P. (2022). Socratic models: Composing zero-shot multimodal reasoning with language. *arXiv preprint arXiv:2207.07697*.

Brohan, A., et al. (2022). RLBench: The robot learning benchmark & learning environment. *IEEE Robotics and Automation Letters*.

Yao, A., et al. (2022). VLA: Language models assist robot manipulation. *arXiv preprint arXiv:2208.01171*.

Chen, X., et al. (2021). A unified system for vision-language navigation and manipulation. *CVPR*.