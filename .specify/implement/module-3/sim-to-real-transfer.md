# Save this file as specify/implement/module-3/sim-to-real-transfer.md

# Sim-to-Real Transfer Techniques

## Overview

Sim-to-real transfer enables models trained in simulation to perform effectively on physical robots. This requires addressing the reality gap between simulated and real environments. NVIDIA Isaac provides tools and techniques to bridge this gap effectively.

## Domain Randomization

Domain randomization involves randomizing simulation parameters to improve transfer learning:

```python
# Example domain randomization in Isaac Sim
import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage

def randomize_lighting():
    """Randomize lighting conditions in the simulation"""
    # Randomize key light
    key_light = get_prim_at_path("/World/key_light")
    intensity = np.random.uniform(100, 1000)
    key_light.GetAttribute("intensity").Set(intensity)
    
    # Randomize light color
    color = np.random.uniform(0.5, 1.0, 3)
    key_light.GetAttribute("color").Set(color)
    
    # Randomize fill light
    fill_light = get_prim_at_path("/World/fill_light")
    intensity = np.random.uniform(50, 500)
    fill_light.GetAttribute("intensity").Set(intensity)

def randomize_materials():
    """Randomize object materials in the simulation"""
    # Randomize object textures
    material_prim = get_prim_at_path("/World/Object/Materials/PreviewSurface")
    
    # Randomize roughness (0.0 to 1.0)
    roughness = np.random.uniform(0.1, 0.9)
    material_prim.GetAttribute("inputs:roughness").Set(roughness)
    
    # Randomize metallic (0.0 to 1.0)
    metallic = np.random.uniform(0.0, 1.0)
    material_prim.GetAttribute("inputs:metallic").Set(metallic)
    
    # Randomize color
    color = np.random.uniform(0.0, 1.0, 3)
    material_prim.GetAttribute("inputs:diffuse_tint").Set(color)

def randomize_physics():
    """Randomize physics properties in the simulation"""
    # Randomize friction coefficients
    articulation_root = get_prim_at_path("/World/Robot")
    
    # For each joint, randomize friction
    for joint_path in ["/World/Robot/Joint1", "/World/Robot/Joint2"]:
        joint_prim = get_prim_at_path(joint_path)
        if joint_prim:
            # Randomize joint friction
            friction = np.random.uniform(0.0, 1.0)
            joint_prim.GetAttribute("physics:friction").Set(friction)

def apply_domain_randomization():
    """Apply all domain randomization techniques"""
    randomize_lighting()
    randomize_materials()
    randomize_physics()
```

## Domain Adaptation

Domain adaptation techniques help improve model performance on real-world data:

### Adversarial Domain Adaptation

```python
import torch
import torch.nn as nn

class DomainAdversarialNetwork(nn.Module):
    def __init__(self, feature_extractor, classifier, domain_classifier):
        super(DomainAdversarialNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
        
    def forward(self, x, alpha=0.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Classify (for main task)
        class_pred = self.classifier(features)
        
        # Classify domain (for adaptation)
        domain_pred = self.domain_classifier(features, alpha)
        
        return class_pred, domain_pred

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, alpha):
        x = GradientReversalLayer.apply(x, alpha)
        return self.domain_classifier(x)
```

## Texture Randomization

```python
# Example texture randomization
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdShade, Sdf

def randomize_textures():
    """Randomize textures in the simulation"""
    # Get material prim
    material_prim = get_prim_at_path("/World/Object/Materials/PreviewSurface")
    
    # Create random texture variations
    texture_options = [
        "textures/wood_01.png",
        "textures/metal_01.png",
        "textures/plastic_01.png",
        "textures/concrete_01.png"
    ]
    
    # Randomly select texture
    import random
    selected_texture = random.choice(texture_options)
    
    # Apply texture to material
    shader = UsdShade.Shader.Define(material_prim.GetStage(), 
                                   Sdf.Path("/World/Object/Materials/PreviewSurface/Texture"))
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(selected_texture)
```

## Physics Randomization

```python
# Physics parameter randomization
def randomize_robot_dynamics(robot_prim_path):
    """Randomize robot dynamics parameters"""
    # Randomize mass properties
    link_prim = get_prim_at_path(f"{robot_prim_path}/base_link")
    
    # Randomize mass (±20% variation)
    base_mass = 1.0  # default mass
    randomized_mass = base_mass * np.random.uniform(0.8, 1.2)
    
    # Randomize friction (0.1 to 1.0)
    friction = np.random.uniform(0.1, 1.0)
    
    # Randomize damping
    linear_damping = np.random.uniform(0.01, 0.1)
    angular_damping = np.random.uniform(0.01, 0.1)
    
    # Apply parameters
    link_prim.GetAttribute("physics:mass").Set(randomized_mass)
    link_prim.GetAttribute("physics:linearDamping").Set(linear_damping)
    link_prim.GetAttribute("physics:angularDamping").Set(angular_damping)
```

## System Identification

System identification helps calibrate simulation parameters to match real robot behavior:

```python
# Example system identification
import numpy as np
from scipy.optimize import minimize

def system_identification(real_data, sim_data):
    """Identify simulation parameters to match real robot behavior"""
    
    def objective_function(params):
        # Update simulation with new parameters
        update_sim_params(params)
        
        # Run simulation and collect data
        sim_output = run_simulation()
        
        # Calculate difference with real data
        error = np.mean((real_data - sim_output) ** 2)
        return error
    
    # Initial parameter guess
    initial_params = [1.0, 0.1, 0.05]  # mass, friction, damping
    
    # Optimize parameters
    result = minimize(objective_function, initial_params, method='BFGS')
    
    return result.x

def update_sim_params(params):
    """Update simulation with identified parameters"""
    mass, friction, damping = params
    # Apply parameters to simulation
    pass

def run_simulation():
    """Run simulation and return output data"""
    # Run simulation
    # Return relevant data
    pass
```

## Synthetic-to-Real Data Augmentation

```python
import cv2
import numpy as np

def augment_synthetic_data(synthetic_image):
    """Apply real-world style augmentations to synthetic data"""
    
    # Add noise to simulate real sensor noise
    noise = np.random.normal(0, 0.01, synthetic_image.shape)
    noisy_image = np.clip(synthetic_image + noise, 0, 1)
    
    # Add blur to simulate camera imperfections
    kernel_size = np.random.randint(1, 3)
    blurred_image = cv2.GaussianBlur(noisy_image, (kernel_size*2+1, kernel_size*2+1), 0)
    
    # Adjust brightness and contrast
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    
    adjusted_image = np.clip(blurred_image * contrast + (1 - contrast) / 2 + (brightness - 1), 0, 1)
    
    return adjusted_image
```

## Practical Exercise 5.1: Domain Randomization Implementation

Implement domain randomization for a perception task (e.g., object detection) in Isaac Sim:

1. Create a simulation environment with objects to detect
2. Implement domain randomization for lighting, materials, and textures
3. Train a model using the randomized synthetic data
4. Test the model's performance on real-world data
5. Apply domain adaptation techniques to improve sim-to-real transfer
6. Document the improvement in performance

Compare the performance of:
- A model trained on non-randomized synthetic data
- A model trained on randomized synthetic data
- A model with domain adaptation techniques applied

Analyze the results and provide recommendations for effective sim-to-real transfer.

## References

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 23-30.

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *The Journal of Machine Learning Research*, 17(1), 2096-2030.

Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *2018 IEEE International Conference on Robotics and Automation (ICRA)*, 1-8.