# Save this file as specify/implement/module-2/physics-simulation.md

# Physics Simulation: Gravity, Collisions, Sensors

## Overview

Physics simulation is crucial for realistic robot behavior in digital twin environments. Gazebo uses the Open Dynamics Engine (ODE), Bullet, or DART physics engines to simulate realistic physical interactions.

## Gravity Configuration

Gravity is defined in the world file and affects all objects in the simulation:

```xml
<gravity>0 0 -9.8</gravity>  <!-- Standard Earth gravity -->
```

For different environments:
- Earth: `<gravity>0 0 -9.8</gravity>`
- Moon: `<gravity>0 0 -1.6</gravity>`
- Mars: `<gravity>0 0 -3.7</gravity>`

## Collision Detection

Collision detection in Gazebo is handled through collision elements in model definitions:

```xml
<collision name="collision">
  <geometry>
    <box>
      <size>1.0 1.0 1.0</size>
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <contact>
      <ode>
        <kp>1e+16</kp>
        <kd>1e+13</kd>
      </ode>
    </contact>
  </surface>
</collision>
```

### Friction Parameters

- **mu**: Primary friction coefficient (longitudinal)
- **mu2**: Secondary friction coefficient (lateral)
- **slip1**: Inverse of longitudinal slip compliance
- **slip2**: Inverse of lateral slip compliance

## Physics Parameters

Important physics parameters include:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Key Parameters Explained

- **ERP (Error Reduction Parameter)**: Controls how much of the joint error is corrected per step (0-1, higher = more aggressive correction)
- **CFM (Constraint Force Mixing)**: Adds a small stabilizing force to constraints (positive values for stability)
- **Max Step Size**: Time step for physics simulation (smaller = more accurate but slower)
- **Real Time Factor**: Desired simulation speed relative to real time (1.0 = real-time)

## Material Properties

Different materials can be defined for various objects:

```xml
<material name="rubber">
  <script>
    <uri>file://media/materials/scripts/gazebo.material</uri>
    <name>Gazebo/Rubber</name>
  </script>
</material>

<material name="blue">
  <ambient>0.2 0.3 1.0 1.0</ambient>
  <diffuse>0.2 0.3 1.0 1.0</diffuse>
  <specular>0.2 0.3 1.0 1.0</specular>
  <emissive>0.0 0.0 0.0 1.0</emissive>
</material>
```

## Practical Exercise 3.1: Physics Parameter Optimization

Create a simulation with different materials and friction coefficients. Implement the following:

1. Create 3 objects with different friction coefficients (0.1, 0.5, 1.0)
2. Add a ramp with adjustable angle (0°, 15°, 30°)
3. Observe how different objects behave when placed on the ramp
4. Document the optimal physics parameters for stable simulation without jittering

Test different ERP and CFM values to find the most stable configuration for your simulation.

## References

Open Dynamics Engine. (2023). *ODE User Guide*. http://ode.org/

Gazebo Sim. (2023). *Physics Engine Configuration*. https://gazebosim.org/api/sim/6/physics.html