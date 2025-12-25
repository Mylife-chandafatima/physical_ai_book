import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics Book',
      items: [
        {
          type: 'link',
          label: 'Chat with Book',
          href: '/chat',
        },
        {
          type: 'category',
          label: 'Module 1 - Robotic Nervous System (ROS 2)',
          items: [
            'module-1/introduction',
            'module-1/ros2-nodes',
            'module-1/topics-messaging',
            'module-1/nodes-topics-services',
            'module-1/bridging-python-ros',
            'module-1/urdf-humanoids'
          ]
        },
        {
          type: 'category',
          label: 'Module 2 - Digital Twin (Gazebo & Unity)',
          items: [
            'module-2/introduction',
            'module-2/introduction-to-digital-twin',
            'module-2/gazebo-simulation-setup',
            'module-2/physics-simulation',
            'module-2/gazebo-physics-sensors',
            'module-2/unity-rendering-hri'
          ]
        },
        {
          type: 'category',
          label: 'Module 3 - AI-Robot Brain (NVIDIA Isaac)',
          items: [
            'module-3/introduction',
            'module-3/introduction-to-ai-robot-brain',
            'module-3/nvidia-isaac-sim',
            'module-3/isaac-sim',
            'module-3/isaac-ros',
            'module-3/isaac-ros-vslam-navigation',
            'module-3/nav2-path-planning',
            'module-3/reinforcement-learning-control',
            'module-3/advanced-perception-pipelines',
            'module-3/sim-to-real-transfer'
          ]
        },
        {
          type: 'category',
          label: 'Module 4 - Vision-Language-Action (VLA)',
          items: [
            'module-4/introduction-to-vla-concepts',
            'module-4/voice-to-action-whisper',
            'module-4/cognitive-planning-llms',
            'module-4/natural-language-ros2-actions',
            'module-4/object-identification-manipulation',
            'module-4/testing-debugging',
            'module-4/capstone-example'
          ]

        }
      ]
    }
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
