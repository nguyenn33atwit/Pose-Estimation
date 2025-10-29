# Pose-Estimation
🧍‍♂️ Raspberry Pi Pose Estimation
This project implements real-time human pose estimation on a Raspberry Pi 5 using the YOLOv8-pose model. The system detects key body joints from a live camera feed, calculates joint angles, and provides instant visual feedback on posture quality — perfect for fitness tracking, physical therapy, or ergonomic correction.

🚀 Features
Real-Time Pose Detection:
Runs a YOLOv8-pose model on the Raspberry Pi to detect body keypoints in live video.
Joint Angle Calculation:
Uses NumPy to compute angles for key joints such as:
Knee
Hip
Back

Dynamic Feedback Visualization:
Displays a skeleton overlay on the live video using OpenCV.

Joints dynamically change color:
🟢 Green – good posture/form
🔴 Red – needs correction

Lightweight and Efficient:
Optimized for the Raspberry Pi 5’s hardware constraints.
