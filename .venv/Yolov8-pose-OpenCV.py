import cv2
import numpy as np
import torch
from ultralytics import YOLO
import ncnn


def draw(kp, frame):
    # Extract the necessary keypoints
    shoulder = kp[5, :2]
    elbow = kp[7, :2]
    wrist = kp[9, :2]
    hip = kp[11, :2]
    knee = kp[13, :2]
    ankle = kp[15, :2]

    # Scale keypoints to frame dimensions
    shoulder = (int(shoulder[0] * frame.shape[1]), int(shoulder[1] * frame.shape[0]))
    elbow = (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0]))
    wrist = (int(wrist[0] * frame.shape[1]), int(wrist[1] * frame.shape[0]))
    hip = (int(hip[0] * frame.shape[1]), int(hip[1] * frame.shape[0]))
    knee = (int(knee[0] * frame.shape[1]), int(knee[1] * frame.shape[0]))
    ankle = (int(ankle[0] * frame.shape[1]), int(ankle[1] * frame.shape[0]))

    # Draw keypoints
    cv2.circle(frame, shoulder, 5, (0, 255, 0), -1)
    cv2.circle(frame, elbow, 5, (0, 255, 0), -1)
    cv2.circle(frame, wrist, 5, (0, 255, 0), -1)
    cv2.circle(frame, hip, 5, (0, 255, 0), -1)
    cv2.circle(frame, knee, 5, (0, 255, 0), -1)
    cv2.circle(frame, ankle, 5, (0, 255, 0), -1)

    # Calculate the knee angle
    knee_angle = calcAngle(hip, knee, ankle)

    # Display the angle on the frame
    cv2.putText(frame, f'Knee Angle: {knee_angle:.2f}', knee, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

    # Draw lines connecting the keypoints
    if(knee_angle >= 170):
        cv2.line(frame, shoulder, elbow, (0, 0, 255), 2)
        cv2.line(frame, elbow, wrist, (0, 0, 255), 2)
        cv2.line(frame, shoulder, hip, (0, 0, 255), 2)
        cv2.line(frame, knee, ankle, (0, 0, 255), 2)
        cv2.line(frame, hip, knee, (0, 0, 255), 2)
        cv2.line(frame, knee, ankle, (0, 0, 255), 2)
    else:
        cv2.line(frame, shoulder, elbow, (0, 255, 0), 2)
        cv2.line(frame, elbow, wrist, (0, 255, 0), 2)
        cv2.line(frame, shoulder, hip, (0, 255, 0), 2)
        cv2.line(frame, knee, ankle, (0, 255, 0), 2)
        cv2.line(frame, hip, knee, (0, 255, 0), 2)
        cv2.line(frame, knee, ankle, (0, 255, 0), 2)

def calcSideLength(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    length = np.linalg.norm(point1 - point2)
    if length == 0:
        length = 0.0001
    return length

def calcAngle(point1, point2, point3):
    """Calculate the angle between three points (in degrees)."""
    # initialize the points
    A = np.array(point1)
    B = np.array(point2)
    C = np.array(point3)

    # Side length calculation
    a = calcSideLength(B, C)
    b = calcSideLength(C, A)
    c = calcSideLength(A, B)

    # Angle calculations
    angleB = (np.arccos((a * a + c * c - b * b) / (2 * a * c))) * 180 / np.pi
    return angleB

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('yolov8n-pose.pt')  # you might need to download or specify the path to the model
    model.export(format="OpenVINO", int8=True, data='coco8-pose.yaml')
    model = YOLO("yolov8n-pose_int8_openvino_model")

    cap = cv2.VideoCapture('Munya.mp4')
    cv2.namedWindow('Pose Estimation', cv2.WINDOW_FULLSCREEN)
    video = cv2.VideoWriter('Inferenced.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(source=frame, conf=0.8)
        for result in results:
            # Access the keypoints from the result
            keypoints = result.keypoints.xyn.cpu().numpy()
            if keypoints.size > 0:  # Check if any keypoints were detected
                for kp in keypoints:
                    draw(kp, frame)

        # Display the frame
        video.write(frame)
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

