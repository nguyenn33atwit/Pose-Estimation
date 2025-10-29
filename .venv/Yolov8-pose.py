import cv2
import numpy as np
import torch
from ultralytics import YOLO
import openvino.properties.hint as hints

def calcSideLength(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    length = np.linalg.norm(point1-point2)
    return length

def calcAngle(point1, point2, point3):
    """Calculate the angle between three points (in degrees)."""
    #initialize the points
    A = np.array(point1)
    B = np.array(point2)
    C = np.array(point3)

    #Side length calculation
    a = calcSideLength(B,C)
    b = calcSideLength(C,A)
    c = calcSideLength(A,B)

    #Angle calculations
    angleB = (np.arccos((a*a + c*c - b*b)/(2*a*c)))*180/np.pi
    return angleB

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('yolov8s-pose.pt')  # you might need to download or specify the path to the model
    model.export(format="openvino", dynamic=True, int8=True, data='coco8-pose.yaml')
    ov_model = YOLO("yolov8s-pose_int8_openvino_model")
    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}

    results = ov_model(source="Munya.mp4", show=True, conf=0.8, save=True, stream=True)
    for result in results:
        # Access the keypoints from the result
        keypoints = result.keypoints.xyn.cpu().numpy()
        if keypoints.size > 0:  # Check if any keypoints were detected
            print(calcAngle(keypoints[:,11,:2], keypoints[:,13,:2], keypoints[:,15,:2]))