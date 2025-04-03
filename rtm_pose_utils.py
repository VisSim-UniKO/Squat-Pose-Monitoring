import logging
import onnxruntime as ort
from rtmlib import BodyWithFeet, Wholebody, Body, PoseTracker
import os
from image_utils import frame_enumerator
from scipy.signal import savgol_filter
import scipy.signal as signal
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Set pose keypoints IDs (Halpe26)
BODY_KEYPOINTS = {
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
    "Head": 17,
    "Neck": 18,
    "Hip": 19,
    "LBigToe": 20,
    "RBigToe": 21,
    "LSmallToe": 22,
    "RSmallToe": 23,
    "LHeel": 24,
    "RHeel": 25
}

def initialize_pose_tracker(pose_model='HALPE_26', mode='balanced', det_frequency=1, tracking=False):
    """
    Initialize the pose tracker based on the given parameters.
    """

    # Model class selection based on pose model
    model_map = {
        'HALPE_26': (BodyWithFeet, 'body and feet'),
        'COCO_133': (Wholebody, 'body, feet, hands, and face'),
        'COCO_17': (Body, 'body')
    }

    if pose_model.upper() not in model_map:
        raise ValueError(f"Invalid model_type: {pose_model}. Must be 'HALPE_26', 'COCO_133', or 'COCO_17'.")
    
    ModelClass, model_desc = model_map[pose_model.upper()]
    logging.info(f"Using {pose_model} model ({model_desc}) for pose estimation.")

    # Select backend and device
    device, backend = (
        ('cuda', 'onnxruntime') if 'CUDAExecutionProvider' in ort.get_available_providers() else
        ('mps', 'onnxruntime') if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers() else
        ('cpu', 'openvino')
    )
    
    logging.info(f"Using {backend} backend with {device.upper()}.")
    print("Using", backend, "backend with", device.upper())

    # Initialize and return the pose tracker
    return PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=False
    )



def get_kp(keypoints, keypoint_name, swap_side=False):
    """
    Retrieve the coordinates of a specific keypoint from the keypoints array.

    Args:
        keypoints (list): List of keypoints with their coordinates.
        keypoint_name (str): Name of the keypoint to retrieve (e.g., 'LEar', 'REar').
        swap_side (bool): If True, swaps 'L' and 'R' in the keypoint name (e.g., 'LEar' to 'REar').

    Returns:
        tuple: Coordinates of the keypoint (x, y) or None if the keypoint is not found.
    """
    if keypoint_name in BODY_KEYPOINTS:
        if swap_side:
            prevname = keypoint_name
            # Swap the first letter of keypoint_name from 'R' to 'L' or 'L' to 'R'
            if keypoint_name.startswith('R'):
                keypoint_name = 'L' + keypoint_name[1:]
            elif keypoint_name.startswith('L'):
                keypoint_name = 'R' + keypoint_name[1:]
            print("Swapped keypoint name from", prevname, "to", keypoint_name)
        
        keypoint_id = BODY_KEYPOINTS[keypoint_name]
        keypoint = keypoints[keypoint_id]
        return keypoint
    else:
        logging.warning(f"Keypoint name '{keypoint_name}' not found in BODY_KEYPOINTS.")
        return None



# Function to apply smoothing to the data
def apply_smoothing(data, window_length=15, polyorder=3):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


def init_squats(input_source, pose_tracker, side):

    ankle_height = 0
    body_height = 0
    floor_height = 0
    head_ypositions = []

    for _,frame,_ in frame_enumerator(input_source):

        keypoints, _ = pose_tracker(frame)

        if len(keypoints[0]) == 26:

            # get image coordinates
            head = get_kp(keypoints[0], 'Head')
            small_toe = get_kp(keypoints[0], side + 'SmallToe')
            ankle = get_kp(keypoints[0], side + 'Ankle')
            
            frame_floor_height = small_toe[1]

            # image body height is distance between head and floor
            frame_body_height = abs(head[1] - frame_floor_height)

            # image ankle height is distance between ankle and floor
            frame_ankle_height = abs(ankle[1] - frame_floor_height)

            # update values for the highest body size
            if frame_body_height > body_height:
                body_height = frame_body_height
                ankle_height = frame_ankle_height
                floor_height = frame_floor_height

            # get image coordinates
            head = get_kp(keypoints[0], 'Head')
            head_ypositions.append(head[1])


    # Apply smoothing to the head y-positions
    smoothed_head_ypositions = apply_smoothing(head_ypositions)

    # Detect local maxima in the smoothed data to find peaks
    peak_indices, _ = signal.find_peaks(smoothed_head_ypositions, distance=30, prominence=5)

    valley_indices = []
    for i in range(len(peak_indices)-1):
        # find valley between peaks i and i+1
        valley = np.argmin(smoothed_head_ypositions[peak_indices[i]:peak_indices[i+1]]) + peak_indices[i]
        valley_indices.append(valley)

    # # plot peaks and valleys
    # plt.plot(smoothed_head_ypositions)
    # plt.plot(peak_indices, smoothed_head_ypositions[peak_indices], "x")
    # plt.plot(valley_indices, smoothed_head_ypositions[valley_indices], "o")
    # plt.show()

    squat_segments = [[0, valley_indices[0]]] + [[valley_indices[i], valley_indices[i+1]] for i in range(len(valley_indices)-1)] + [[valley_indices[-1], len(smoothed_head_ypositions)]]

    if not valley_indices:
        squat_segments = [[0, len(smoothed_head_ypositions)]]
        logging.warning(f"Could not find squat segments. Using full video.")

    return floor_height, ankle_height, body_height, squat_segments

# blur face at center of ear keypoints, width region size 0.2 times the body height
import cv2

def blur_face(frame, keypoints, body_height):
    # Get positions of the ears
    left_ear = get_kp(keypoints, 'LEar')
    right_ear = get_kp(keypoints, 'REar')
    
    # Calculate the center and size of the blur region
    blur_center = (
        int((left_ear[0] + right_ear[0]) / 2), 
        int((left_ear[1] + right_ear[1]) / 2)
    )
    blur_size = int(0.2 * body_height)
    
    # Define the bounding box for the blur region
    x1 = max(0, int(blur_center[0] - blur_size / 2))
    y1 = max(0, int(blur_center[1] - blur_size / 2))
    x2 = min(frame.shape[1], int(blur_center[0] + blur_size / 2))
    y2 = min(frame.shape[0], int(blur_center[1] + blur_size / 2))

    kernel_size = max(10, int(0.05 * body_height) // 2 * 2 + 1)  # Kernel size must be odd

    
    # Blur the face region
    face_region = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)

    return frame


def calculate_pose_angle(point1, point2, point3):
    v0 = np.array(point2) - np.array(point1)
    v1 = np.array(point3) - np.array(point2)
    angle = np.arctan2(np.linalg.det([v0, v1]), np.dot(v0, v1))  # Use np.arctan2 instead of np.math.atan2
    angle_deg = np.degrees(angle)
    return angle_deg


def calculate_clockwise_angle(v1_start, v1_end, v2_start, v2_end):
    """
    Calculate the clockwise angle between two vectors defined by their start and end points.

    Args:
        v1_start (tuple): Start point of the first vector (x, y).
        v1_end (tuple): End point of the first vector (x, y).
        v2_start (tuple): Start point of the second vector (x, y).
        v2_end (tuple): End point of the second vector (x, y).

    Returns:
        float: The clockwise angle in degrees (0 to 360).
    """
    # Define the two vectors
    v1 = np.array(v1_end) - np.array(v1_start)
    v2 = np.array(v2_end) - np.array(v2_start)

    # Calculate the signed angle using arctan2
    angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))  # Angle in radians

    # Convert to degrees and normalize to the range [0, 360)
    angle_deg = np.degrees(angle)
    clockwise_angle = angle_deg if angle_deg >= 0 else 360 + angle_deg

    return clockwise_angle