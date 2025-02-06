import logging
import onnxruntime as ort
from rtmlib import BodyWithFeet, Wholebody, Body, PoseTracker
import os
from image_utils import frame_enumerator, blur
from scipy.signal import savgol_filter
import scipy.signal as signal
import numpy as np
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


# find keypoint by name
def get_kp(keypoints, keypoint_name):

    if keypoint_name in BODY_KEYPOINTS:
        keypoint_id = BODY_KEYPOINTS[keypoint_name]
        keypoint = keypoints[keypoint_id]
        return keypoint
    else:
        logging.warning(f"Keypoint name '{keypoint_name}' not found in BODY_KEYPOINTS.")
        return None
    



# smoothing with savgol
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
    


def blur_face(image, keypoints, body_height):

    ear_l = get_kp(keypoints, 'LEar')
    ear_r = get_kp(keypoints, 'REar')
    avg_face = [(ear_l[0] + ear_r[0]) / 2, (ear_l[1] + ear_r[1]) / 2]
    radius = body_height / 10

    image = blur(image, avg_face[0], avg_face[1], radius)

    return image