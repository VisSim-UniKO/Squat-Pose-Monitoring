import cv2
import numpy as np
import os
# from rtmlib import draw_skeleton
import logging
import time
from rtm_pose_utils import initialize_pose_tracker, get_kp, init_squats, blur_face, calculate_pose_angle, calculate_clockwise_angle
from image_utils import frame_enumerator, draw_skeleton, draw_arrow, images_to_video, fullscreen_display, add_overlay, generate_video_with_audio
import pandas as pd
import sys
import cv2
import threading
from realsense_utils import create_pipelines
from enum import Enum
from datetime import timedelta
from gtts import gTTS
import pygame
import glob
import subprocess
import tempfile
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip, AudioClip
from tqdm import tqdm

FPS = 30
init_time_threshold = 2
useAudioFeedback = True
err_knee_thrs_factor = 0.01
knee_angle_threshold = 5

class State(Enum):
    INIT = 1
    CALIBRATION = 2
    EXERCISING = 3
    FINISHING = 4

class SquatState(Enum):
    INIT = 0
    ACTIVE = 1
    INACTIVE = 2

state = State.INIT
squat_state = SquatState.INACTIVE

# Directory where audio files will be stored
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)



def set_squat_variables():
    global track_squat_conditions

    if PERSPECTIVE == "frontal":
        track_squat_conditions = {
            "Left_Knee_Angle": True,
            "Right_Knee_Angle": True,
        }

    elif PERSPECTIVE.startswith("lateral"):
        track_squat_conditions = {
            "Heel": True,
            "Femur": True,
            "Knee": True
        }


def process_image(pose_tracker, frame, perspective):

    global squat_state
    global track_squat_conditions
    global timestamp
    overlays = []

    # exclude side of the image for pose estimation
    temp_frame = frame.copy()
    temp_frame[:, :int(frame.shape[1] * 0.25)] = np.ones((temp_frame.shape[0], int(temp_frame.shape[1] * 0.25), 3), dtype=np.uint8) * 255

    # Perform pose estimation on the frame
    keypoints, scores = pose_tracker(temp_frame)

    # Draw skeleton
    if perspective.startswith("lateral"):   # for lateral perspective
        side = 'right' if perspective.endswith("right") else 'left'
        selected_keypoints = [side + '_hip', side + '_knee', side + '_ankle', side + '_big_toe']

    elif perspective.startswith("frontal"): # for frontal perspective
        selected_keypoints = ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'right_big_toe', 'left_big_toe']

    img_show = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5, radius=skeleton_kp_radius, line_width=skeleton_kp_radius//2,
                            selected_kpts=selected_keypoints)
    
    # blur face
    try:
        img_show = blur_face(img_show, keypoints[0], BODY_HEIGHT)
    except Exception as e:
        print(f"Error blurring face: {e}")



    ############################ CHECK SQUAT CONDITIONS ################################

    if len(keypoints[0]) == 26: # if body is visible (26 keypoints visible)

        # SET SQUAT STATE AND AUDIO FEEDBACK

        head  = get_kp(keypoints[0], "Head")
        frame_body_height = abs(head[1] - FLOOR_HEIGHT)
        diff_body_height = abs(frame_body_height - BODY_HEIGHT)

        # if the body height changes by more than 10% of the original body height
        if diff_body_height > 0.1 * BODY_HEIGHT:
            if squat_state == SquatState.INACTIVE:
                set_squat_variables()
            squat_state = SquatState.ACTIVE
        else:
            if squat_state == SquatState.ACTIVE:
                # Check if all conditions are False
                if all(value == True for value in track_squat_conditions.values()):
                    audio = "sound_positive.mp3"  # All conditions True, so play positive sound
                else:
                    audio = "sound_negative.mp3" # If any condition is False, play negative sound
                play_audio(audio, timestamp)
            squat_state = SquatState.INACTIVE


        # CHECK SQUAT CONDITIONS FOR LATERAL PERSPECTIVE

        if perspective.startswith("lateral"):
            BODY_SIDE = 'R' if perspective.endswith("right") else 'L'
            hip   = get_kp(keypoints[0], BODY_SIDE + "Hip")
            knee  = get_kp(keypoints[0], BODY_SIDE + "Knee")
            ankle = get_kp(keypoints[0], BODY_SIDE + "Ankle")
            toe   = get_kp(keypoints[0], BODY_SIDE + "BigToe")

            # check condition 1: Heels remain flat on the ground
            ankle_height = abs(ankle[1] - FLOOR_HEIGHT)
            frame_rel_ankle_height = (ankle_height / BODY_HEIGHT) * 100
            error_heel = round(max(0,(frame_rel_ankle_height - REL_ANKLE_HEIGHT)),2)
            condition_heel = error_heel < 1

            # check condition 2: Femur not below knee
            condition_femur = hip[1] < knee[1]

            # check condition 3: knee joint does not exceed the vertical extension of the toe key point during flexion
            error_knee = abs(max(0, knee[0] - toe[0]))
            if perspective.endswith("right"):
                condition_knee = error_knee < error_knee_threshold
            else:
                condition_knee = error_knee > error_knee_threshold

            # Define conditions and corresponding labels
            conditions = {
                "Heel": condition_heel,
                "Femur": condition_femur,
                "Knee": condition_knee
            }

            condition_markers = {
                "Heel": [ankle, (ankle[0], ankle[1]+20)],
                "Femur": [hip, (hip[0], knee[1])],
                "Knee": [knee, (toe[0], knee[1])]
            }

            conditions_feedback = {
                "Heel": "Keep your feet on the ground!",
                "Femur": "Adjust hip position!",
                "Knee": "Adjust knee position!"
            }
        
        # CHECK SQUAT CONDITIONS FOR FRONTAL PERSPECTIVE

        elif perspective.startswith("frontal"):
            right_hip     = get_kp(keypoints[0], "RHip")
            left_hip      = get_kp(keypoints[0], "LHip")
            right_knee    = get_kp(keypoints[0], "RKnee")
            left_knee     = get_kp(keypoints[0], "LKnee")
            right_ankle   = get_kp(keypoints[0], "RAnkle")
            left_ankle    = get_kp(keypoints[0], "LAnkle")
            
            # Check condition 1: Right knee angle
            # right_knee_angle = calculate_pose_angle(right_hip, right_knee, right_ankle)
            right_knee_angle = calculate_clockwise_angle(right_knee, right_ankle, right_knee, right_hip)
            # angle should be between 165 and 190 degrees
            condition_right_knee = 165 < right_knee_angle < 190

            # Check condition 2: Left knee angle
            # left_knee_angle = calculate_pose_angle(left_hip, left_knee, left_ankle)
            left_knee_angle = calculate_clockwise_angle(left_knee, left_hip, left_knee, left_ankle)
            condition_left_knee = 165 < left_knee_angle < 190

            # Define conditions and corresponding labels
            conditions = {
                "Left_Knee_Angle": condition_left_knee,
                "Right_Knee_Angle": condition_right_knee
            }

            condition_markers = {
                "Left_Knee_Angle": [left_knee, (int((left_hip[0]+left_ankle[0])/2), left_knee[1])],
                "Right_Knee_Angle": [right_knee, (int((right_hip[0]+right_ankle[0])/2), right_knee[1])]
            }

            conditions_feedback = {
                "Left_Knee_Angle": "Adjust left knee position!",
                "Right_Knee_Angle": "Adjust right knee position!"
            }

            # Get overlay images to display feedback
            if condition_left_knee:
                overlays.append(("./pictograms/ampel_green.png", 0.1, (0.2, 0.5)))
            else:
                overlays.append(("./pictograms/ampel_red.png", 0.1, (0.2, 0.5)))
            if condition_right_knee:
                overlays.append(("./pictograms/ampel_green.png", 0.1, (0.8, 0.5)))
            else:
                overlays.append(("./pictograms/ampel_red.png", 0.1, (0.8, 0.5)))


        # Draw error labels in the white space at the bottom with color based on condition status
        for i, (label, value) in enumerate(conditions.items()):
            if not conditions[label]:
                # print(f"Condition {label} not met.")
                # Draw red circle at condition keypoint and a small arrow pointing upwards
                kp = condition_markers[label][0]
                tip = condition_markers[label][1]
                cv2.circle(img_show, (int(kp[0]), int(kp[1])), skeleton_kp_radius, (0, 0, 255), -1)
                cv2.arrowedLine(img_show, (int(kp[0]), int(kp[1])), (int(tip[0]), int(tip[1])), (0, 0, 255), skeleton_kp_radius // 2, tipLength=0.3)

                if squat_state == SquatState.ACTIVE:
                    track_squat_conditions[label] = False
                    # print(f"Tracked Condition {label} updated.")

            if not track_squat_conditions[label]:
                # put text over image
                cv2.putText(img_show, conditions_feedback[label], (start_x, start_y + i * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness, cv2.LINE_AA)

        timestamp += 1 / FPS

        overlays = []

    return img_show, overlays



def workout_monitoring(mode=None, input=None, output=None):
    # Reset global variables
    reset_global_variables()

    # Initialize the pose tracker
    pose_tracker = initialize_pose_tracker(pose_model='HALPE_26', mode='balanced', det_frequency=1, tracking=False)

    if mode == "realsense":
        print("Starting RealSense pose estimation.")
        print("Press 'q' to exit.")
        # Start RealSense stream (not implemented here)
        # workout_monitor_realsense(int(input))  # TODO

    if mode == "video":
        print("Starting pose estimation from video data.")

        if output.lower().endswith(".mp4"):
            video_audio_path = output
        else:
            os.makedirs(output, exist_ok=True)
            video_audio_path = os.path.join(output, "squat-monitor-output.mp4")

        # Use frame_enumerator to handle both directories and video files
        frame_generator = frame_enumerator(input)

        # Initialize variables
        video_writer = None
        first_frame_processed = False  # Flag to ensure calibration and squat variables are set only once

        # Use a temporary file for the intermediate video without audio
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            temp_video_path = temp_video_file.name

        # Process frames with tqdm progress bar
        for frame_index, frame, frame_name in tqdm(frame_generator, desc="Processing frames", unit="frame"):
            # Initialize video writer on the first frame
            if video_writer is None:
                height, width, _ = frame.shape
                frame_rate = FPS  # Use global FPS
                video_writer = cv2.VideoWriter(
                    temp_video_path,
                    cv2.VideoWriter_fourcc(*'XVID'),  # Use 'XVID' or 'H264'
                    frame_rate,
                    (width, height)
                )

            # Perform calibration and set squat variables only for the first frame
            if not first_frame_processed:
                calibration(pose_tracker, frame)
                set_squat_variables()
                first_frame_processed = True

            # Process the frame
            output_frame, overlays = process_image(pose_tracker, frame, PERSPECTIVE)
            output_frame = add_overlay(output_frame, overlays)

            # Write the processed frame to the video
            video_writer.write(output_frame)

        # Release the video writer
        if video_writer:
            video_writer.release()

        # Generate video with audio
        generate_video_with_audio(temp_video_path, video_audio_path, audio_timeline)
        print(f"Video with audio saved at: {video_audio_path}")

        # Delete the temporary video file without audio
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    if mode == "webcam":
        logging.info("Starting webcam pose estimation.")
        logging.info("Press 'q' to exit.")
        print("Starting webcam pose estimation.")
        print("Input: ", int(input))

        cap = cv2.VideoCapture(0)
        start_init(audio_feedback=useAudioFeedback)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if state == State.INIT:
                output = init_standing_pose(pose_tracker, frame)
                # Display the frame without annotated poses
                fullscreen_display(output, "Squat-Pose-Monitoring", flipImage=True)

            elif state == State.CALIBRATION:
                output = calibration(pose_tracker, frame)
                set_squat_variables()
                fullscreen_display(output, "Squat-Pose-Monitoring", flipImage=True)

            elif state == State.EXERCISING:
                # Process the image with pose estimation
                output, overlays = process_image(pose_tracker, frame, PERSPECTIVE)

                # Display the frame with annotated poses
                fullscreen_display(output, "Squat-Pose-Monitoring", overlays, flipImage=True)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Reset on pressing 'r'
            if cv2.waitKey(1) & 0xFF == ord('r'):
                start_init(audio_feedback=useAudioFeedback)


def has_movement(current_keypoints, movement_threshold=10):
    """Check if there is movement by comparing keypoints with the previous frame."""
    global previous_keypoints

    if previous_keypoints is None:
        previous_keypoints = current_keypoints
        return False  # No previous frame to compare

    # Compute the Euclidean distance between keypoints
    diff = np.linalg.norm(current_keypoints - previous_keypoints, axis=1)

    # Check if movement exceeds threshold for any keypoint
    if np.any(diff > movement_threshold):
        previous_keypoints = current_keypoints  # Update keypoints
        return True  # Movement detected

    return False  # No significant movement


def init_standing_pose(pose_tracker, frame):
    global state
    global time_init
    global previous_keypoints  # Ensure global access

    # Perform pose estimation on the frame
    keypoints, scores = pose_tracker(frame)

    # Draw skeleton on the frame
    img_show = frame.copy()
    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

    # Ensure body is visible (26 keypoints detected)
    if len(keypoints[0]) == 26:

        time_counter = (timedelta(seconds=time.perf_counter() - time_init)).total_seconds()

        # specify keypoints for checking movement 
        check_keypoints_ids = ['Neck', 'RShoulder', 'RHip', 'RKnee', 'RAnkle']
        check_keypoints = [get_kp(keypoints[0], kp) for kp in check_keypoints_ids]
        movement = has_movement(np.array(check_keypoints), movement_threshold=10)  # Convert to NumPy array for processing

        # Reset time_init if movement is detected
        if movement:
            print("Movement detected! Resetting timer.")
            time_init = time.perf_counter()  # Restart the timer

        # If no movement for `init_time_threshold` seconds, change state
        if time_counter > init_time_threshold:

            # now check the angle between lefttoe-leftankle and righttoe-rightankle
            left_toe = get_kp(keypoints[0], 'LBigToe')
            left_ankle = get_kp(keypoints[0], 'LAnkle')
            right_toe = get_kp(keypoints[0], 'RBigToe')
            right_ankle = get_kp(keypoints[0], 'RAnkle')
            ankle_angle = calculate_clockwise_angle(left_ankle, left_toe, right_ankle, right_toe)
            ankle_angle = min(abs(ankle_angle), 360 - abs(ankle_angle))

            feet_angle_threshold = 10
            print("Feet angle: ", ankle_angle)
            if abs(ankle_angle) < feet_angle_threshold:

                # now check hipwidth stance
                left_hip = get_kp(keypoints[0], 'LHip')
                right_hip = get_kp(keypoints[0], 'RHip')
                hip_distance = abs(left_hip[0] - right_hip[0])
                ankle_distance = abs(left_ankle[0] - right_ankle[0])
                # check if the distances are proportionally approximately equal
                hip_to_ankle_ratio = ankle_distance / hip_distance

                # Define a tolerance range for the ratio to consider it approximately equal
                tolerance = 0.3  # 30% tolerance
                print("Check: ", (1 - tolerance <= hip_to_ankle_ratio <= 1 + tolerance))

                if 1 - tolerance <= hip_to_ankle_ratio <= 1 + tolerance:
                    print("Hip-width stance detected!")
                    print("Pose stabilized! Changing state to CALIBRATION.")
                    state = State.CALIBRATION
                else:
                    print("Adjust your stance to be hip-width.")

    return img_show


def calibration(pose_tracker, frame):

    print("Calibrating...")

    # set global variables for formatting/drawing feedback
    global font_scale
    global thickness
    global start_x
    global start_y
    global line_spacing
    global skeleton_kp_radius
    global error_knee_threshold

    frame_width = frame.shape[1]
    font_scale = 0.001 * frame_width
    thickness = max(1, int(0.005 * frame_width))
    start_x = int(0.05 * frame_width)
    start_y = int(0.0625 * frame_width)
    line_spacing = int(0.05 * frame_width)



    # set global variables for pose estimation
    global BODY_HEIGHT
    global FLOOR_HEIGHT
    global REL_ANKLE_HEIGHT
    global state
    global PERSPECTIVE

    # exclude side of the image for pose estimation
    temp_frame = frame.copy()
    temp_frame[:, :int(frame.shape[1] * 0.25)] = np.ones((temp_frame.shape[0], int(temp_frame.shape[1] * 0.25), 3), dtype=np.uint8) * 255

    # Perform pose estimation on the frame
    keypoints, scores = pose_tracker(temp_frame)

    if len(keypoints[0]) == 26:

        # get the perspective
        head = get_kp(keypoints[0], 'Head')
        nose = get_kp(keypoints[0], 'Nose')
        right_toe = get_kp(keypoints[0], 'RSmallToe')
        left_toe  = get_kp(keypoints[0], 'LSmallToe')
        toes_height = (right_toe[1] + left_toe[1]) / 2
        BODY_HEIGHT = abs(head[1] - toes_height)
        right_hip = get_kp(keypoints[0], 'RHip')
        left_hip = get_kp(keypoints[0], 'LHip')

        # get the difference between the right and left hip keypoints, and scale with body height
        rel_hip_distance = abs((right_hip[0] - left_hip[0]) / BODY_HEIGHT)
        print("Body height:", BODY_HEIGHT)
        print("Hip distance:", abs(right_hip[0] - left_hip[0]))
        print("Relative hip distance:", rel_hip_distance)

        # with large hip distance, the perspective is frontal. Otherwise, it is lateral, then check the side
        if abs(rel_hip_distance) > 0.1:
            PERSPECTIVE = "frontal"
        else:
            PERSPECTIVE = "lateral_right" if nose[0] > head[0] else "lateral_left"
        print("Perspective:", PERSPECTIVE)
    
        # get image coordinates
        BODY_SIDE = 'R' if PERSPECTIVE == "lateral_right" else 'L'
        small_toe = get_kp(keypoints[0], BODY_SIDE + 'SmallToe')
        ankle     = get_kp(keypoints[0], BODY_SIDE + 'Ankle')

        # set global variables
        FLOOR_HEIGHT = small_toe[1]
        BODY_HEIGHT = abs(head[1] - FLOOR_HEIGHT) # image body height is distance between head and floor
        ankle_height = abs(ankle[1] - FLOOR_HEIGHT) # image ankle height is distance between ankle and floor
        REL_ANKLE_HEIGHT = (ankle_height / BODY_HEIGHT) * 100 # ankle height relative to body height

        # skeleton keypoint radius depends on body height
        skeleton_kp_radius = int(0.015 * BODY_HEIGHT)
        error_knee_threshold = err_knee_thrs_factor * BODY_HEIGHT

        state = State.EXERCISING
        if useAudioFeedback:
            play_audio("Start your workout.")
    
    else: 
        print("Body not visible.")


    return frame


def start_init(audio_feedback=False, viz_feedback=False):
    global time_init
    global previous_keypoints
    global state

    # reset variables
    previous_keypoints = None
    time_init = time.perf_counter()
    state = State.INIT

    if audio_feedback:
        play_audio("Initialize standing pose. Stand still for 3 seconds.")


def play_audio(audio, timestamp=None):

    # if audio is a filename .mp3
    if audio.endswith(".mp3"):
        audio_path = os.path.join(AUDIO_DIR, audio)

    # if audio is a text string
    else:
        """Play or generate an audio file from text."""
        # Generate a filename based on the text
        filename = f"{audio.replace(' ', '_')}.mp3"
        audio_path = os.path.join(AUDIO_DIR, filename)

        # Check if the audio file already exists
        if not os.path.exists(audio_path):
            print(f"Generating new audio: {audio}")
            tts = gTTS(text=audio, lang='en')
            tts.save(audio_path)

    # Play the audio file using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    # save to audio timeline
    if timestamp is not None:
        global audio_timeline
        audio_timeline.append((audio_path, timestamp))




def reset_global_variables():
    """
    Resets global variables to their initial state.
    """
    global timestamp, audio_timeline, PERSPECTIVE
    global previous_keypoints, REL_ANKLE_HEIGHT, BODY_HEIGHT, FLOOR_HEIGHT

    # Reset variables
    timestamp = 0
    audio_timeline = []
    PERSPECTIVE = None
    previous_keypoints = None
    REL_ANKLE_HEIGHT = None
    BODY_HEIGHT = None
    FLOOR_HEIGHT = None


def create_text_overlay(text, position, width, height, font_scale=1, thickness=2, color=(0, 0, 255)):

    # Create a transparent image
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    text_x = position[0]
    text_y = position[1]

    # Add text to the overlay
    cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return overlay


if __name__ == "__main__":

    # # use webcam
    # workout_monitoring(mode="webcam", input="0")

    # Define the base directory containing the data
    base_dir = "./data"

    # Print the full path of the base directory
    base_dir = os.path.abspath(base_dir)
    print(f"Base directory: {base_dir}")

    # Iterate through the participant directories (e.g., A-squat, B-squat)
    for participant_dir in os.listdir(base_dir):
        participant_path = os.path.join(base_dir, participant_dir)

        # Check if it's a directory
        if os.path.isdir(participant_path):
            print(f"Processing participant directory: {participant_path}")

            # Look for .mp4 files directly in the participant directory
            for file in os.listdir(participant_path):
                if file.lower().endswith(".mp4"):
                    input_video = os.path.join(participant_path, file)

                    # Define the output path for the processed video
                    output_video = input_video.replace(".mp4", "_SQUAT_MONITOR.mp4")

                    # Call workout_monitoring for each video
                    print(f"Processing video: {input_video}")
                    workout_monitoring(mode="video", input=input_video, output=output_video)
                    print(f"Processed video saved at: {output_video}")