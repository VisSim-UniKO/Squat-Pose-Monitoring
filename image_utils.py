import os
import cv2
from natsort import natsorted
import numpy as np
import math
from halpe26 import halpe26
import glob
import pyautogui
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip, AudioClip
import logging

# enumerate frames in a directory or video file (optionally from start_frame to end_frame)
def frame_enumerator(input_source, start_frame=None, end_frame=None):
    """
    Enumerates frames from a directory of images or a video file.

    Args:
        input_source (str): Path to the directory containing images or a video file.
        start_frame (int, optional): The frame index to start processing (inclusive).
        end_frame (int, optional): The frame index to stop processing (inclusive).

    Yields:
        tuple: (frame_index, frame, frame_name) where frame_index is the index of the frame,
               frame is the image as a numpy array, and frame_name is the name of the frame file
               or "frame_<index>" for video frames.
    """
    # Check if the input source is a directory
    if os.path.isdir(input_source):
        frame_files = natsorted(
            [f for f in os.listdir(input_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        for i, frame_file in enumerate(frame_files):
            if start_frame is not None and i < start_frame:
                continue
            if end_frame is not None and i > end_frame:
                break
            frame_path = os.path.join(input_source, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                yield i, frame, frame_file

    # Check if the input source is a video file
    elif input_source.lower().endswith(('.mp4', '.MP4')):
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_source}.")
            return

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame is not None and frame_index < start_frame:
                frame_index += 1
                continue
            if end_frame is not None and frame_index > end_frame:
                break
            yield frame_index, frame, f"frame_{frame_index:04d}.png"
            frame_index += 1

        cap.release()

    else:
        print(f"Error: Invalid input source {input_source}. Must be a directory or a video file.")


def mp4_to_frames(input_video, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read and save frames from the video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Construct the full output path
        output_path = os.path.join(output_dir, f"{frame_count:04d}.png")

        # Save the frame
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_count:04d} to {output_path}")

        frame_count += 1

    cap.release()
    print(f"Total frames saved: {frame_count}")



def save_cropped_frames(input_source, output_dir, start_frame=None, end_frame=None):
    """
    Loops through frames from input_source, crops each frame, and saves them to the output directory.
    Args:
        input_source (str): Path to the directory containing the input frames.
        output_dir (str): Path to the directory where cropped frames will be saved.
        start_frame (int, optional): The frame index to start processing (inclusive).
        end_frame (int, optional): The frame index to stop processing (inclusive).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over frames from the input source using the frame_enumerator
    for i, frame, frame_file in frame_enumerator(input_source, start_frame=start_frame, end_frame=end_frame):

        # get the image dimensions
        height, width, _ = frame.shape
        # crop width 15% each side
        crop_width = int(width * 0.2)
        cropped_frame = frame[:, crop_width:width-crop_width]
        # Crop the frame (from column 300 to 700)
        # cropped_frame = frame[:, 300:700]

        # # blur image 20% at right and left (strong blur)
        # blur_width = int(width * 0.2)
        # blur_factor = 51
        # cropped_frame[:, :blur_width] = cv2.GaussianBlur(cropped_frame[:, :blur_width], (blur_factor, blur_factor), 0)
        # cropped_frame[:, -blur_width:] = cv2.GaussianBlur(cropped_frame[:, -blur_width:], (blur_factor, blur_factor), 0)


        # Construct the full output path
        output_path = os.path.join(output_dir, frame_file)

        # Save the cropped frame
        cv2.imwrite(output_path, cropped_frame)
        print(f"Saved cropped frame {frame_file} to {output_path}")





def draw_skeleton(img,
                  keypoints,
                  scores,
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2,
                  selected_kpts=None):
    """
    Draws a pose skeleton on an image with options to select specific keypoints.
    Automatically filters skeleton links based on selected keypoints.

    Parameters:
        img (numpy.ndarray): The image on which to draw.
        keypoints (numpy.ndarray): The array of keypoints (shape: [1, N, 2]).
        scores (numpy.ndarray): The confidence scores for keypoints (shape: [1, N]).
        kpt_thr (float): Threshold for keypoint visibility.
        radius (int): Radius of keypoint circles.
        line_width (int): Thickness of skeleton lines.
        selected_kpts (list): List of keypoint names to draw. If None, draw all.

    Returns:
        numpy.ndarray: Image with drawn skeleton.
    """
    keypoint_info = halpe26['keypoint_info']
    skeleton_info = halpe26['skeleton_info']
    keypoints = keypoints[0]
    scores = scores[0]

    assert len(keypoints.shape) == 2
    vis_kpt = [s >= kpt_thr for s in scores]

    # Create a mapping from keypoint names to indices
    kpt_name_to_id = {info['name']: i for i, info in keypoint_info.items()}

    # If selected_kpts is specified, filter out unavailable keypoints
    selected_kpt_ids = set(kpt_name_to_id[kpt] for kpt in selected_kpts) if selected_kpts else set(kpt_name_to_id.values())

    # Auto-select links based on chosen keypoints
    for i, ske_info in skeleton_info.items():
        kpt1_name, kpt2_name = ske_info['link']
        kpt1_id, kpt2_id = kpt_name_to_id[kpt1_name], kpt_name_to_id[kpt2_name]

        if kpt1_id in selected_kpt_ids and kpt2_id in selected_kpt_ids and vis_kpt[kpt1_id] and vis_kpt[kpt2_id]:
            link_color = ske_info['color']
            kpt1 = keypoints[kpt1_id]
            kpt2 = keypoints[kpt2_id]

            img = cv2.line(img, (int(kpt1[0]), int(kpt1[1])),
                           (int(kpt2[0]), int(kpt2[1])),
                           [255,0,0],
                           thickness=line_width)
            
    # Draw keypoints (over links)
    for i, kpt_info in keypoint_info.items():
        kpt_name = kpt_info['name']
        kpt_color = tuple(kpt_info['color'])

        if i in selected_kpt_ids and vis_kpt[i]:
            kpt = keypoints[i]
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), [0,255,0], -1)


    return img




#####################################



def draw_arrow(img, start, end, color, thickness=2, tip_size=10, tip_angle=30):
    # Convert to NumPy arrays
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)
    
    # Compute direction vector
    direction = end - start
    length = np.linalg.norm(direction)

    if length == 0:
        return  # Avoid division by zero

    # Normalize direction
    unit_direction = direction / length

    # Compute new end position (shortened by tip size)
    new_end = end - unit_direction * tip_size

    # Draw the main line
    cv2.line(img, tuple(start.astype(int)), tuple(new_end.astype(int)), color, thickness)

    # Rotate the unit direction by ±tip_angle to get the arrowhead sides
    angle_rad = np.radians(tip_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Compute the two arrowhead lines
    left_tip = new_end + tip_size * np.array([unit_direction[0] * cos_a - unit_direction[1] * sin_a, 
                                              unit_direction[0] * sin_a + unit_direction[1] * cos_a])
    
    right_tip = new_end + tip_size * np.array([unit_direction[0] * cos_a + unit_direction[1] * sin_a, 
                                               -unit_direction[0] * sin_a + unit_direction[1] * cos_a])

    # Draw the arrowhead
    cv2.line(img, tuple(new_end.astype(int)), tuple(left_tip.astype(int)), color, thickness)
    cv2.line(img, tuple(new_end.astype(int)), tuple(right_tip.astype(int)), color, thickness)



def images_to_video(input_dir, output_video="output.mp4", fps=30):
    # Get all PNG and JPG images in the directory
    image_files = natsorted(glob.glob(os.path.join(input_dir, "*.png")) + 
                            glob.glob(os.path.join(input_dir, "*.jpg")))

    # Read the first image to get the original frame width and height
    first_image = cv2.imread(image_files[0])
    original_height, original_width, _ = first_image.shape
    print(f"Original video resolution: {original_width}x{original_height}, FPS: {fps}")

    # Double the width and height
    new_width = original_width * 2
    new_height = original_height * 2
    print(f"Upscaled video resolution: {new_width}x{new_height}")

    # Use the obtained new resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

    if not out.isOpened():
        print("Error: Failed to initialize VideoWriter.")
        return

    # Write images to video with upscaled frames
    for img_file in image_files:
        frame = cv2.imread(img_file)

        # Resize frame to double the original dimensions
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        out.write(frame)

    out.release()
    print(f"Video saved as {output_video}")


def frames_to_mp4(input_dir, output_video, fps=30):
    """
    Converts a directory of frames (images) into an MP4 video.

    Args:
        input_dir (str): Path to the directory containing the frames.
        output_video (str): Path to the output MP4 video file.
        fps (int): Frames per second for the output video.

    Returns:
        None
    """
    # Get all image files in the directory, sorted naturally
    image_files = natsorted(
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    if not image_files:
        print(f"Error: No image files found in directory {input_dir}.")
        return

    # Read the first image to get the frame dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Could not read the first image {image_files[0]}.")
        return

    height, width, _ = first_image.shape
    print(f"Video resolution: {width}x{height}, FPS: {fps}")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Failed to initialize VideoWriter for {output_video}.")
        return

    # Write each frame to the video
    for img_file in image_files:
        frame = cv2.imread(img_file)
        if frame is None:
            print(f"Warning: Could not read image {img_file}. Skipping.")
            continue

        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video}")


def generate_video_with_audio(video_path, final_output_path, audio_timeline):
    # Suppress moviepy logs
    logging.getLogger("moviepy").setLevel(logging.ERROR)

    # Step 1: Create audio clips with proper start times
    audio_clips = []
    for audio_file, start_time in audio_timeline:
        actual_audio_clip = AudioFileClip(audio_file).with_start(start_time)
        audio_clips.append(actual_audio_clip)

    if not audio_clips:
        print("No audio clips found. Saving video without audio.")
        video = VideoFileClip(video_path)
        video.write_videofile(final_output_path, codec="libx264")
        return

    final_audio = CompositeAudioClip(audio_clips)
    video = VideoFileClip(video_path)
    final_video = video.with_audio(final_audio)
    final_video.write_videofile(
        final_output_path,
        codec="libx264",
        audio_codec="aac",
        bitrate="5000k"
    )
    video.close()
    final_video.close()

    temp_video_path = video_path  # Assuming video_path is the temporary file path
    try:
        os.remove(temp_video_path)
    except PermissionError as e:
        print(f"Could not delete temporary file: {e}")


def fullscreen_display(frame, window_name="Window", overlays=None, flipImage=False):
    """
    Displays a frame in fullscreen mode with optional image overlays.

    Args:
        frame (numpy.ndarray): The main image to display.
        window_name (str): The name of the display window.
        overlays (list of tuples): Each tuple contains:
            - image_path (str): Path to the overlay image.
            - scale (float): Scaling factor for the overlay image (percentage of the base image width, e.g., 0.2 for 20%).
            - position (tuple): (float, float): (x, y) position as percentages of the base image dimensions, defining the center of the overlay.
    """
    screen_w, screen_h = pyautogui.size()
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    y_offset, x_offset = (screen_h - resized.shape[0]) // 2, (screen_w - resized.shape[1]) // 2
    canvas[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    # Add overlays if provided
    if overlays:
        canvas = add_overlay(canvas, overlays)

    if flipImage:
        canvas = cv2.flip(canvas, 1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)


def add_overlay(frame, overlays):
    """
    Adds overlays to the given frame.

    Args:
        frame (numpy.ndarray): The base image to which overlays will be added.
        overlays (list of tuples): Each tuple contains:
            - overlay_path (str): Path to the overlay image.
            - scale (float): Scaling factor for the overlay image (percentage of the base image width, e.g., 0.2 for 20%).
            - position (tuple): (float, float): (x, y) position as percentages of the base image dimensions, defining the center of the overlay.

    Returns:
        numpy.ndarray: The frame with overlays added.
    """
    for overlay_path, overlay_scale, overlay_position in overlays:
        # Load the overlay image from the path
        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            print(f"Warning: Could not load overlay image from {overlay_path}")
            continue

        # Resize the overlay based on the frame width
        overlay_h, overlay_w = overlay_img.shape[:2]
        target_width = int(overlay_scale * frame.shape[1])
        target_height = int((target_width / overlay_w) * overlay_h)
        overlay_resized = cv2.resize(overlay_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Calculate the position
        center_x = int(overlay_position[0] * frame.shape[1])
        center_y = int(overlay_position[1] * frame.shape[0])
        pos_x = center_x - target_width // 2
        pos_y = center_y - target_height // 2

        # Add the overlay to the frame
        for y in range(max(0, pos_y), min(pos_y + target_height, frame.shape[0])):
            for x in range(max(0, pos_x), min(pos_x + target_width, frame.shape[1])):
                if overlay_resized.shape[2] == 4:  # If overlay has an alpha channel
                    alpha = overlay_resized[y - pos_y, x - pos_x, 3] / 255.0
                    for c in range(3):  # Blend each color channel
                        frame[y, x, c] = (1 - alpha) * frame[y, x, c] + alpha * overlay_resized[y - pos_y, x - pos_x, c]
                else:  # No alpha channel, direct overlay
                    frame[y, x] = overlay_resized[y - pos_y, x - pos_x]

    return frame