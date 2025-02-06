import os
import cv2
from natsort import natsorted
import numpy as np

# enumerate frames in a directory (optionally from start_frame to end_frame)
def frame_enumerator(dir_path, start_frame=None, end_frame=None):
    frame_files = natsorted(
        [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )
    for i, frame_file in enumerate(frame_files):
        if start_frame is not None and i < start_frame:
            continue
        if end_frame is not None and i > end_frame:
            break
        frame_path = os.path.join(dir_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            yield i, frame, frame_file



def save_cropped_frames(input_source, output_dir, start_frame=None, end_frame=None):

    os.makedirs(output_dir, exist_ok=True)

    # Iterate over frames
    for i, frame, frame_file in frame_enumerator(input_source, start_frame=start_frame, end_frame=end_frame):

        # Crop the frame
        cropped_frame = frame[:, 300:700]
        output_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_path, cropped_frame)
        print(f"Saved cropped frame {frame_file} to {output_path}")



def blur(img, x, y, radius):
    blurred_img = cv2.GaussianBlur(img, (29, 29), 0)
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.circle(mask, (int(x), int(y)), int(radius), (255, 255, 255), -1)


    out = np.where(mask==(255, 255, 255), blurred_img, img)
    return out

# cwd = os.getcwd()
# data_dir = os.path.join(cwd, "src/data")
# participants = ["A", "B", "C", "D", "E", "I", "J", "K", "L", "M", "N", "O"]

# for participant in participants:
#     participant_dir = participant + "-squat"
#     input_path = os.path.join(data_dir, "videos", participant_dir, "videos/244622073485_right")
#     output_path = os.path.join(data_dir, "videos_cropped", participant_dir)
#     os.makedirs(output_path, exist_ok=True)
#     save_cropped_frames(input_path, output_path)
