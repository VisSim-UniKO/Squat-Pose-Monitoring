import cv2
import numpy as np
import os
import logging
from rtm_pose_utils import initialize_pose_tracker, get_kp, init_squats
from image_utils import frame_enumerator, draw_skeleton, draw_arrow, images_to_video
import pandas as pd
import sys
import cv2

from realsense_utils import create_pipelines

BODY_SIDE='R'
FPS = 30



def process_image(pose_tracker, frame):

    # # make image higher resolution
    # height, width, _ = frame.shape
    # frame = cv2.resize(frame, (width * 2, height * 2))  # Scale the image up if it's low-res


    # Perform pose estimation on the frame
    keypoints, scores = pose_tracker(frame)

    # Copy the original frame to avoid modifying it directly
    img_show = frame.copy()

    # # # Add extra white space at the bottom of the image
    # extra_height = 80 
    # img_show = np.vstack([img_show, np.ones((extra_height, frame.shape[1], 3), dtype=np.uint8) * 255])

    # Draw skeleton on the frame
    side = 'right' if BODY_SIDE == 'R' else 'left'
    selected_keypoints = [side + '_hip', side + '_knee', side + '_ankle', side + '_big_toe']

    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1, radius=4,
                            selected_kpts=selected_keypoints)

    # if body is visible (26 keypoints visible)
    if len(keypoints[0]) == 26:

        ################# CHECK SQUAT CONDITIONS #######################

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
        error_femur = round(((max(0, hip[1] - knee[1])) / BODY_HEIGHT ) * 100, 2)
        condition_femur = hip[1] < knee[1]

        # check condition 3: knee joint does not exceed the vertical extension of the toe key point during flexion
        error_knee = round(((max(0, knee[0] - toe[0])) / BODY_HEIGHT ) * 100, 2)
        condition_knee = knee[0] < toe[0]

        # Define conditions and corresponding labels
        conditions = {
            "Heel": condition_heel,
            "Femur": condition_femur,
            "Knee": condition_knee
        }

        # Define errors and corresponding labels
        errors = {
            "Heel": error_heel,
            "Femur": error_femur,
            "Knee": error_knee
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

        # Smaller font size for the text
        font_scale = 0.65  # Smaller text size
        thickness = 2
        start_x, start_y = 20, frame.shape[0] + 20  # Start drawing text in the white space
        line_spacing = 20


        # Draw error labels in the white space at the bottom with color based on condition status
        for i, (label, value) in enumerate(errors.items()):
            # # Get the corresponding condition to determine the color
            # condition_color = (0, 255, 0) if conditions[label] else (0, 0, 255)
            # label_text = f"{label} error: {value:.2f}"
            # cv2.putText(img_show, label_text, (start_x, start_y + i * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_scale, condition_color, thickness)

            if not conditions[label]:
                # draw red circle at condition keypoint and a small arrow pointing upwards
                kp = condition_markers[label][0]
                tip = condition_markers[label][1]
                cv2.circle(img_show, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)
                cv2.arrowedLine(img_show, (int(kp[0]), int(kp[1])), (int(tip[0]), int(tip[1])), (0, 0, 255), 2, tipLength=0.3)
                # draw_arrow(img_show, (int(kp[0]), int(kp[1])), (int(tip[0]), int(tip[1])), color=(0, 0, 255), thickness=2, tip_size=15)

                # Add feedback text
                # # resize to higher res
                # height, width, _ = img_show.shape
                # img_show = cv2.resize(img_show, (width * 2, height * 2))  # Scale the image up if it's low-res
                feedback_text = conditions_feedback[label]
                # cv2.putText(img_show, feedback_text, (start_x, start_y + i * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                cv2.putText(img_show, feedback_text, (20, 25 + i * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)



    # # Display the frame with annotated poses
    # cv2.imshow("Webcam Pose Estimation", img_show)

    return img_show, [condition_heel, condition_femur, condition_knee], [error_heel, error_femur, error_knee]

def process_image_live(pose_tracker, frame):

    # flip frame vertically
    frame = cv2.flip(frame, 1)

    # Perform pose estimation on the frame
    keypoints, scores = pose_tracker(frame)

    # Copy the original frame to avoid modifying it directly
    img_show = frame.copy()

    # Add extra white space at the bottom of the image
    # Draw skeleton on the frame
    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1)

    # if body is visible (26 keypoints visible)
    if len(keypoints[0]) == 26:

        ################# CHECK SQUAT CONDITIONS #######################

        hip   = get_kp(keypoints[0], BODY_SIDE + "Hip")
        knee  = get_kp(keypoints[0], BODY_SIDE + "Knee")
        ankle = get_kp(keypoints[0], BODY_SIDE + "Ankle")
        toe   = get_kp(keypoints[0], BODY_SIDE + "BigToe")

        # # check condition 1: Heels remain flat on the ground
        # ankle_height = abs(ankle[1] - FLOOR_HEIGHT)
        # frame_rel_ankle_height = (ankle_height / BODY_HEIGHT) * 100

        # error_heel = round(max(0,(frame_rel_ankle_height - REL_ANKLE_HEIGHT)),2)
        # condition_heel = error_heel < 1

        # check condition 2: Femur not below knee
        # error_femur = round(((max(0, hip[1] - knee[1])) / BODY_HEIGHT ) * 100, 2)
        condition_femur = hip[1] < knee[1]

        # check condition 3: knee joint does not exceed the vertical extension of the toe key point during flexion
        # error_knee = round(((max(0, knee[0] - toe[0])) / BODY_HEIGHT ) * 100, 2)
        condition_knee = knee[0] < toe[0]

        # Define conditions and corresponding labels
        conditions = {
            # "Heel": condition_heel,
            "Femur": condition_femur,
            "Knee": condition_knee
        }

        # # Define errors and corresponding labels
        # errors = {
        #     "Heel": error_heel,
        #     "Femur": error_femur,
        #     "Knee": error_knee
        # }

        # # Smaller font size for the text
        font_scale = 0.5  # Smaller text size
        thickness = 2
        start_x, start_y = frame.shape[1]-100, 50  # Start drawing text in the white space
        line_spacing = 20


        # Draw error labels in the white space at the bottom with color based on condition status
        for i, (label, value) in enumerate(conditions.items()):
            # Get the corresponding condition to determine the color
            condition_color = (0, 255, 0) if conditions[label] else (0, 0, 255)
            label_text = f"{label}"
            cv2.putText(img_show, label_text, (start_x, start_y + i * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, font_scale, condition_color, thickness)
        
        # # put text in middle of the frame
        # cv2.putText(img_show, "Squat", (int(frame.shape[1]/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # # put text in upper right corner
        # cv2.putText(img_show, "Heel", (frame.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # # Display the frame with annotated poses
    # cv2.imshow("Webcam Pose Estimation", img_show)

    #return img_show, [condition_heel, condition_femur, condition_knee], [error_heel, error_femur, error_knee]
    return img_show


def workout_monitor_realsense(num_cams):
    """
    Starts the RealSense camera pipelines and displays the video streams.
    Arguments:
    - num_cams: Number of cameras to use for streaming.
    """

    pose_tracker = initialize_pose_tracker(pose_model='HALPE_26', mode='balanced', det_frequency=1, tracking=False)
    
    # Create and configure pipelines for the cameras
    pipelines, camera_serial_numbers = create_pipelines(num_cams)

    # Create window to display images
    cv2.namedWindow("Color Images", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Capture frames from all cameras
            color_images = []
            for pipeline in pipelines:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                # Convert to numpy array for OpenCV
                color_image = np.asanyarray(color_frame.get_data())

                # process image
                color_image = process_image(pose_tracker, color_image)
                color_images.append(color_image)

            if len(color_images) < len(pipelines):
                continue

            # Combine all color images horizontally
            combined_image = np.vstack(color_images)

            # Show combined color images
            cv2.imshow("Color Images", combined_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # Press 'q' or Esc to exit
                break

    finally:
        # Stop streaming for all cameras
        for pipeline in pipelines:
            pipeline.stop()

        # Close OpenCV windows
        cv2.destroyAllWindows()



def workout_monitoring(mode=None, input=None, output_path=None):

    '''
    Capture video from the webcam or process recorded images from a directory, estimate poses using RTM, and display the annotated poses live.
    '''
    
    pose_tracker = initialize_pose_tracker(pose_model='HALPE_26', mode='balanced', det_frequency=1, tracking=False)

    if mode == "realsense":
        logging.info("Starting RealSense pose estimation.")
        logging.info("Press 'q' to exit.")

        # start realsense stream
        print("Starting RealSense pose estimation.")
        workout_monitor_realsense(int(input))


    if mode == "webcam":

        logging.info("Starting webcam pose estimation.")
        logging.info("Press 'q' to exit.")
        print("Starting webcam pose estimation.")

        print("Input: ", int(input))


        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            cv2.imshow("Webcam Pose Estimation", frame)

            # # Process the image with pose estimation
            # output = process_image_live(pose_tracker, frame)

            # # Display the frame with annotated poses
            # cv2.imshow("Webcam Pose Estimation", output)

            # # Introduce a delay if needed
            # time.sleep(1 / FPS)

            # Exit on pressing 'q'
            cv2.waitkey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    
    if mode == "videos" and os.path.isdir(input):
        logging.info("Starting pose estimation for all images in the directory.")

        # get floor height and body height
        global REL_ANKLE_HEIGHT
        global BODY_HEIGHT
        global FLOOR_HEIGHT
        FLOOR_HEIGHT , ANKLE_HEIGHT, BODY_HEIGHT, squat_segments = init_squats(input_path, pose_tracker, BODY_SIDE)
        REL_ANKLE_HEIGHT = (ANKLE_HEIGHT / BODY_HEIGHT) * 100
        squat_data = []

        for i, [start_frame, end_frame] in enumerate(squat_segments):

            [CONDITION_HEEL, CONDITION_FEMUR, CONDITION_KNEE] = [True, True, True]
            [ERROR_HEEL, ERROR_FEMUR, ERROR_KNEE] = [0.0, 0.0, 0.0]
            
            # Map names to their values
            condition_dict = {
                "CONDITION_HEEL": CONDITION_HEEL,
                "CONDITION_FEMUR": CONDITION_FEMUR,
                "CONDITION_KNEE": CONDITION_KNEE
            }

            error_dict = {
                "ERROR_HEEL": ERROR_HEEL,
                "ERROR_FEMUR": ERROR_FEMUR,
                "ERROR_KNEE": ERROR_KNEE
            }

            max_error_frames = {
                "ERROR_HEEL": None,
                "ERROR_FEMUR": None,
                "ERROR_KNEE": None
            }
            
            # Process images
            for frame_num, frame, frame_file in frame_enumerator(input_path, start_frame, end_frame):

                # Process the image with pose estimation
                output, conditions, errors = process_image(pose_tracker, frame)

                # Update conditions
                condition_dict = {name: cond and condition_dict[name] for name, cond in zip(condition_dict.keys(), conditions)}

                # Update errors and track frames with the highest errors
                for name, error, condition in zip(error_dict.keys(), errors, condition_dict.keys()):
                    if error > error_dict[name]:  # Update error and frame file if a new max is found
                        error_dict[name] = error
                        max_error_frames[name] = frame_file

                # Save output to output directory
                output_file = os.path.join(output_path, frame_file)
                cv2.imwrite(output_file, output)

                # # Introduce a delay if needed
                # time.sleep(1 / FPS)

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Append squat data including frames with highest errors
            squat_data.append([[start_frame, end_frame], condition_dict, error_dict, max_error_frames])



        cv2.destroyAllWindows()
        logging.info("Pose estimation for all images in the directory ended.")

        return squat_data



def save_to_excel(excel_file, participant_data):
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        workbook = writer.book  # Get the workbook for formatting

        # Try to access the first sheet, or create a new one if it doesn't exist
        if 'Squat Evaluation' in writer.sheets:
            worksheet = writer.sheets['Squat Evaluation']
        else:
            worksheet = workbook.add_worksheet('Squat Evaluation')

        # Define header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        # Set column headers
        headers = [
            "PARTICIPANT", "SQUAT", "CONDITION_HEEL", "ERROR_HEEL", "MAX_ERROR_HEEL_FRAME",
            "CONDITION_FEMUR", "ERROR_FEMUR", "MAX_ERROR_FEMUR_FRAME", "CONDITION_KNEE",
            "ERROR_KNEE", "MAX_ERROR_KNEE_FRAME"
        ]
        for col_num, value in enumerate(headers):
            worksheet.write(0, col_num, value, header_format)

        # Starting row for data
        row = 1

        for participant, data in participant_data.items():
            # Write participant name in the first column
            worksheet.write(row, 0, participant)


            # Process data for this participant
            for index, squat in enumerate(data):
                # Combine Squat Number, Start Frame, and End Frame into a single cell
                formatted_data = [
                    f"Squat {index + 1} ({squat[0][0]}-{squat[0][1]})",
                    str(squat[1]["CONDITION_HEEL"]),  # Convert to string
                    str(squat[2]["ERROR_HEEL"]),  # Convert to string
                    str(squat[3]["ERROR_HEEL"]),  # Convert to string
                    str(squat[1]["CONDITION_FEMUR"]),  # Convert to string
                    str(squat[2]["ERROR_FEMUR"]),  # Convert to string
                    str(squat[3]["ERROR_FEMUR"]),  # Convert to string
                    str(squat[1]["CONDITION_KNEE"]),  # Convert to string
                    str(squat[2]["ERROR_KNEE"]),  # Convert to string
                    str(squat[3]["ERROR_KNEE"])  # Convert to string
                    ]


                # Write the formatted data to the subsequent columns
                for col_num, value in enumerate(formatted_data):
                    worksheet.write(row, col_num + 1, value)

                # Move to the next row for the next squat
                row += 1

            # Add a blank row after each participant's data
            row += 1



if __name__ == "__main__":

    # use webcam
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        input = sys.argv[2]

        if mode == "webcam" or mode == "realsense":
            workout_monitoring(mode=mode, input=input)
    
    # use recorded videos
    else:
        cwd = os.getcwd()
        data_dir = os.path.join(cwd, "src/data")
        excel_file = os.path.join(data_dir, "evaluation.xlsx")
        #participants = ["A", "D", "I"]
        participants = ["A", "B", "C", "D", "E", "I", "J", "K", "L", "M", "N", "O"]
        participant_data = {}


        for participant in participants:

            try:
                participant_dir = participant + "-squat"
                input_path = os.path.join(data_dir, "videos_lateral_cropped", participant_dir)
                output_path = os.path.join(data_dir, "output_feedback", participant_dir)
                os.makedirs(output_path, exist_ok=True)
                
                # Assuming you have a function `workout_monitoring` that processes the data
                squat_data = workout_monitoring("videos", input_path, output_path)
                
                # Print the data to ensure it's being processed
                print(f"Squat data for participant {participant}: {squat_data}")

                # Ensure the data is being correctly added to participant_data
                participant_data[participant] = squat_data
                print(f"Added data for participant {participant}")

                output_video_path = os.path.join(output_path, "output.mp4")
                # convert png to mp4
                images_to_video(output_path, output_video_path, fps=30)
            
            except Exception as e:
                print(f"Error processing participant {participant}: {e}")
                participant_data[participant] = []
                continue

        # Now save the data to the Excel file
        save_to_excel(excel_file, participant_data)
