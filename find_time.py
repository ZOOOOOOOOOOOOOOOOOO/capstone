import cv2
import mediapipe as mp
from landmark import get_landmark

def find_back_top_time(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
        prev_wrist_y = None
        frame_number = 0
        start_time = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks_dict = get_landmark(mp_pose, results.pose_landmarks.landmark)

            current_left_wrist_y = landmarks_dict["left_wrist"][1] * frame_size[1]

            if prev_wrist_y is not None and current_left_wrist_y > prev_wrist_y:
                start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                break

            prev_wrist_y = current_left_wrist_y
            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

        return start_time
