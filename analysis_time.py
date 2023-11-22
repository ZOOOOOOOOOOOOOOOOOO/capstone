import cv2
import mediapipe as mp
from landmark import get_landmark

Time = {'address': 0, 'back': 0, 'back_top': 0, 'impact': 0, 'finish': 0}

def video_time(video_path, landmarks_dict, image_width, image_height):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
        # 첫 프레임
        is_first = True
        first_right_ankle_x = 0
        first_left_ankle_x = 0
        first_right_shoulder_y = 0

        if is_first:
            first_right_ankle_x = landmarks_dict["right_ankle"][0] * image_width
            first_left_ankle_x = landmarks_dict["left_ankle"][0] * image_width
            first_right_shoulder_y = landmarks_dict["right_shoulder"][1] * image_height
            is_first = False

        first_ankle_center_x = int((first_left_ankle_x + first_right_ankle_x) / 2)

        prev_wrist_y = None  # 이전 프레임의 y좌표
        frame_cnt = 0  # 프레임을 세는 기준

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # landmark
            landmarks_dict = get_landmark(mp_pose, results.pose_landmarks.landmark)

            # 현재 시각
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # 현재 손목의 좌표를 계속해서 추적
            current_wrist_y = landmarks_dict["right_wrist"][1]
            current_pinky_y = landmarks_dict["pinky"][1]

            # 어드레스
            if first_ankle_center_x == current_wrist_y:
                Time['address'] = current_time
            # 백스윙
            if first_right_shoulder_y == current_pinky_y:
                Time['back'] = current_time
            # 백스윙_탑
            if prev_wrist_y is not None and current_wrist_y > prev_wrist_y:
                Time['back_top'] = current_time
            # 임팩트
            if first_ankle_center_x == current_wrist_y: 
                Time['impact'] = current_time
            # 피니시

    cap.release()
    cv2.destroyAllWindows()

    return Time
