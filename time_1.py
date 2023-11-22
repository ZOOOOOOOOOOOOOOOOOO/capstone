import cv2
import mediapipe as mp
import landmark
from landmark import get_landmark
import time

Time={'address':0,'back':0,'back_top':0,'impact':0,'finish':0}
def vedio_time(video_path,landmarks_dict,image_width,image_height):
    cap=cv2.VideoCapture(video_path)
    mp_pose=mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
        # 첫 프레임
        is_first = True
        if is_first:
            first_right_ankle_x = landmarks_dict["right_ankle"][0] * image_width
            first_left_ankle_x = landmarks_dict["left_ankle"][0] * image_width

            first_right_shoulder_y = landmarks_dict["right_shoulder"][1] * image_height

            first_right_eye_inner_y = landmarks_dict["right_eye_inner"][1] * image_height
            is_first = False

        first_ankle_center_x = int((first_left_ankle_x + first_right_ankle_x) / 2)


        while cap.isOpened():
            ret,frame =cap.read()
            if not ret:
                break

            #BGR -> RGB
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            #landmark
            landmarks_dict = get_landmark(mp_pose, results.pose_landmarks.landmark)

            #현재 시각
            current_time = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)

            #현재 프레임의 개수
            current_frame_cnt=0

            #어드레스
            if(first_ankle_center_x == landmarks_dict["right_wrist"][0]):
                Time['address'] = current_time
            #백스윙
            if(first_right_shoulder_y == landmarks_dict["left_pinky"][1]):
                Time['back'] = current_time
            #백스윙_탑

            #임팩트
            if (first_ankle_center_x == landmarks_dict["right_wrist"][0]):
                Time['impact'] = current_time
            #피니시

    cap.release()
    cv2.destroyAllWindows()

    return Time


