import cv2
import mediapipe as mp
import landmark

time={'address':0,'back':0,'back_top':0,'impact':0,'finish':0}
def time(video_path):
    cap=cv2.VideoCapture(video_path)
    mp_pose=mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
        # 첫 프레임
        is_first = True
        # 첫 프레임에서 기준값
        first_left_ankle_x,first_left_ankle_y=None,None
        first_right_ankle_x,first_right_ankle_y=None,None
        first_ankle_center_x, first_ankle_center_y = None, None
        first_shoulder_x,first_shoulder_y=None
        first_eye_inner_x,first_eye_inner_y=None

        first_ankle_center_x=int((first_left_ankle_x + first_right_ankle_x)/2)
        first_ankle_center_y=int((first_left_ankle_x + first_right_ankle_y)/2)

        while cap.isOpened():
            ret,frame =cap.read()
            if not ret:
                break

            #BGR -> RGB
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            #landmark
            landmark = results.pose_landmarks.landmark
            landmarks_dict = get_landmark(mp_pose, landmark)

            #현재 시각
            current_time=(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)

            #현재 프레임의 개수
            current_frame_cnt=0

            #어드레스
            if(first_ankle_center_x==landmarks_dict[16][0]):
                time['address']=current_time
            #백스윙
            if(first_shoulder_y==landmarks_dict[17][1]):
                time['back']=current_time
            #백스윙_탑

            #임팩트
            if (first_ankle_center_x == landmarks_dict[16][0]):
                time['impact'] = current_time
            #피니시

    cap.release()
    cv2.destroyAllWindows()