#시간을 list에 저장

import cv2
import mediapipe as mp
import time

def coordinates_1(video_path):
    time_list = []  # 시간을 저장할 리스트를 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    target_y1 = 0  # 초기값 설정
    target_y2 = 0  # 초기값 설정

    first_frame = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 포즈인식
        results = pose.process(rgb_frame)
        if results.pose_landmarks:  # 포즈가 감지되면
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            if first_frame:
                target_y1 = right_wrist.y  # 첫 프레임에서 손목을 기준으로 한 y좌표 설정
                target_y2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y  # 첫 프레임에서 어깨를 기준으로 한 y좌표 설정
                first_frame = False

            if right_wrist.y == target_y1:
                time_list.append(time.time())  # 도달하면 값 추가(어드레스, 임팩트)
            if right_wrist.y == target_y2:
                time_list.append(time.time())  # 도달하면 값 추가(테이크백)
    print(time_list)

    cap.release()