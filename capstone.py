#ctrl+s -> git add . -> git commit -m "할 말" -> git push
#git pull

import cv2
import mediapipe as mp
import numpy as np

def pose_drawing(video_path, output_path):
    # mediapipe pose 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    # 동영상 파일의 프레임 수, 프레임 크기 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 출력 동영상 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)


    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:

        while cap.isOpened():
            #프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break

            # 동영상 프레임을 BGR에서 RGB로 변환
            results = pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

            image_height,image_width, _ = frame.shape
            if not results.pose_landmarks:
                continue

            #결과 그리기
            annotated_frame = frame.copy()


            # Draw only the nose landmark
            nose_landmark = results.pose_landmarks.landmark[mp.pose.PoseLandmark.NOSE]
            if nose_landmark.visibility > 0:  # Check if the landmark is visible
                x, y = int(nose_landmark.x * image_width), int(nose_landmark.y * image_height)
                cv2.circle(annotated_frame, (x, y), 10, (255, 255, 255), -1)

            # 결과 동영상 파일에 추가
            out.write(annotated_frame)

            # 결과 출력
            cv2.imshow("MediaPipe Pose", annotated_frame)

            if cv2.waitKey(1)== ord('q'):
                break

    # 종료 후 정리
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "C:\\Users\\eju20\\OneDrive\\capstone\\practice_3.mp4"  # 입력 동영상 파일 경로
    output_path = "C:\\Users\\eju20\\OneDrive\\capstone\\output_1.mp4"  # 출력 동영상 파일 경로
    pose_drawing(video_path, output_path)
