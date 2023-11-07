#첫 어드레스 시 얼굴 좌표 기준 원 형성 &사용자 얼굴 인식



import cv2
import mediapipe as mp

def pose_drawing(video_path, output_path):
    # mediapipe pose 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils #그래픽 생성
    mp_drawing_styles = mp.solutions.drawing_styles #그래픽 스타일
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    # 동영상 파일의 프레임 수, 프레임 크기 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 출력 동영상 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:

    #어드레스 시 첫 프레임을 받아오기 위한 플래기
        is_first = True
    #어드레스 시 첫 프레임의 좌표를 저장할 변수
        first_center_x, first_center_y,first_radius = None,None,None

        while cap.isOpened():
            #프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break

            # 동영상 프레임을 BGR에서 RGB로 변환
            results = pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

            image_height, image_width, _ = frame.shape

            if not results.pose_landmarks:
                continue

            # 결과 그리기
            annotated_frame = frame.copy()

            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4))
#*첫 지점
            if results.pose_landmarks:
                # https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
                landmark = results.pose_landmarks.landmark

                left_ear_x = landmark[mp_pose.PoseLandmark.LEFT_EAR].x * image_width
                left_ear_y = landmark[mp_pose.PoseLandmark.LEFT_EAR].y * image_height

                right_ear_x = landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * image_width
                right_ear_y = landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * image_height

                center_x = int((left_ear_x + right_ear_x) / 2)
                center_y = int((left_ear_y + right_ear_y) / 2)

                radius = int((left_ear_x - right_ear_x) / 2)
                radius = max(radius, 20)

                if is_first:  # 어드레스 시 첫 프레임의 머리 좌표 저장
                    first_center_x = center_x
                    first_center_y = center_y
                    first_radius = int(radius * 2)

                    is_first = False
                else:
                    cv2.circle(annotated_frame, center=(first_center_x, first_center_y),
                               radius=first_radius, color=(0, 255, 255), thickness=2)

                    color = (0, 255, 0)  # 초록색

                    # 머리가 원래 위치보다 많이 벗어난 경우
                    if center_x - radius < first_center_x - first_radius \
                            or center_x + radius > first_center_x + first_radius:
                        color = (0, 0, 255)  # 빨간색

                    cv2.circle(annotated_frame, center=(center_x, center_y),
                               radius=radius, color=color, thickness=2)
#*끝 지점
            # 결과 동영상 파일에 추가
            out.write(annotated_frame)

            # 결과 출력
            cv2.imshow("MediaPipe Pose", annotated_frame)


            if cv2.waitKey(1) == ord('q'):
                break

            #좌표값
            #nose_landmark = results.pose_landmarks.landmark[0]
            #nose_x = nose_landmark.x
            #nose_y = nose_landmark.y
            #print('x : {} y : {}'.format(nose_x,nose_y))

    # 종료 후 정리

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'C:\\Users\\hyeeu\\OneDrive\\사진\\카메라 앨범\\golf_vd2.mp4'  # 입력 동영상 파일 경로
    output_path = 'C:\\Users\\hyeeu\\OneDrive\\사진\\카메라 앨범\\output_file.mp4'  # 출력 동영상 파일 경로
    pose_drawing(video_path, output_path)
