#기준 선
#각도 재는 방법 (일단 하나만)
#58-84 line

import cv2
import mediapipe as mp
import numpy as np

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
        first_head_center_x, first_head_center_y,first_radius = None,None,None
        first_ankle_center_x,first_ankle_center_y =None,None

        second_center_x, second_center_y = None,None

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

            #특정 랜드마크 x,y,z좌표,visibility 알 수 있음 (3줄)
            #landmark = results.pose_landmarks.landmark
            #mark_x=landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            #print(mark_x)

            def calculate_angle(a,b,c):

                #각 값을 받아 넘파이 배열로 변형
                a = np.array(a) #첫 번째
                b = np.array(b) #두 번째
                c = np.array(c) #세 번째

                #라디안을 계산하고 실제 각도로 변경한다.
                radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
                angle  = np.abs(radians*180.0/np.pi)

                #180도가 넘으면 360에서 뺀 값을 계산한다.
                if angle > 180.0:
                    angle = 360-angle

                #각도를 리턴한다.
                return print(angle)

            landmark = results.pose_landmarks.landmark #랜드마크 지정
            shoulder = [landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]


            calculate_angle(shoulder,elbow,wrist)
#페이스/기준선 라인 첫 지점
            if results.pose_landmarks:
                # https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
                landmark = results.pose_landmarks.landmark
                
                #페이스 작업을 위한 랜드마크
                left_ear_x = landmark[mp_pose.PoseLandmark.LEFT_EAR].x * image_width
                left_ear_y = landmark[mp_pose.PoseLandmark.LEFT_EAR].y * image_height

                right_ear_x = landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * image_width
                right_ear_y = landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * image_height

                head_center_x = int((left_ear_x + right_ear_x) / 2)
                head_center_y = int((left_ear_y + right_ear_y) / 2)

                radius = int((left_ear_x - right_ear_x) / 2)
                radius = max(radius, 20)

                #기준선 작업을 위한 랜드마크
                left_ankle_x = landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width  # image_width 이미지 너비(가로)
                left_ankle_y = landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height  # image_height 이미지 높이(세로)

                right_ankle_x = landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width
                right_ankle_y = landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height

                ankle_center_x = int((left_ankle_x + right_ankle_x) / 2)
                ankle_center_y = int((left_ankle_y + right_ankle_y) / 2)

                if is_first:  # 어드레스 시 첫 프레임의 머리 좌표 저장
                    first_head_center_x = head_center_x
                    first_head_center_y = head_center_y
                    first_ankle_center_x = ankle_center_x
                    first_ankle_center_y = ankle_center_y
                    first_radius = int(radius * 2)

                    is_first = False
                else:
                    #첫 프레임 헤드 생성
                    cv2.circle(annotated_frame, center=(first_head_center_x, first_head_center_y),
                               radius=first_radius, color=(0, 255, 255), thickness=2)
                    #기준 선 라인 생성
                    cv2.line(annotated_frame,(first_ankle_center_x,0),(first_ankle_center_x,image_height),(198,219,218),2)
                    
                    color = (0, 255, 0)  # 초록색

                    # 머리가 원래 위치보다 많이 벗어난 경우
                    if head_center_x - radius < first_head_center_x - first_radius \
                            or head_center_x + radius > first_head_center_x + first_radius:
                        color = (255, 0,0)  # 빨간색

                    #실시간 헤드 
                    cv2.circle(annotated_frame, center=(head_center_x, head_center_y),
                               radius=radius, color=color, thickness=2)
#페이스/기준선 끝 지점

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
