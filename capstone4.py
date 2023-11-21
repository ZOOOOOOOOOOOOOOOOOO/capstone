import cv2
import mediapipe as mp
from calculate_angle import calculate_angle
from input_slow import slowmotion
from landmark import get_landmark
from line_landmark import line_landmark


def pose_drawing(video_path, output_path):
    # mediapipe pose 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils  # 그래픽 생성
    mp_drawing_styles = mp.solutions.drawing_styles  # 그래픽 스타일
    cap = cv2.VideoCapture(video_path) #동영상 파일 열기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 총 프레임 개수
    current_frame_cnt = 0 #현재 프레임 개수
    fps = int(cap.get(cv2.CAP_PROP_FPS)) #초당? 프레임 수
    # 시각 리스트
    time_list = [10]
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 출력 동영상 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    address_1 = 0  # 발과 어꺠넓이 확인용 #1이면 피드백 0이면 ㅇㅋ
    backswing_1 = 0  # 어드레스 백스윙 전 구간동안 팔이 곧게 뻗어있는지 확인용 #1이면 피드백 필요
    count = 0  # 백스윙 시 160도 안 넘을 경우 count 증가

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:

        # 어드레스 시 첫 프레임을 받아오기 위한 플래기
        is_first = True
        # 어드레스 시 첫 프레임의 좌표를 저장할 변수
        first_head_center_x, first_head_center_y, first_radius = None, None, None
        first_ankle_center_x, first_ankle_center_y = None, None

        while cap.isOpened():
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #동영상 프레임을 BGER->RGB로 변환
            current_time = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) #프레임 당 현재 시각
            image_height, image_width, _ = frame.shape

            if not results.pose_landmarks:
                continue

            annotated_frame = frame.copy() # 결과 그리기

            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4))

# get_landmark 함수 호출
            landmark = results.pose_landmarks.landmark
            landmarks_dict = get_landmark(mp_pose, landmark)

# 사잇각 구하는 방법 (landmark 이용)
            #print(calculate_angle(landmarks_dict["left_shoulder"], landmarks_dict["left_elbow"],landmarks_dict["left_wrist"]))
            # 끝

# 페이스/기준선 라인 첫 지점
                # 페이스 작업을 위한 랜드마크
            head_center_x, head_center_y, ankle_center_x, ankle_center_y, radius = line_landmark(landmarks_dict, image_width, image_height)

            if is_first:  # 어드레스 시 첫 프레임의 머리 좌표 저장
                first_head_center_x, first_head_center_y, first_ankle_center_x, first_ankle_center_y, first_radius = head_center_x, head_center_y, ankle_center_x, ankle_center_y, int(radius * 2)
                is_first = False
            else:
                # 첫 프레임 헤드 생성
                cv2.circle(annotated_frame, center=(first_head_center_x, first_head_center_y),radius=first_radius, color=(0, 255, 255), thickness=2)
                # 기준 선 라인 생성
                cv2.line(annotated_frame, (first_ankle_center_x, 0), (first_ankle_center_x, image_height),(198, 219, 218), 2)
                color = (0, 255, 0)  # 초록색
                # 머리가 원래 위치보다 많이 벗어난 경우 ->초록에서 빨강
                if head_center_x - radius < first_head_center_x - first_radius or head_center_x + radius > first_head_center_x + first_radius: color = (0, 0, 255)  # 빨간색
                # 실시간 헤드라인
                cv2.circle(annotated_frame, center=(head_center_x, head_center_y),radius=radius, color=color, thickness=2)
# 페이스/기준선 끝 지점

            ###backswing_1 : 어드레스-백스윙 첫 구간동안의 사잇각 확인 지점

            # 현재 프레임 개수
            current_frame_cnt = current_frame_cnt + 1

            left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # ankle_center_x가 image너비를 곱한 값이라 right_wrist또한 같은 방식으로 치뤄줌
            right_wrist_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_width
            right_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height
            left_elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height
            # 어드레스 시작 구간의 시각 time_list에 담기
            if (ankle_center_x == int(right_wrist_x)):
                time_list[0] = current_time

            print(right_wrist_y, left_elbow_y)
            if (right_wrist_y >= (left_elbow_y)):
                print(current_time)

            if time_list[0] <= current_time <= 5000:  # 주현이가 어드레스랑 백스윙 구간 내의 시간을 주면 가능함
                # 0과 5000(5초)은 현재 임시 초
                # 사잇각 계산

                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                # 사잇각이 170도를 넘으면
                if left_arm_angle < 170:
                    # count 1 증가 (1프레임 당 1씩 증가임)
                    count = count + 1
            # 165도 넘은게 구간 내의 (총프레임/2)보다 많으면 (==반 이상이 잘못된 자세일 경우)
            if (count > (current_frame_cnt / 2)):
                backswing_1 = 0  # 피드백 필요한 경우
            else:
                backswing_1 = 1  # 피드백 필요없는 경

            # 결과 동영상 파일에 추가
            out.write(annotated_frame)

            # 결과 출력
            cv2.imshow("MediaPipe Pose", annotated_frame)

            if cv2.waitKey(1) == ord('q'):
                break

    # 어드레스, 벡스윙 결과
    print('어드레스', address_1)
    print('170 못 넘긴 개수', count, '|프레임개수', current_frame_cnt)
    print('백스윙 피드백 필요시 0', '|백스윙 결과', backswing_1)
    # 종료 후 정리

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'C:\\Users\\hyeeu\\OneDrive\\사진\\카메라 앨범\\pro2.mp4'  # 입력 동영상 파일 경로
    output_path = 'C:\\Users\\hyeeu\\OneDrive\\사진\\카메라 앨범\\output_file4.mp4'  # 출력 동영상 파일 경로
    # 쭈현이꺼
    # video_path = "C:\\Users\\eju20\\OneDrive\\capstone\\practice_3.mp4"  # 입력 동영상 파일 경로
    # output_path = "C:\\Users\\eju20\\OneDrive\\capstone\\output_1.mp4"  # 출력 동영상 파일 경로
    slow_path = slowmotion(video_path)
    pose_drawing(slow_path, output_path)

