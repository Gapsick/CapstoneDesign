import cv2
import os
import math
import csv
import numpy as np

# 데이터 저장 경로 설정
frame_save_path = "data/frames"
mask_save_path = "data/masks"
os.makedirs(frame_save_path, exist_ok=True)  # 프레임 저장 폴더 생성
os.makedirs(mask_save_path, exist_ok=True)  # 마스크 저장 폴더 생성

# CSV 파일 초기화
csv_file = "data/training_data.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frame_path', 'mask_path', 'steering_angle', 'speed'])  # 헤더 작성

# 카메라 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
collecting_data = False  # 데이터를 수집 중인지 여부

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라 입력을 읽을 수 없습니다.")
        break

    height, width, _ = frame.shape

    # ROI 설정 (하단 1/3 영역)
    roi = frame[int(height * 2 / 3):, :]

    # HSV 변환 및 색상 필터링
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120])  # 흰색 하한값
    upper_white = np.array([180, 80, 255])  # 흰색 상한값
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Gaussian Blur 추가
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 데이터 수집 중일 때만 저장
    if collecting_data:
        # 컨투어 검출 및 방향 각도 계산
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                center = width // 2
                steering_angle = math.atan2(cx - center, height) * (180 / math.pi)
                speed = 70 if abs(steering_angle) < 10 else 50 if abs(steering_angle) < 20 else 30

                # 파일 이름 지정
                frame_filename = f"frame_{frame_count:05d}.jpg"
                mask_filename = f"mask_{frame_count:05d}.jpg"

                # 파일 경로 결합
                frame_path = os.path.join(frame_save_path, frame_filename)
                mask_path = os.path.join(mask_save_path, mask_filename)

                # 이미지 저장
                cv2.imwrite(frame_path, frame)
                cv2.imwrite(mask_path, mask)

                # CSV 데이터 저장
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_path, mask_path, steering_angle, speed])

                print(f"Saved: {frame_path}, {mask_path}, Steering Angle: {steering_angle:.2f}, Speed: {speed}")
                frame_count += 1

    # 화면 출력
    cv2.imshow("ROI", roi)
    cv2.imshow("Mask", mask)

    # 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q'를 누르면 시작/중지 토글
        collecting_data = not collecting_data
        if collecting_data:
            print("데이터 수집 시작")
        else:
            print("데이터 수집 중지")
    elif key == ord('x'):  # 'x'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
