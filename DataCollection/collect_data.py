import cv2
import os
import math
import csv
import numpy as np

# 데이터 저장 경로 설정
image_save_path = "data/images"
os.makedirs(image_save_path, exist_ok=True)  # 폴더 생성

# CSV 파일 초기화
csv_file = "training_data.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_path', 'steering_angle', 'speed'])  # 헤더 작성

# 카메라 초기화
cap = cv2.VideoCapture(0)  # 0번 카메라 (필요 시 변경)

frame_count = 0  # 프레임 번호 초기화

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. ROI 설정 (이미지 하단 1/3만 처리)
    height, width, _ = frame.shape
    roi = frame[int(height * 2 / 3):, :]  # 하단 1/3 영역

    # 2. HSV 변환 및 색상 필터링
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 3. 라인의 중심 계산
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])  # 중심 x 좌표
            cy = int(M['m01'] / M['m00'])  # 중심 y 좌표

            # 방향 각도 계산
            center = width // 2
            steering_angle = math.atan2(cx - center, height) * (180 / math.pi)

            # 속도 계산 (단순 로직)
            if abs(steering_angle) > 20:
                speed = 30  # 큰 각도에서는 낮은 속도
            elif abs(steering_angle) > 10:
                speed = 50  # 중간 각도에서는 중간 속도
            else:
                speed = 70  # 작은 각도에서는 높은 속도

            # 이미지 저장
            image_filename = f"frame_{frame_count:05d}.jpg"
            image_path = os.path.join(image_save_path, image_filename)
            cv2.imwrite(image_path, frame)

            # 데이터 저장
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_path, steering_angle, speed])

            print(f"Saved: {image_path}, Steering Angle: {steering_angle:.2f}, Speed: {speed}")

            # 시각화 (중심점과 라인 표시)
            cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)
            cv2.line(roi, (cx, 0), (cx, height), (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("ROI", roi)
    cv2.imshow("Mask", mask)

    # 프레임 증가
    frame_count += 1

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
