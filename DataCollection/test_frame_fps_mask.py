import cv2
import os
import math
import csv
import numpy as np
import time

# 데이터 저장 경로 설정
image_save_path = "data/images"
os.makedirs(image_save_path, exist_ok=True)

# CSV 파일 초기화
csv_file = "training_data.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_path', 'mask_path', 'steering_angle', 'speed'])  # 마스크 경로 추가

# 카메라 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

desired_fps = 30
frame_time = 1.0 / desired_fps

frame_count = 0

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("카메라 입력을 읽을 수 없습니다.")
        break

    height, width, _ = frame.shape

    # ROI 설정
    roi = frame[int(height * 2 / 3):, :]

    # HSV 변환 및 색상 필터링
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 라인의 중심 계산
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            center = width // 2
            steering_angle = math.atan2(cx - center, height) * (180 / math.pi)

            if abs(steering_angle) > 20:
                speed = 30
            elif abs(steering_angle) > 10:
                speed = 50
            else:
                speed = 70

            # 이미지 저장
            image_filename = f"frame_{frame_count:05d}.jpg"
            image_path = os.path.join(image_save_path, image_filename)
            cv2.imwrite(image_path, frame)

            # 마스크 저장
            mask_filename = f"mask_{frame_count:05d}.jpg"
            mask_path = os.path.join(image_save_path, mask_filename)
            cv2.imwrite(mask_path, mask)

            # 데이터 저장
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_path, mask_path, steering_angle, speed])

            print(f"Saved: {image_path}, Mask: {mask_path}, Steering Angle: {steering_angle:.2f}, Speed: {speed}")

            # 시각화
            cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)
            cv2.line(roi, (cx, 0), (cx, height), (0, 255, 0), 2)

    # 시각화 출력
    cv2.imshow("ROI", roi)
    cv2.imshow("Mask", mask)

    elapsed_time = time.time() - start_time
    time.sleep(max(0, frame_time - elapsed_time))

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
