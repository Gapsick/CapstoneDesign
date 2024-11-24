import cv2
import os
import math
import csv
import numpy as np
import time

# 데이터 저장 경로 설정
image_save_path = "data/images"  # 이미지 파일을 저장할 경로
os.makedirs(image_save_path, exist_ok=True)  # 데이터 폴더가 없으면 생성

# CSV 파일 초기화
csv_file = "training_data.csv"  # 학습 데이터를 저장할 CSV 파일
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_path', 'steering_angle', 'speed'])  # 헤더 작성

# 카메라 초기화
cap = cv2.VideoCapture(0)  # 0번 카메라 (필요 시 변경)

# 프레임 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

# FPS 설정
desired_fps = 30  # 원하는 FPS
frame_time = 1.0 / desired_fps  # 한 프레임 처리에 필요한 시간 (초)

frame_count = 0  # 프레임 번호 초기화

while cap.isOpened():
    start_time = time.time()  # 프레임 시작 시간 기록

    # 카메라에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라 입력을 읽을 수 없습니다. 프로그램을 종료합니다.")
        break

    # 프레임 크기 확인
    height, width, _ = frame.shape

    # 1. ROI(Region of Interest) 설정 (이미지 하단 1/3만 처리)
    roi = frame[int(height * 2 / 3):, :]  # 프레임의 하단 1/3 영역

    # 2. HSV 변환 및 색상 필터링
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # BGR에서 HSV 색공간으로 변환
    lower_white = np.array([0, 0, 200])  # 흰색 하한선
    upper_white = np.array([180, 30, 255])  # 흰색 상한선
    mask = cv2.inRange(hsv, lower_white, upper_white)  # 흰색 마스크 생성

    # 3. 라인의 중심 계산
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 컨투어 감지
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # 가장 큰 컨투어 선택
        M = cv2.moments(largest_contour)  # 선택된 컨투어의 모멘트 계산
        if M['m00'] > 0:  # 모멘트의 면적이 0보다 크면 중심 계산
            cx = int(M['m10'] / M['m00'])  # 중심 x 좌표
            cy = int(M['m01'] / M['m00'])  # 중심 y 좌표

            # 방향 각도 계산 (차량 중심과 라인의 중심 간의 기울기)
            center = width // 2  # 프레임 중심 (차량 중심으로 가정)
            steering_angle = math.atan2(cx - center, height) * (180 / math.pi)  # 각도 계산

            # 속도 계산 (단순 조건 기반 로직)
            if abs(steering_angle) > 20:
                speed = 30  # 큰 각도에서는 속도 낮춤
            elif abs(steering_angle) > 10:
                speed = 50  # 중간 각도에서는 중간 속도
            else:
                speed = 70  # 작은 각도에서는 높은 속도

            # 이미지 저장
            image_filename = f"frame_{frame_count:05d}.jpg"  # 파일 이름 지정
            image_path = os.path.join(image_save_path, image_filename)  # 경로와 파일명 결합
            cv2.imwrite(image_path, frame)  # 이미지를 지정된 경로에 저장

            # 데이터 저장 (CSV 파일에 저장)
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_path, steering_angle, speed])  # 경로와 라벨 데이터 저장

            print(f"Saved: {image_path}, Steering Angle: {steering_angle:.2f}, Speed: {speed}")

            # 시각화 (ROI에 중심점과 라인 표시)
            cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)  # 중심점 표시 (파란색 원)
            cv2.line(roi, (cx, 0), (cx, height), (0, 255, 0), 2)  # 수직선 표시 (초록색 선)

    # 4. 프레임 출력
    cv2.imshow("ROI", roi)  # ROI 영역 출력
    cv2.imshow("Mask", mask)  # 흰색 마스크 출력

    # FPS 조정 (처리 속도에 따라 대기 시간 설정)
    elapsed_time = time.time() - start_time  # 프레임 처리에 소요된 시간
    sleep_time = max(0, frame_time - elapsed_time)  # 남은 대기 시간 계산
    time.sleep(sleep_time)  # 대기 시간 적용

    # 프레임 번호 증가
    frame_count += 1

    # 종료 조건 (q 키를 누르면 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
