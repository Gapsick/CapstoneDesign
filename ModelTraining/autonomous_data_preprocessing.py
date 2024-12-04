import os
import cv2
import csv
import numpy as np

# 현재 스크립트 기준 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
processed_frame_path = os.path.join(base_dir, "data/processed_frames")
processed_mask_path = os.path.join(base_dir, "data/processed_masks")
os.makedirs(processed_frame_path, exist_ok=True)
os.makedirs(processed_mask_path, exist_ok=True)

# 기존 절대 경로 기반 CSV 파일 경로
csv_input = r"C:\Code\CapstoneDesign\CapstoneDesign\ModelTraining\data\training_data_absolute_fixed.csv"
csv_output = os.path.join(base_dir, "data/processed_training_data.csv")

# CSV 파일 초기화
with open(csv_output, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frame_path', 'mask_path', 'steering_angle', 'speed'])  # 헤더 작성

# 이미지 크기 및 정규화 처리
with open(csv_input, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # CSV에서 경로 읽기
        frame_path = row['frame_path']
        mask_path = row['mask_path']

        # 이미지 파일 확인
        if not os.path.exists(frame_path):
            print(f"프레임 이미지를 찾을 수 없습니다: {frame_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"마스크 이미지를 찾을 수 없습니다: {mask_path}")
            continue

        # 데이터 로드
        steering_angle = float(row['steering_angle'])
        speed = float(row['speed'])

        # 이미지 파일 로드
        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if frame is None or mask is None:
            print(f"이미지를 열 수 없습니다: {frame_path} 또는 {mask_path}")
            continue

        # 이미지 크기 조정
        resized_frame = cv2.resize(frame, (200, 66))
        resized_mask = cv2.resize(mask, (200, 66))

        # 정규화
        normalized_frame = resized_frame / 255.0
        normalized_mask = resized_mask / 255.0

        # 처리된 이미지 저장 경로
        processed_frame_name = os.path.join(processed_frame_path, os.path.basename(frame_path))
        processed_mask_name = os.path.join(processed_mask_path, os.path.basename(mask_path))

        # 처리된 이미지 저장
        cv2.imwrite(processed_frame_name, (normalized_frame * 255).astype(np.uint8))
        cv2.imwrite(processed_mask_name, (normalized_mask * 255).astype(np.uint8))

        # 처리된 데이터 CSV에 기록
        with open(csv_output, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([processed_frame_name, processed_mask_name, steering_angle, speed])

print("데이터 전처리가 완료되었습니다.")
