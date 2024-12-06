import os
import cv2
import csv
import numpy as np

# 현재 스크립트 기준 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
processed_frame_path = os.path.join(base_dir, "data/processed_frames")
os.makedirs(processed_frame_path, exist_ok=True)

# 이미지가 있는 폴더 경로 (cropped_frames 폴더)
image_folder = r"C:\Code\CapstoneDesign\CapstoneDesign\MakeDataset\cropped_frames"

# CSV 출력 파일 경로
csv_output = os.path.join(base_dir, "data/processed_training_data.csv")

# CSV 파일 초기화
with open(csv_output, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frame_path', 'steering_angle', 'speed'])  # 헤더 작성

# 전처리 실행
for img_name in os.listdir(image_folder):
    # 이미지 파일 경로
    img_path = os.path.join(image_folder, img_name)
    
    # 이미지 파일 확인
    if not os.path.isfile(img_path) or not img_name.lower().endswith(('.jpg', '.png')):
        print(f"이미지 파일이 아닙니다: {img_path}")
        continue

    # 이미지 로드
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"이미지를 열 수 없습니다: {img_path}")
        continue

    # 이미지 크기 조정
    resized_frame = cv2.resize(frame, (200, 66))

    # 정규화
    normalized_frame = resized_frame / 255.0

    # 처리된 이미지 저장 경로
    processed_frame_name = os.path.join(processed_frame_path, img_name)

    # 처리된 이미지 저장
    cv2.imwrite(processed_frame_name, (normalized_frame * 255).astype(np.uint8))

    # 임시 라벨 (steering_angle, speed 예제 값 사용)
    steering_angle = 0.0  # 각도 예제 값
    speed = 0.0  # 속도 예제 값

    # 처리된 데이터 CSV에 기록
    with open(csv_output, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([processed_frame_name, steering_angle, speed])

print("데이터 전처리가 완료되었습니다.")
