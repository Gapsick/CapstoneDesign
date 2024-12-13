import os
import cv2
import pandas as pd
import numpy as np

# CSV 파일 경로
csv_path = "C:/Code/CapstoneDesign/CapstoneDesign/steering_data_preprocessor/data/oversampled_training_data.csv"

# 데이터 로드
df = pd.read_csv(csv_path)

# 전처리된 데이터 저장 경로
processed_images_path = "C:/Code/CapstoneDesign/CapstoneDesign/steering_data_preprocessor/data/processed_images"
os.makedirs(processed_images_path, exist_ok=True)

# 데이터 전처리 함수
def preprocess_image(image_path):
    # 경로에서 백슬래시(\)를 슬래시(/)로 변환
    image_path = image_path.replace("\\", "/")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 크기 조정
    resized_image = cv2.resize(image, (200, 66))
    
    # 정규화 (픽셀 값: 0~1)
    normalized_image = resized_image / 255.0
    
    return normalized_image

# 조향각 정규화 함수
def normalize_steering_angle(angle):
    return angle / 180.0  # -180~180도 데이터를 -1~1로 변환

# 전처리 수행
processed_data = []
for _, row in df.iterrows():
    try:
        # 이미지 전처리
        image_path = row['frame_path']
        preprocessed_image = preprocess_image(image_path)
        
        # 이미지 저장
        new_image_path = os.path.join(processed_images_path, os.path.basename(image_path)).replace("\\", "/")
        cv2.imwrite(new_image_path, (preprocessed_image * 255).astype(np.uint8))  # 정규화 해제 후 저장
        
        # 조향각 정규화
        normalized_angle = normalize_steering_angle(row['steering_angle'])
        
        # 결과 저장
        processed_data.append({
            "frame_path": new_image_path,
            "steering_angle": normalized_angle
        })
    except Exception as e:
        print(f"에러 발생: {e}")

# 처리된 데이터 저장
processed_csv_path = "C:/Code/CapstoneDesign/CapstoneDesign/steering_data_preprocessor/data/processed_training_data.csv"
pd.DataFrame(processed_data).to_csv(processed_csv_path, index=False)

print(f"전처리가 완료되었습니다. 데이터는 {processed_csv_path}에 저장되었습니다.")
