import os
import pandas as pd

# 데이터 로드
csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/training_data_absolute_fixed.csv"
data = pd.read_csv(csv_file)

# 특정 조향각 값 확인 (-30도 이하와 30도 이상이 부족하다고 가정)
angle_threshold_low = -30
angle_threshold_high = 30

# 부족한 데이터 필터링
low_angles = data[data['steering_angle'] < angle_threshold_low]
high_angles = data[data['steering_angle'] > angle_threshold_high]

# 복제하여 Oversampling 수행
oversampled_data = pd.concat([data, low_angles, high_angles], ignore_index=True)

# 저장 경로 설정
output_dir = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data"
output_file = os.path.join(output_dir, "oversampled_training_data.csv")

# 디렉터리 확인 및 생성
os.makedirs(output_dir, exist_ok=True)

# Oversampled 데이터 저장
oversampled_data.to_csv(output_file, index=False)

print(f"Original data size: {len(data)}")
print(f"Oversampled data size: {len(oversampled_data)}")
print(f"Oversampled data saved to: {output_file}")
