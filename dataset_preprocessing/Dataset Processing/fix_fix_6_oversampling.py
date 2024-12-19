import pandas as pd
import matplotlib.pyplot as plt
import os

# 현재 실행 파일의 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# CSV 파일 경로 설정
csv_path = os.path.join(current_dir, "data", "training_data_cleaned.csv")
filtered_csv_path = os.path.join(current_dir, "data", "filtered_training_data.csv")
balanced_csv_path = os.path.join(current_dir, "data", "oversampled_training_data.csv")

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 1. 유효한 각도 값만 필터링
valid_angles = [30, 60, 90, 120, 150]
df_filtered = df[df['steering_angle'].isin(valid_angles)]
print(f"유효한 각도만 필터링 완료: {valid_angles}")
print(f"필터링 후 데이터 개수: {len(df_filtered)}")

# 2. Oversampling
angle_counts = df_filtered['steering_angle'].value_counts()
print("각도별 데이터 분포:\n", angle_counts)

# 가장 많은 각도 데이터 개수를 기준으로 Oversampling
max_count = angle_counts.max()
df_oversampled = df_filtered.groupby('steering_angle', group_keys=False).apply(
    lambda x: x.sample(max_count, replace=True, random_state=42)
)

# 3. 결과 저장
os.makedirs(os.path.dirname(balanced_csv_path), exist_ok=True)  # 폴더 생성
df_oversampled.to_csv(balanced_csv_path, index=False)
print(f"유효 각도 필터링 및 Oversampled 데이터셋이 저장되었습니다: {balanced_csv_path}")

# 4. 새로운 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(df_oversampled['steering_angle'], bins=len(valid_angles), color='green', alpha=0.7, edgecolor='black')
plt.title("Oversampled Steering Angle Distribution (Filtered)")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
