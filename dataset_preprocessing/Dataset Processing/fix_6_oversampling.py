import pandas as pd
import matplotlib.pyplot as plt
import os

# 현재 실행 파일의 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# CSV 파일 경로를 동적으로 설정
csv_path = os.path.join(current_dir, "data", "training_data_cleaned.csv")
balanced_csv_path = os.path.join(current_dir, "data", "oversampled_training_data.csv")

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 1. -10과 10 제거
df_filtered = df[(df['steering_angle'] != -10) & (df['steering_angle'] != 10)]

# 2. Oversampling
# 각 각도의 데이터 개수를 확인
angle_counts = df_filtered['steering_angle'].value_counts()
print("각도별 데이터 분포:\n", angle_counts)

# Oversampling 대상: 가장 많은 각도 데이터(90도)의 개수를 기준으로 맞춤
max_count = angle_counts.max()

# 각 각도를 Oversampling
df_oversampled = df_filtered.groupby('steering_angle', group_keys=False).apply(
    lambda x: x.sample(max_count, replace=True, random_state=42)
)

# 3. 결과 저장
os.makedirs(os.path.dirname(balanced_csv_path), exist_ok=True)  # 폴더 생성 (없으면)
df_oversampled.to_csv(balanced_csv_path, index=False)
print(f"Oversampled 데이터셋이 저장되었습니다: {balanced_csv_path}")

# 4. 새로운 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(df_oversampled['steering_angle'], bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title("Oversampled Steering Angle Distribution")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
