import os
import pandas as pd

# 현재 실행 파일의 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# **1. CSV 파일 경로**
csv_path = os.path.join(current_dir, "data", "oversampled_training_data.csv")
updated_csv_path = os.path.join(current_dir, "data", "final_training_data_updated.csv")

# **2. 데이터 로드**
df = pd.read_csv(csv_path)

# **3. 경로 변경 설정**
# 입력 프레임 경로를 수정 (예시: augmented_frames/cropped_ 추가)
old_base_path = os.path.join(current_dir, "data", "frames") + os.sep
new_base_path = os.path.join(current_dir, "data", "augmented_frames", "cropped_")

# 기존 경로를 새로운 경로로 변경
df['frame_path'] = df['frame_path'].str.replace(old_base_path, new_base_path, regex=False)

# **4. 경로 변경 결과 확인**
# 경로 확인 예시 출력 (5개 항목만 출력)
print("변경된 경로 예시 (상위 5개):")
print(df['frame_path'].head())

# **5. 파일 존재 여부 확인**
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

print(f"누락된 파일 개수: {len(missing_files)}")
if missing_files:
    print(f"누락된 파일 예시: {missing_files[:5]}")
else:
    print("모든 파일이 정상적으로 존재합니다.")

# **6. 변경된 CSV 저장**
os.makedirs(os.path.dirname(updated_csv_path), exist_ok=True)
df.to_csv(updated_csv_path, index=False)
print(f"경로가 수정된 CSV를 저장했습니다: {updated_csv_path}")
