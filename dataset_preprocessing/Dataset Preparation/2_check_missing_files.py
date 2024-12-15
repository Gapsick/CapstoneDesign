import pandas as pd
import os

# 수정된 CSV 파일 경로
csv_path = "C:/Code/CapstoneDesign/CapstoneDesign/steering_data_preprocessor/data/training_data_updated.csv"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 존재하지 않는 파일 확인
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

# 결과 출력
print(f"최종 누락된 파일 개수: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"누락된 파일 예시: {missing_files[:5]}")
else:
    print("모든 파일이 존재합니다.")
