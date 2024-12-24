import os
import pandas as pd

# 현재 스크립트의 디렉토리 기준 경로 생성
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "data", "training_data.csv")
updated_csv_path = os.path.join(current_dir, "data", "training_data_updated.csv")

# 디렉토리 생성
os.makedirs(os.path.dirname(updated_csv_path), exist_ok=True)

# CSV 파일 로드
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # 경로 수정
    df['frame_path'] = df['frame_path'].str.replace("1213", "steering_data_preprocessor", regex=False)

    # 수정된 CSV 저장
    df.to_csv(updated_csv_path, index=False)
    print(f"CSV 경로가 수정되어 저장되었습니다: {updated_csv_path}")
else:
    print(f"CSV 파일이 존재하지 않습니다: {csv_path}")
