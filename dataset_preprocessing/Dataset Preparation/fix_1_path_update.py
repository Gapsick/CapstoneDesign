import pandas as pd
import os

# 현재 실행 파일의 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# CSV 파일 경로 및 수정된 CSV 저장 경로를 동적으로 설정
csv_path = os.path.join(current_dir, "data/training_data.csv")
updated_csv_path = os.path.join(current_dir, "data/training_data_updated.csv")

# 디렉토리가 없으면 생성
os.makedirs(os.path.dirname(updated_csv_path), exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_path)

# frame_path를 현재 디렉토리에 맞게 재생성
frame_base_path = os.path.join(current_dir, "data", "frames")  # 새 프레임 경로
df['frame_path'] = df['frame_path'].apply(lambda x: os.path.join(frame_base_path, os.path.basename(x)))

# 수정된 CSV 저장
df.to_csv(updated_csv_path, index=False)

print(f"CSV 경로가 수정되어 저장되었습니다: {updated_csv_path}")
