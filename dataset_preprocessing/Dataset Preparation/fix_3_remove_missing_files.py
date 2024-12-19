import pandas as pd
import os

# 현재 실행 파일의 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# CSV 파일 경로를 동적으로 설정
csv_path = os.path.join(current_dir, "data/training_data_updated.csv")
updated_csv_path = os.path.join(current_dir, "data/training_data_cleaned.csv")

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 존재하지 않는 파일 확인
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

# 누락된 파일 경로 출력
print(f"누락된 파일 개수: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"누락된 파일 예시: {missing_files[:5]}")

# 누락된 파일을 데이터프레임에서 제거
df = df[~df['frame_path'].isin(missing_files)]

# 수정된 CSV 저장
os.makedirs(os.path.dirname(updated_csv_path), exist_ok=True)  # 폴더 생성 (없으면)
df.to_csv(updated_csv_path, index=False)

print(f"수정된 CSV 파일이 저장되었습니다: {updated_csv_path}")
