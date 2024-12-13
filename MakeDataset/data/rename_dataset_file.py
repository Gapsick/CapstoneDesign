import os
import pandas as pd

# CSV 파일 로드
csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/updated_training_data.csv"
data = pd.read_csv(csv_file)

# 경로 정규화
data['frame_path'] = data['frame_path'].apply(lambda x: os.path.normpath(x))

# 업데이트된 CSV 저장
data.to_csv(csv_file, index=False)
print("경로 정규화 완료 및 CSV 업데이트 완료")

# 누락된 파일 확인
missing_files = data[~data['frame_path'].apply(os.path.exists)]

if not missing_files.empty:
    print(f"누락된 파일 개수: {len(missing_files)}")
    print(missing_files)
else:
    print("모든 파일이 존재합니다.")
