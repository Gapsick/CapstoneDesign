import pandas as pd
import os

# CSV 파일 경로와 이미지 폴더
csv_path = "C:/Code/CapstoneDesign/CapstoneDesign/steering_data_preprocessor/data/oversampled_training_data.csv"
df = pd.read_csv(csv_path)

# 이미지가 실제로 존재하는지 확인
missing_files = []
for path in df['frame_path']:
    if not os.path.exists(path):
        missing_files.append(path)

if missing_files:
    print(f"누락된 파일:\n{missing_files}")
else:
    print("모든 이미지가 존재합니다.")
