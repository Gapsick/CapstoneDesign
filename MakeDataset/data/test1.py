import os
import pandas as pd

# 수정된 CSV 파일 경로
csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/training_data_absolute_fixed.csv"

# CSV 파일 읽기
data = pd.read_csv(csv_file)

# 경로 유효성 검사 함수
def validate_path(path):
    # 모든 경로를 슬래시(`/`)로 변환
    corrected_path = path.replace("\\", "/")
    return os.path.exists(corrected_path)

# 경로 유효성 검사
invalid_paths = []
for index, path in enumerate(data['frame_path']):
    corrected_path = path.replace("\\", "/")  # 슬래시 수정
    if not os.path.exists(corrected_path):
        invalid_paths.append((index, corrected_path))  # 인덱스와 경로를 저장

# 결과 출력
if invalid_paths:
    print(f"총 {len(invalid_paths)}개의 유효하지 않은 경로가 발견되었습니다:")
    for index, path in invalid_paths:
        print(f"유효하지 않은 경로: {path} (Index: {index})")
else:
    print("모든 경로가 유효합니다!")
