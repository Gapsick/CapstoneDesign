import os
import csv

# CSV 파일 경로 설정
base_path = "C:/Code/CapstoneDesign/CapstoneDesign/steering_data_preprocessor"
csv_file = os.path.join(base_path, "data/training_data.csv")
frames_directory = os.path.join(base_path, "data/frames")

# CSV 파일 수정 함수
def fix_csv_paths(csv_file, frames_directory):
    fixed_rows = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 헤더 저장
        for row in reader:
            frame_path, angle = row
            # 경로를 절대 경로로 변환
            absolute_frame_path = os.path.join(frames_directory, os.path.basename(frame_path))
            fixed_rows.append([absolute_frame_path, angle])

    # 수정된 CSV 파일 저장
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 헤더 다시 작성
        writer.writerows(fixed_rows)
    print(f"CSV 파일 경로가 수정되었습니다: {csv_file}")

# 경로 수정 실행
fix_csv_paths(csv_file, frames_directory)
