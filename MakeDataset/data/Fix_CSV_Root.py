import os

# 기존 및 새로운 CSV 파일 경로
input_csv = r"C:\Code\CapstoneDesign\CapstoneDesign\MakeDataset\data\training_data_absolute.csv"
output_csv = r"C:\Code\CapstoneDesign\CapstoneDesign\MakeDataset\data\training_data_absolute_fixed.csv"

# `processed_frames` 디렉토리 경로 설정
processed_frames_dir = r"C:\Code\CapstoneDesign\CapstoneDesign\MakeDataset\data\processed_frames"

# CSV 파일 수정
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    lines = infile.readlines()
    outfile.write(lines[0])  # 헤더 복사 (첫 줄)

    for line in lines[1:]:  # 데이터 라인
        columns = line.strip().split(',')  # CSV를 쉼표로 분리
        frame_path = columns[0]  # frame_path 가져오기
        # 새로운 frame_path 생성
        new_frame_path = os.path.join(processed_frames_dir, os.path.basename(frame_path)).replace("\\", "/")
        # 수정된 데이터 작성
        outfile.write(f"{new_frame_path},{','.join(columns[1:])}\n")

print("CSV 파일 경로가 Windows 절대 경로로 수정되었습니다.")
