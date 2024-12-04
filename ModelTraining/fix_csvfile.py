import os

# 기존 및 새로운 CSV 파일 경로
input_csv = r"C:\Code\CapstoneDesign\CapstoneDesign\ModelTraining\data\training_data_absolute.csv"
output_csv = r"C:\Code\CapstoneDesign\CapstoneDesign\ModelTraining\data\training_data_absolute_fixed.csv"

# Windows 경로로 변환
base_dir = r"C:\Code\CapstoneDesign\CapstoneDesign\ModelTraining"
frame_dir = os.path.join(base_dir, "data", "frames")
mask_dir = os.path.join(base_dir, "data", "masks")

# CSV 파일 수정
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    lines = infile.readlines()
    outfile.write(lines[0])  # 헤더 복사

    for line in lines[1:]:
        frame_path, mask_path, steering_angle, speed = line.strip().split(',')
        new_frame_path = os.path.join(frame_dir, os.path.basename(frame_path))
        new_mask_path = os.path.join(mask_dir, os.path.basename(mask_path))
        outfile.write(f"{new_frame_path},{new_mask_path},{steering_angle},{speed}\n")

print("CSV 파일 경로가 Windows 절대 경로로 수정되었습니다.")
