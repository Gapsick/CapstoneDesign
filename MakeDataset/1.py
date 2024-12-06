import cv2
import os

# 원본 이미지가 있는 폴더 경로
input_folder = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/frames"

# 결과 이미지를 저장할 폴더 경로
output_folder = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/cropped_frames"

# 저장할 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 모든 파일 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # jpg 파일만 처리
        # 원본 이미지 경로
        image_path = os.path.join(input_folder, filename)
        
        # 이미지 불러오기
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러오지 못했습니다: {image_path}")
            continue
        
        # 이미지 크기 확인
        height, width, _ = image.shape

        # ROI 설정 (도로 영역만 남기기)
        roi_top = int(height * 0.45)  # 상단 60% 잘라내기 (도로만 남기기)
        roi = image[roi_top:height, :]  # 아래쪽 40%만 사용

        # 결과 이미지 저장 경로
        output_path = os.path.join(output_folder, f"cropped_{filename}")

        # 결과 저장
        cv2.imwrite(output_path, roi)
        print(f"전처리된 이미지를 저장했습니다: {output_path}")

print("모든 이미지의 전처리가 완료되었습니다.")
