import cv2
import os

# 현재 실행 파일의 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# 원본 이미지 폴더와 출력 폴더 경로 설정 (동적 경로)
input_folder = os.path.join(current_dir, "data", "frames")
output_folder = os.path.join(current_dir, "data", "augmented_frames")

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 파일 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # 다양한 이미지 파일 처리
        # 원본 이미지 경로
        image_path = os.path.join(input_folder, filename)

        # 이미지 불러오기
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러오지 못했습니다: {image_path}")
            continue

        # 이미지 크기 확인
        height, width, _ = image.shape
        print(f"원본 이미지 크기: {width}x{height}")

        # ROI 설정 (상단 30% 제거)
        roi_top = int(height * 0.3)  # 상단 30%
        roi = image[roi_top:height, :]  # 상단 제거 후 하단 영역만 남기기
        print(f"ROI 크기: {roi.shape[1]}x{roi.shape[0]}")

        # 크기 조정 (모델 입력 크기)
        roi_resized = cv2.resize(roi, (200, 66))  # 200x66 크기로 조정
        print(f"리사이즈 후 크기: {roi_resized.shape[1]}x{roi_resized.shape[0]}")

        # 결과 이미지 저장 경로
        output_path = os.path.join(output_folder, f"cropped_{os.path.splitext(filename)[0]}.jpg")

        # 결과 저장
        cv2.imwrite(output_path, roi_resized)
        print(f"전처리된 이미지를 저장했습니다: {output_path}")

print("모든 이미지의 전처리가 완료되었습니다.")
