import cv2

# 전처리된 샘플 이미지 경로
sample_frame_path = r"C:\Code\CapstoneDesign\CapstoneDesign\ModelTraining\data\processed_frames\frame_00000.jpg"
sample_mask_path = r"C:\Code\CapstoneDesign\CapstoneDesign\ModelTraining\data\processed_masks\mask_00000.jpg"

# 이미지 로드
frame = cv2.imread(sample_frame_path)
mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)

# 이미지 시각화
cv2.imshow("Frame", frame)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
