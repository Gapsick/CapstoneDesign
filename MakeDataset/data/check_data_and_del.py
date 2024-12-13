import os
import pandas as pd
import cv2
import keyboard  # 키 입력 감지용 모듈

# CSV 파일 경로 및 데이터 로드
csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/updated_training_data.csv"
data = pd.read_csv(csv_file)

# CSV 파일 업데이트용 리스트
updated_data = []

print("데이터 확인 시작: 이미지를 확인하고, 삭제하려면 'D'를 누르세요. 다음으로 넘어가려면 아무 키나 누르세요.")

for idx, row in data.iterrows():
    frame_path = row['frame_path']
    steering_angle = row['steering_angle']

    # 이미지 로드
    if not os.path.exists(frame_path):
        print(f"파일이 존재하지 않습니다: {frame_path}")
        continue

    image = cv2.imread(frame_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {frame_path}")
        continue

    # 이미지 표시
    cv2.imshow("Frame", image)
    print(f"파일: {frame_path}, 조향각: {steering_angle:.2f}")
    print("D를 눌러 삭제하거나, 다른 키를 눌러 다음으로 넘어가세요.")

    key = cv2.waitKey(0)  # 키 입력 대기
    if key == ord('d') or key == ord('D'):  # 'D' 키를 누르면 삭제
        print(f"삭제: {frame_path}")
        os.remove(frame_path)  # 이미지 파일 삭제
    else:
        updated_data.append(row)  # 삭제하지 않을 경우 데이터를 유지

cv2.destroyAllWindows()

# CSV 파일 업데이트
updated_csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/filtered_training_data.csv"
updated_df = pd.DataFrame(updated_data)
updated_df.to_csv(updated_csv_file, index=False)

print(f"데이터 확인 및 삭제 완료. 새로운 CSV 파일 저장: {updated_csv_file}")
