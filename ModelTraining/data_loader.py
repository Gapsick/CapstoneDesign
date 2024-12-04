import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 데이터셋 정의
class SteeringDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frame = cv2.imread(row['frame_path']) / 255.0  # 픽셀 값을 [0, 1]로 정규화
        frame = cv2.resize(frame, (200, 66))  # 크기 조정
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        steering_angle = float(row['steering_angle'])
        
        # 텐서를 GPU로 전송
        return torch.tensor(frame, dtype=torch.float32).to(device), torch.tensor(steering_angle, dtype=torch.float32).to(device)

# 테스트 코드
if __name__ == "__main__":
    dataset = SteeringDataset("C:/Code/CapstoneDesign/CapstoneDesign/ModelTraining/data/processed_training_data.csv")   # CSV 경로 지정
    print(f"데이터셋 크기: {len(dataset)}")  # 데이터셋 크기 출력

    # 첫 번째 데이터 확인
    frame, steering_angle = dataset[0]
    print(f"첫 번째 프레임 크기: {frame.shape}")  # (3, 66, 200) 확인
    print(f"첫 번째 조향각: {steering_angle}")
