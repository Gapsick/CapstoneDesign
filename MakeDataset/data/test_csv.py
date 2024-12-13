import torch
import csv
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np

# PyTorch 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 학습된 모델 클래스 정의 (PilotNet)
class PilotNet(torch.nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 36, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(36, 48, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 1 * 18, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Test 데이터셋 클래스 정의
class SteeringDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frame_path = row['frame_path']

        # 이미지 로드
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {frame_path}")

        # 이미지 전처리
        frame = frame / 255.0  # 픽셀 값을 [0, 1]로 정규화
        frame = cv2.resize(frame, (200, 66))  # 크기 조정
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW (채널 먼저)

        # 조향각 로드
        steering_angle = float(row['steering_angle'])

        return torch.tensor(frame, dtype=torch.float32), torch.tensor(steering_angle, dtype=torch.float32)

# CSV 파일 로드
csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/updated_training_data.csv"
data = pd.read_csv(csv_file)

# Test 데이터셋 로드
test_data = data.sample(frac=0.1, random_state=42)  # Test 데이터로 10% 샘플링
test_dataset = SteeringDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 학습된 모델 로드
model = PilotNet().to(device)
model_path = "best_pilotnet_model.pth"  # 학습된 모델 경로
model.load_state_dict(torch.load(model_path))
model.eval()

# Test 결과를 CSV 파일로 저장
output_csv_file = "test_predictions.csv"

actuals = []
predictions = []

with torch.no_grad():
    for images, angles in test_loader:
        images, angles = images.to(device), angles.to(device)
        outputs = model(images).squeeze()
        actuals.extend(angles.cpu().numpy())
        predictions.extend(outputs.cpu().numpy())

# CSV 저장
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Actual Steering Angle", "Predicted Steering Angle"])
    writer.writerows(zip(actuals, predictions))

print(f"Test 결과가 {output_csv_file}에 저장되었습니다.")
