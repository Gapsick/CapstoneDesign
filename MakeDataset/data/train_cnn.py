import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

# SteeringDataset 클래스 정의
class SteeringDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # frame_path를 'cropped_frame_'으로 변경
        self.data['frame_path'] = self.data['frame_path'].str.replace("frame_", "cropped_frame_")

        # 유효한 경로만 필터링
        self.data = self.data[self.data['frame_path'].apply(os.path.exists)]
        if self.data.empty:
            raise ValueError("유효한 경로가 없습니다. CSV 파일과 실제 경로를 확인하세요.")

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

# PilotNet 모델 정의
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),  # 크기를 맞추기 위해 64*1*18로 수정
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 조향각 예측
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 데이터 경로 설정
csv_file_path = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/training_data_absolute_fixed.csv"

# 데이터셋 및 데이터로더
dataset = SteeringDataset(csv_file_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 초기화
model = PilotNet().to(device)
criterion = nn.SmoothL1Loss()  # 손실 함수
optimizer = Adam(model.parameters(), lr=0.001)

# 학습률 스케줄러
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# Warm-up 스케줄러 정의
class WarmUpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up_steps, last_epoch=-1):
        self.warm_up_steps = warm_up_steps
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_steps:
            return [base_lr * (self.last_epoch + 1) / self.warm_up_steps for base_lr in self.base_lrs]
        return self.base_lrs

# Warm-up 스케줄러
warm_up_scheduler = WarmUpScheduler(optimizer, warm_up_steps=5)

# 학습 루프
epochs = 500
losses = []
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, angles in dataloader:
        images, angles = images.to(device), angles.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)

    # 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "./best_pilotnet_model.pth")
        print(f"Epoch {epoch+1}: 새로운 Best 모델 저장 (Loss: {best_loss:.4f})")

    # 학습률 업데이트
    warm_up_scheduler.step()
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), "./final_pilotnet_model.pth")
print("모델 학습 완료 및 저장")

# 손실 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("training_loss_graph.png")
plt.show()
