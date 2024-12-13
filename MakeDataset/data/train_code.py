import os
import pandas as pd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# PyTorch 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 데이터셋 로드 및 분할
csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/updated_training_data.csv"
data = pd.read_csv(csv_file)

# 조향각 정규화
data['steering_angle'] = data['steering_angle'] / 30.0  # 최대값 기준 정규화

# 데이터셋 분할 (Train: 80%, Validation: 10%, Test: 10%)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

# SteeringDataset 클래스
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

# 모델 정의
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Dataset 및 DataLoader 생성
train_dataset = SteeringDataset(train_data)
val_dataset = SteeringDataset(val_data)
test_dataset = SteeringDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 초기화
model = PilotNet().to(device)
criterion = nn.SmoothL1Loss()  # 손실 함수 변경
optimizer = Adam(model.parameters(), lr=0.0001)  # 학습률 낮춤
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

# 학습 루프
epochs = 100  # 에포크를 100으로 설정
train_losses = []
val_losses = []
best_val_loss = float("inf")  # 초기 최저 Validation Loss 설정
best_model_path = "best_pilotnet_model.pth"  # 최적 모델 저장 경로

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for images, angles in train_loader:
        images, angles = images.to(device), angles.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, angles in val_loader:
            images, angles = images.to(device), angles.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), angles)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # 최적 Validation Loss 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch + 1}: Validation Loss 개선, 모델 저장 (Loss: {val_loss:.4f})")

    # 학습률 업데이트
    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Test 평가
model.load_state_dict(torch.load(best_model_path))  # 최적 모델 로드
model.eval()
total_absolute_error = 0
with torch.no_grad():
    for images, angles in test_loader:
        images, angles = images.to(device), angles.to(device)
        outputs = model(images).squeeze()
        total_absolute_error += torch.sum(torch.abs(outputs - angles)).item()

mae = total_absolute_error / len(test_dataset)  # Mean Absolute Error
print(f"Mean Absolute Error (Test): {mae:.2f}")

# 손실 그래프 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()
