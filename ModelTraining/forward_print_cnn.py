import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # FC 레이어 입력 크기 계산을 위한 임시 텐서
        self.fc_input_size = self._get_fc_input_size()

        # FC 레이어 정의
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 조향각 예측
        )

    def _get_fc_input_size(self):
        # 임시 데이터를 사용하여 cnn 레이어의 출력 크기 계산
        with torch.no_grad():
            x = torch.randn(1, 3, 66, 200)  # (배치 크기, 채널 수, 높이, 너비)
            x = self.conv(x)
            return x.view(1, -1).size(1)  # fc 레이어로 넘기기 전에 평탄화된 크기

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
