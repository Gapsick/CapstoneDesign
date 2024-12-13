import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 로드
csv_file = "C:/Code/CapstoneDesign/CapstoneDesign/MakeDataset/data/training_data_absolute_fixed.csv"
data = pd.read_csv(csv_file)

# 조향각 분포 확인
def plot_steering_angle_distribution(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data['steering_angle'], bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Steering Angle Distribution')
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 분포 그리기
plot_steering_angle_distribution(data)
