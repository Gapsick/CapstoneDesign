import pandas as pd
import matplotlib.pyplot as plt
import os

# 현재 실행 파일의 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# CSV 파일 경로를 동적으로 설정
csv_path = os.path.join(current_dir, "data", "training_data_cleaned.csv")

try:
    # CSV 파일 로드
    df = pd.read_csv(csv_path)

    # 데이터 시각화를 위해 조향각 분포 확인
    plt.figure(figsize=(10, 6))
    plt.hist(df['steering_angle'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(True)

    # 그래프 보여주기
    plt.show()
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {csv_path}")
except Exception as e:
    print(f"오류 발생: {e}")
