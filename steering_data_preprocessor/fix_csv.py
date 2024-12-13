import pandas as pd

# CSV 파일 경로
csv_path = "C:/Code/CapstoneDesign/CapstoneDesign/steering_data_preprocessor/data/training_data.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 'direction' 열 제거
if 'direction' in df.columns:
    df = df.drop(columns=['direction'])
    print("'direction' 열이 제거되었습니다.")

# 수정된 CSV 파일 저장
df.to_csv(csv_path, index=False)
print(f"수정된 CSV 파일이 저장되었습니다: {csv_path}")
