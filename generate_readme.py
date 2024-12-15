# generate_readme.py

# README 파일에 들어갈 내용
readme_content = """# Autonomous RC Car Project

## **프로젝트 개요**
본 프로젝트는 Jetson Nano를 활용하여 자율 주행 RC카를 개발하는 것입니다.  
PilotNet 모델을 사용하여 **End-to-End** 방식으로 경로를 추종하며 라인 트래킹을 수행합니다.

---

## **폴더 구조**
- **dataset_creation**: 데이터셋 생성 코드  
- **dataset_preprocessing**: 데이터 전처리 및 증강 코드  
- **model_execution**: 학습된 모델을 실행하는 코드  

---

## **파일 설명**

### **1. dataset_creation**
- **1_path_update.py**: 이미지 경로 수정  
- **2_check_missing_files.py**: 누락된 파일 확인  
- **3_remove_missing_files.py**: 누락된 파일 제거  
- **4_angle_verification.py**: 잘못된 각도 데이터 확인 및 삭제  

### **2. dataset_preprocessing**
- **5_visualize_data_distribution.py**: 데이터 분포 시각화  
- **6_oversampling.py**: 데이터 오버샘플링  
- **6_1_combination.py**: 균등한 데이터셋 생성  
- **7_crop_and_augment.py**: 이미지 크롭 및 증강  

### **3. model_execution**
- **main_pilotnet_execution.py**: 학습된 모델을 로드하여 RC카 제어  

---

## **하드웨어 설정**
1. **Jetson Nano**  
2. **DC 모터**  
3. **서보 모터**  
4. **카메라 모듈**  

---

## **소프트웨어 스택**
- **Python 3.10**  
- **PyTorch**  
- **OpenCV**  
- **Jetson.GPIO**  

---

## **실행 방법**
### **1. 데이터셋 생성 및 전처리**
```bash
python3 dataset_creation/1_path_update.py
python3 dataset_preprocessing/5_visualize_data_distribution.py
2. 모델 실행
bash
코드 복사
python3 model_execution/main_pilotnet_execution.py
"""


with open("README.md", "w", encoding="utf-8") as file:
    file.write(readme_content) 
print("README.md 파일이 성공적으로 생성되었습니다.")
