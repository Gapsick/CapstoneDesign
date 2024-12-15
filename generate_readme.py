# generate_readme.py

# README 파일에 들어갈 내용
readme_content = """# Autonomous RC Car Project

## **프로젝트 개요**
본 프로젝트는 Jetson Nano를 활용하여 자율 주행 RC카를 개발하는 것입니다.  
PilotNet 모델을 사용하여 **End-to-End** 방식으로 경로를 추종하며 라인 트래킹을 수행합니다.

---

## **폴더 구조**
├── dataset_creation/ │ ├── 1_path_update.py │ ├── 2_check_missing_files.py │ ├── 3_remove_missing_files.py │ └── 4_angle_verification.py ├── dataset_preprocessing/ │ ├── 5_visualize_data_distribution.py │ ├── 6_oversampling.py │ ├── 6_1_combination.py │ └── 7_crop_and_augment.py ├── model_execution/ │ └── main_pilotnet_execution.py ├── best_pilotnet_model.pth ├── README.md └── generate_readme.py

yaml
코드 복사

---

## **파일 설명**

### **1. dataset_creation**
데이터셋 생성 및 정제를 위한 코드들입니다.  
- **1_path_update.py**: 이미지 경로 수정  
- **2_check_missing_files.py**: 누락된 파일 확인  
- **3_remove_missing_files.py**: 누락된 파일 제거  
- **4_angle_verification.py**: 잘못된 각도 데이터 확인 및 삭제  

### **2. dataset_preprocessing**
전처리 및 데이터 증강 코드입니다.  
- **5_visualize_data_distribution.py**: 데이터 분포 시각화  
- **6_oversampling.py**: 오버샘플링을 통해 데이터 균형 맞춤  
- **6_1_combination.py**: 데이터셋을 균등하게 조합  
- **7_crop_and_augment.py**: 이미지 자르기 및 데이터 증강 수행  

### **3. model_execution**
학습된 모델을 사용하여 RC카를 실행하는 코드입니다.  
- **main_pilotnet_execution.py**: 모델을 로드하고 RC카를 제어  

---

## **하드웨어 설정**
1. **Jetson Nano**: 메인 컴퓨팅 보드  
2. **DC 모터**: 후진 및 전진 제어  
3. **Servo 모터**: 방향 제어  
4. **카메라 모듈**: 실시간 영상 입력  

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
"""