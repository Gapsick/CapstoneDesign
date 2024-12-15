# generate_readme.py

# README 파일에 들어갈 내용
readme_content = """# 🚗 Autonomous RC Car Project

## **프로젝트 개요**
본 프로젝트는 Jetson Nano를 활용하여 라인 트래킹 기반 자율주행 RC카를 개발하는 것을 목표로 합니다.  
데이터 수집, 데이터 전처리, 모델 학습, 그리고 실시간 실행까지의 전체 파이프라인을 포함합니다.

---

## **폴더 구조**
├── dataset_creation/ # 데이터셋 생성 및 정제 코드 │ ├── 1_path_update.py │ ├── 2_check_missing_files.py │ ├── 3_remove_missing_files.py │ └── 4_angle_verification.py ├── dataset_preprocessing/ # 전처리 및 데이터 증강 코드 │ ├── 5_visualize_data_distribution.py │ ├── 6_oversampling.py │ ├── 6_1_combination.py │ └── 7_crop_and_augment.py ├── model_execution/ # 모델 실행 코드 │ └── main_pilotnet_execution.py ├── best_pilotnet_model.pth # 최종 학습된 모델 ├── README.md # 프로젝트 설명 파일 └── generate_readme.py # README 자동 생성 코드

yaml
코드 복사

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
2. 모델 실행
bash
코드 복사
python3 model_execution/main_pilotnet_execution.py
"""