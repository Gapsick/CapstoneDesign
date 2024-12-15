import os

# README 파일에 들어갈 내용
readme_content = """# 🚗 Autonomous RC Car Project

## **프로젝트 개요**
이 프로젝트는 **Jetson Nano**를 활용하여 **라인 트래킹 기반 자율주행 RC카**를 개발하는 것을 목표로 합니다.  
데이터 수집, 데이터 전처리, 모델 학습, 그리고 실시간 실행까지의 전체 파이프라인을 포함합니다.

---

## **폴더 구조**
{folder_structure}

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
2. 모델 실행
bash
코드 복사
python3 model_execution/main_pilotnet_execution.py
"""

폴더 구조 생성 함수
def get_folder_structure(path, indent=0): result = "" for item in os.listdir(path): item_path = os.path.join(path, item) if os.path.isdir(item_path): result += " " * indent + f"📁 {item}\n" result += get_folder_structure(item_path, indent + 1) else: result += " " * indent + f"📄 {item}\n" return result

폴더 구조를 readme_content에 삽입
project_root = "." # 현재 디렉토리를 기준으로 함 folder_structure = get_folder_structure(project_root) readme_content = readme_content.format(folder_structure=folder_structure)

README.md 파일에 쓰기
with open("README.md", "w", encoding="utf-8") as file: file.write(readme_content)

print("README.md 파일이 성공적으로 생성되었습니다!")

yaml
코드 복사

---

### **코드 설명**
1. **폴더 구조 자동 생성**  
   `get_folder_structure()` 함수를 사용하여 현재 디렉토리 및 하위 폴더의 구조를 자동으로 탐색하고, **README.md**에 입력합니다.

2. **README 파일 생성**  
   `README.md` 파일이 새롭게 생성되거나 덮어쓰여집니다.

---

### **실행 방법**
1. 프로젝트 폴더의 **최상위 디렉토리**에서 위의 코드를 실행하세요.
   ```bash
   python3 generate_readme.py