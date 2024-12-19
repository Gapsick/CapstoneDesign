import Jetson.GPIO as GPIO
import cv2
import os
import csv
import time
import keyboard  # 키보드 입력 처리

# GPIO 설정
GPIO.setmode(GPIO.BOARD)

# 서보 모터 설정
servo_pin = 32  # 서보 모터 PWM 핀
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50Hz PWM
servo.start(7.5)  # 초기값 (90도)
current_angle = 90  # 서보 모터 초기 각도
angle_steps = [30, 60, 90, 120, 150]  # 허용 각도 범위

# DC 모터 설정
dir_pin = 29  # IN1
in2_pin = 31  # IN2
pwm_pin = 33  # ENA (속도 제어 핀)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(in2_pin, GPIO.OUT)
GPIO.setup(pwm_pin, GPIO.OUT)
dc_motor = GPIO.PWM(pwm_pin, 1000)  # 1kHz PWM
dc_motor.start(0)  # 초기 속도는 0

# 데이터 저장 경로 설정
base_path = "C:/Code/CapstoneDesign/CapstoneDesign/1213"
frame_save_path = os.path.join(base_path, "data/frames")
csv_file = os.path.join(base_path, "data/training_data.csv")
os.makedirs(frame_save_path, exist_ok=True)

# CSV 초기화
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frame_path', 'steering_angle'])  # 헤더 수정

# 서보 모터 각도 설정 함수
def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 180.0) * 10
    print(f"Setting servo angle to {angle}, Duty Cycle: {duty_cycle:.2f}")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.2)  # 짧은 딜레이
    servo.ChangeDutyCycle(0)  # PWM 끄기 (서보 보호)

# DC 모터 제어 함수
def control_dc_motor(direction, speed):
    if direction == "forward":  # 전진
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.HIGH)
    elif direction == "backward":  # 후진
        GPIO.output(dir_pin, GPIO.HIGH)
        GPIO.output(in2_pin, GPIO.LOW)
    elif direction == "stop":  # 정지
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.LOW)

    dc_motor.ChangeDutyCycle(speed)  # 속도 설정 (0~100)

# 카메라 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
collecting_data = False  # 데이터를 수집 중인지 여부
fps = 5  # 초당 저장할 프레임 수
frame_interval = 1 / fps  # 각 프레임 간의 시간 간격
last_saved_time = time.time()  # 마지막 저장 시간
current_direction = "stop"  # 현재 모터 방향

try:
    print("실시간 데이터 수집 및 모터 제어 시작")
    print("W: 전진, S: 후진, A: 좌회전, D: 우회전, Q: 종료, C: 데이터 수집 시작, X: 데이터 수집 중지")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("카메라 입력을 읽을 수 없습니다.")
            break

        current_time = time.time()

        # 데이터 수집
        if collecting_data and (current_time - last_saved_time >= frame_interval):
            frame_filename = f"frame_{frame_count:05d}.jpg"
            frame_path = os.path.join(frame_save_path, frame_filename)
            cv2.imwrite(frame_path, frame)

            # 현재 각도 기록 (direction 제거)
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_path, current_angle])

            print(f"Saved: {frame_path}, Steering Angle: {current_angle}")
            frame_count += 1
            last_saved_time = current_time

        # 화면 출력
        display_image = frame.copy()
        cv2.putText(display_image, f"Angle: {current_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", display_image)

        # 키보드 입력 처리
        if keyboard.is_pressed('a'):  # 좌회전
            current_index = angle_steps.index(current_angle)
            if current_index > 0:  # 최소값 이하로 내려가지 않도록 제한
                current_angle = angle_steps[current_index - 1]
                set_servo_angle(current_angle)

        elif keyboard.is_pressed('d'):  # 우회전
            current_index = angle_steps.index(current_angle)
            if current_index < len(angle_steps) - 1:  # 최대값 초과하지 않도록 제한
                current_angle = angle_steps[current_index + 1]
                set_servo_angle(current_angle)

        elif keyboard.is_pressed('w'):  # 전진
            current_direction = "forward"
            control_dc_motor("forward", 70)

        elif keyboard.is_pressed('s'):  # 후진
            current_direction = "backward"
            control_dc_motor("backward", 70)

        else:  # 정지
            current_direction = "stop"
            control_dc_motor("stop", 0)

        if keyboard.is_pressed('c'):  # 데이터 수집 시작
            collecting_data = True
            print("데이터 수집 시작")

        elif keyboard.is_pressed('x'):  # 데이터 수집 중지
            collecting_data = False
            print("데이터 수집 중지")

        elif keyboard.is_pressed('q'):  # 프로그램 종료
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("프로그램 종료")

finally:
    cap.release()
    servo.stop()
    dc_motor.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()

