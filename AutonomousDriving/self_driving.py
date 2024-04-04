# 08/18(목) 라인트래킹 소스코드

from buildhat import *
import picamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
import time

l_motor = Motor('B')
r_motor = Motor('A')


def motor_tank(a, b):
    a = a / 100
    b = b / 100
    l_motor.pwm(a)
    r_motor.pwm(b)


def motor_stop():
    l_motor.stop()
    r_motor.stop()


def make_black(image, threshold=200):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_image = cv2.inRange(gray_image, threshold, 255)
    return black_image, gray_image


def path_decision(image, limit=150):  # changed
    height, width = image.shape
    upbox = image[50:height - limit + 50, 64:256]
    image = image[height - limit:height - 70, :]
    mode = 'Black'

    height = limit - 1
    width = width - 1
    image = np.flipud(image)
    mask = image != 0

    upbox = np.flipud(upbox)
    mask2 = upbox != 0

    white_distance = np.where(mask.any(axis=0), mask.argmax(axis=0), height)
    mask2sum = mask2.sum() / (192 * (height - limit))
    if mask2sum > 96 * (height - limit):  # if darker than gray
        mode = 'black'
    else:
        mode = 'white'

    left = 0
    right = width
    block = 65

    center = int((left + right) / 2)
    left_sum = np.sum(white_distance[left:center - block])
    # left2_sum = np.sum(white_distance[block:block*2])
    forward_sum = np.sum(white_distance[center - block:center + block])
    # right2_sum = np.sum(white_distance[block*3:block*4])
    right_sum = np.sum(white_distance[center + block:right])

    print(left_sum, forward_sum, right_sum)

    if mode == 'black':
        if forward_sum > 18000:
            decision = 'f'
        elif left_sum > right_sum + 100:
            decision = 'sl'
        elif left_sum <= right_sum:
            decision = 'sr'

    elif mode == 'white':
        if forward_sum < 10000:
            decision = 'b'
        elif left_sum > right_sum:
            decision = 'bl'
        elif left_sum <= right_sum:
            decision = 'br'

    return decision, mode


def motor_control(decision):
    if decision == 'f':
        motor_tank(-18, 18)
    if decision == 'b':
        motor_tank(20, -20)
    if decision == 'sr':
        motor_tank(-15, 7)
    if decision == 'sl':
        motor_tank(7, 15)
    if decision == 'br':
        motor_tank(-22, 0)
    if decision == 'bl':
        motor_tank(0, 22)


camera = picamera.PiCamera()
camera.resolution = (320, 240)
camera.vflip = True
camera.hflip = True
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(320, 240))
decision = None
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    try:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        image = frame.array
        rawCapture.truncate(0)
        black_image, gray_image = make_black(image)

        decision, mode = path_decision(black_image)
        print(decision, "##mode is:", mode)
        motor_control(decision)
        cv2.rectangle(image, (0, 240 - 150), (320, 240 - 70), (0, 255, 0), 3)  # image roi
        cv2.rectangle(image, (64, 50), (256, (240 - 150 + 50)), (0, 0, 255), 3)  # upbox roi
        cv2.rectangle(image, (160 - 65, 0), (160 + 65, 250), (255, 0, 0), 3)  # 3 part

        cv2.putText(black_image, decision, (20, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255))
        cv2.putText(black_image, mode, (100, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 150, 200))
        cv2.putText(image, decision, (20, 120), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0))
        cv2.imshow("image", image)
        cv2.imshow("black", black_image)

    except Exception as e:
        print('except!###', e)
        break
