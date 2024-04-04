#used at final contest 08/18 and 08/19
from buildhat import *
import picamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
import time
import argparse
from PIL import Image
import tflite_runtime.interpreter as tflite

l_motor = Motor('B')
r_motor = Motor('A')
Grab_Motor = Motor('D')
dist = DistanceSensor('C')

def handle_motor(speed, pos, apos):
    """Motor data
    :param speed: Speed of motor
    :param pos: Position of motor
    :param apos: Absolute position of motor
    """
    print("Motor", speed, pos, apos)
    
def motor_tank(a,b):
    a = a/100
    b = b/100
    l_motor.pwm(a)
    r_motor.pwm(b)

def motor_stop():
    l_motor.stop()
    r_motor.stop()

def make_black(image, threshold = 120):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_image=cv2.inRange(gray_image, threshold, 255)
    return black_image, gray_image

def path_decision(image, limit = 150):
    height, width = image.shape
    image = image[height-limit+20:height-18,:]
    height = limit -1
    width = width -1
    image = np.flipud(image)
    mask = image!=0

    white_distance = np.where(mask.any(axis=0), mask.argmax(axis=0), height)

    left=0
    right=width
    pop = 64 #change it!!
    
    center=int((left+right)/2)
    left_sum = np.sum(white_distance[left:center-pop])
    right_sum = np.sum(white_distance[center+pop:right])
    forward_sum = np.sum(white_distance[center-pop:center+pop])
    #print(left_sum, right_sum, forward_sum)

    if forward_sum > 13000 :
        decision = 'f'
        
    elif forward_sum < 200:
        decision = 'b'
        
    elif left_sum > right_sum :
        decision = 'l'
        

    elif left_sum <= right_sum :
        decision = 'r'
    
    else:
        decision = 'except'
        

    return decision

def motor_control(decision):
    if decision == 'except':
        motor_tank(16,-16)  #changed
    if decision == 'f':
        motor_tank(-28,28)
    if decision == 'r':
        motor_tank(-22,0)
    if decision == 'l':
        motor_tank(0,22)
    if decision == 'b':
        motor_tank(20,-20)    

camera = picamera.PiCamera()
camera.resolution = (320,240)
camera.vflip = True
camera.hflip = True
camera.framerate = 10
rawCapture = PiRGBArray(camera, size =(320,240))
decision = None
time.sleep(0.1)

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:,:] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='File path', required=False, default = './model.tflite')
    parser.add_argument('--labels', help='labels path', required=False, default = './labels.txt')
    args = parser.parse_args()
    labels = load_labels(args.labels)
    interpreter = tflite.Interpreter(model_path = args.model)
    interpreter.allocate_tensors()

    st2 = time.time()
    for frame in camera.capture_continuous(rawCapture, format='rgb', use_video_port=True):
        rawCapture.truncate(0)
        key = cv2.waitKey(1) & 0xFF
        image = frame.array
        cvtimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reimage = cv2.resize(cvtimage,(224,224), Image.ANTIALIAS)
        result = classify_image(interpreter,reimage)
        label_id, prob = result[0]
        print(labels[label_id], prob)
        cv2.putText(image, labels[label_id], (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        cv2.imshow('img',reimage)
        if (time.time()-st2) > 3:
            object_id = label_id
            break
        
    return object_id
        
    


Grab_Motor.run_to_position(0, 20)  #open arm
time.sleep(1)
st = time.time()
#step1: drive
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    image = frame.array
    rawCapture.truncate(0)
    black_image, gray_image = make_black(image)

    decision = path_decision(black_image)
    motor_control(decision)
    cv2.rectangle(image,(0,240-150),(320,240-10),(0,255,0),3) #roi
    cv2.rectangle(image,(80,0),(320-80,255),(255,0,0),3)  #3 part
    cv2.putText(black_image, decision, (20, 120), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255))
    cv2.putText(image, decision, (20, 120), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0))
    cv2.imshow("image", image)
    cv2.imshow("black",black_image)
    if (time.time()-st) > 26: #############
        motor_stop()
        break
    if (time.time()-st) > 12 : #############
    
####### step2: grap a bottle
while True:
    Grab_Motor.run_to_position(0, 20)  #open arm
    motor_tank(-15,15)  #keep going
    dist_mm=dist.get_distance()
    print(dist_mm)
    if dist_mm != -1 and dist_mm < 43 :
        Grab_Motor.run_for_degrees(120,30)  #grap
        motor_stop()
        break
    
######step3: watch what it is
object_id = main()
print('step3:',object_id)

####step4: move to position
if object_id == 0:  #bottle
    motor_tank(-20,-20)  #turn right
    time.sleep(1)
    motor_tank(-15,15) #move forward
    time.sleep(1)
    motor_stop()
    Grab_Motor.run_to_position(0, 20)  #open arm
    motor_tank(20,-20)  #back
    time.sleep(1)
    motor_stop()
    
    
elif object == 1:  #coffee
    motor_tank(20,20)  #turn left
    time.sleep(1)
    motor_tank(-15,15) #move forward
    time.sleep(1)
    motor_stop()
    Grab_Motor.run_to_position(0, 20)  #open arm
    motor_tank(20,-20)  #back
    time.sleep(1)
    motor_stop()
    