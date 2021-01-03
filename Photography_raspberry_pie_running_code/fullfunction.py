from __future__ import absolute_import
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
# ============================================================================
# opencv
# ============================================================================
# ============================================================================
# Raspi PCA9685 16-Channel PWM Servo Driver
# ============================================================================
import socket
import time
import threading

#!/usr/bin/python
import time
import RPi.GPIO as GPIO
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import random
#!/usr/bin/python
import math
import smbus
# ============================================================================
# cut network
# ============================================================================

###############choose photo#################
import requests
import base64
import json
import os

def algorhtmReq(path,picc):
    image = open(path+"/"+picc, 'rb')
    #image = open(picc, 'rb')
    image_read = image.read()
    image_64_encode = base64.encodebytes(image_read).decode('utf-8')
    ss = json.dumps(
    {
        "parameter": {
             "version":"1.0.0",
        },
        "extra": {}, 
        "media_info_list": [{
        "media_data":image_64_encode,
        "media_profiles": {
            "media_data_type": "jpg"
        }
    }]
    }
    )
    AIBeauty_url = "https://openapi.mtlab.meitu.com/v2/AsestheticsAssess?api_key=ZGkfLGqVp4crHGkz1Y2v_Ygq197vWX0Y&api_secret=sejPIqnythJxheYgnh_QzS_vVbBLlT0p"
    print(ss)
    response = requests.post(AIBeauty_url, data=ss)
    ss2 = json.dumps(response.json())
    print(ss2)
    return(response.json()["media_info_list"][0]['media_extra']['AestheticsAssess'][0][5]["score"])
    print(response.status_code)
###############network##############
import pickle
import argparse
import numpy as np
import tensorflow as tf

import skimage.io as io
import network
from actions import command2action, generate_bbox, crop_input
#此函数只能在创建任何图、运算或张量之前调用。
#它可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头。
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

global_dtype = tf.float32

with open('/home/pi/Desktop/TF-A2RL/vfn_rl.pkl', 'rb') as f:
    var_dict = pickle.load(f)

image_placeholder = tf.compat.v1.placeholder(dtype=global_dtype, shape=[None,227,227,3])
global_feature_placeholder = network.vfn_rl(image_placeholder, var_dict)

h_placeholder = tf.compat.v1.placeholder(dtype=global_dtype, shape=[None,1024])
c_placeholder = tf.compat.v1.placeholder(dtype=global_dtype, shape=[None,1024])
action, h, c = network.vfn_rl(image_placeholder, var_dict, global_feature=global_feature_placeholder,
                                                           h=h_placeholder, c=c_placeholder)
sess = tf.compat.v1.Session()
def auto_cropping(origin_image):
    batch_size = len(origin_image)#原图片的数据量

    terminals = np.zeros(batch_size)
    ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)#比率;比例
    img = crop_input(origin_image, generate_bbox(origin_image, ratios))

    global_feature = sess.run(global_feature_placeholder, feed_dict={image_placeholder: img})
    h_np = np.zeros([batch_size, 1024])
    c_np = np.zeros([batch_size, 1024])

    while True:
        action_np, h_np, c_np = sess.run((action, h, c), feed_dict={image_placeholder: img,global_feature_placeholder: global_feature,
                                                                    h_placeholder: h_np,c_placeholder: c_np})
        ratios, terminals = command2action(action_np, ratios, terminals)
        bbox = generate_bbox(origin_image, ratios)
        if np.sum(terminals) == batch_size:
            return bbox
        
        img = crop_input(origin_image, bbox)
# ============================================================================
# connect to the server
# ============================================================================
class PCA9685:

  # Registers/etc.
  __SUBADR1            = 0x02
  __SUBADR2            = 0x03
  __SUBADR3            = 0x04
  __MODE1              = 0x00
  __MODE2              = 0x01
  __PRESCALE           = 0xFE
  __LED0_ON_L          = 0x06
  __LED0_ON_H          = 0x07
  __LED0_OFF_L         = 0x08
  __LED0_OFF_H         = 0x09
  __ALLLED_ON_L        = 0xFA
  __ALLLED_ON_H        = 0xFB
  __ALLLED_OFF_L       = 0xFC
  __ALLLED_OFF_H       = 0xFD


  def __init__(self, address=0x40, debug=False):
    self.bus = smbus.SMBus(1)
    self.address = address
    self.debug = debug
    if (self.debug):
      print("Reseting PCA9685")
    self.write(self.__MODE1, 0x00)

  def write(self, reg, value):
    #"Writes an 8-bit value to the specified register/address"
    self.bus.write_byte_data(self.address, reg, value)
    if (self.debug):
      print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))
 
  def read(self, reg):
    #"Read an unsigned byte from the I2C device"
    result = self.bus.read_byte_data(self.address, reg)
    if (self.debug):
      print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
    return result

  def setPWMFreq(self, freq):
    #"Sets the PWM frequency"
    prescaleval = 25000000.0    # 25MHz
    prescaleval /= 4096.0       # 12-bit
    prescaleval /= float(freq)
    prescaleval -= 1.0
    if (self.debug):
      print("Setting PWM frequency to %d Hz" % freq)
      print("Estimated pre-scale: %d" % prescaleval)
    prescale = math.floor(prescaleval + 0.5)
    if (self.debug):
      print("Final pre-scale: %d" % prescale)

    oldmode = self.read(self.__MODE1);
    newmode = (oldmode & 0x7F) | 0x10        # sleep
    self.write(self.__MODE1, newmode)        # go to sleep
    self.write(self.__PRESCALE, int(math.floor(prescale)))
    self.write(self.__MODE1, oldmode)
    time.sleep(0.005)
    self.write(self.__MODE1, oldmode | 0x80)
    self.write(self.__MODE2, 0x04)

  def setPWM(self, channel, on, off):
    #"Sets a single PWM channel"
    self.write(self.__LED0_ON_L+4*channel, on & 0xFF)
    self.write(self.__LED0_ON_H+4*channel, on >> 8)
    self.write(self.__LED0_OFF_L+4*channel, off & 0xFF)
    self.write(self.__LED0_OFF_H+4*channel, off >> 8)
    if (self.debug):
      print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel,on,off))
    
  def setServoPulse(self, channel, pulse):
    #"Sets the Servo Pulse,The PWM frequency must be 50HZ"
    pulse = pulse*4096/20000        #PWM frequency is 50HZ,the period is 20000us
    self.setPWM(channel, 0, int(pulse))
    
  def setRotationAngle(self, channel, Angle): 
    if(Angle >= 0 and Angle <= 180):
        temp = Angle * (2000 / 180) + 501
        self.setServoPulse(channel, temp)
    else:
        print("Angle out of range")
        
  def start_PCA9685(self):
    self.write(self.__MODE2, 0x04)
    #Just restore the stopped state that should be set for exit_PCA9685
    
  def exit_PCA9685(self):
    self.write(self.__MODE2, 0x00)#Please use initialization or __MODE2 =0x04

# ============================================================================
# 随即一个位置选择拍摄10个位置
# ============================================================================
#该函数用于定义开始框选时没有框选出人的身体的情况
def turntoBoxselectioln(found_filtered):
    if type(found_filtered) == tuple:
        return "turn"
    else :
        return "start work"
#该函数用于随机旋转舵机，暂时未使用
def randomposition(ino):
    ox,oy,ow,oh = ino
    level = random.randint(-9, 9) #定义舵机水平转动的角度
    height = random.randint(-9, 9) #定义舵机垂直转动的角度
    if decide(ino)== "i'm in proper position":
        pwm.setRotationAngle(1, level) #定义舵机水平转动
        pwm.setRotationAngle(0, height)#定义舵机垂直转动
    else :
        pass
    
#该函数用于判读框选框是否在规定的优选框之内        
def decide(ini):
    ix, iy, iw, ih = ini
    print(ix, iy, iw, ih)
    P_left=640/6 
    P_right=640/6*5
    P_bottom=480/6
    P_top=480/6*5
    if ix+iw>P_left and ix+iw<P_right and iy+ih>P_bottom and iy+ih<P_top:
        return "i'm in proper position"
    else :
        return "i'm not in proper position "
# cascPath = "D:/Anaconda5.2.0/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)

def is_size(inp):
    '''
    设定一个门限值当框选大小小于某一个阈值时排除该框
    '''
    ix, iy, iw, ih = inp
    if iw<150 or ih<150:
        return "pass"
    else:
        return "success"
def is_inside(o, i):
    '''
    判断矩形o是不是在i矩形中
    args:
        o：矩形o  (x,y,w,h)
        i：矩形i  (x,y,w,h)
    '''
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def draw_person(img, person):

    '''
    在img图像上绘制矩形框person
    args:
        img：图像img
        person：人所在的边框位置 (x,y,w,h)
    '''
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
# ============================================================================
# opencv获取视频流并进行定点拍照
# ============================================================================
start = 0 #定义为舵机是否开始工作 0代表未工作，初始时刻和发送时刻设置为0
end = 0#定义为一次拍摄的工作是否完成end=1代表已经完成了一次拍摄
position = 1#定义为当前在第几个位置拍照cascPath = "/home/pi/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascPath)
level_now= 90#the level angle of now
vertical_now = 2#the verticle angle of now
def contoraspberry():
# sock.send(b'1')
# print (sock.recv(1024).decode())
# sock.close()
    postion = 1
    msg = 0
    global start,end,position,level_now,vertical_now#调用全局变量
    HOST = '192.168.137.66'
    PORT = 6668
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    s.connect((HOST, PORT))
    while True :
        count = 0
        numberphoto =2#定义为一个定点拍摄图片的张数
        try:
           # print ("This is an PCA9685 routine")
            pwm = PCA9685()
            pwm.setPWMFreq(50)
            #pwm.setServoPulse(1,500) 
            pwm.setRotationAngle(1, 90) #水平方向将舵机水平旋转到0度角
            pwm.setRotationAngle(0, 2) #竖直方向将舵机水平旋转到0度角
        except:
            pwm.exit_PCA9685()
            print ("\nProgram end")
            exit()
        #print("call me your dad")
        if(start == 0 and end == 0):#start to work
            msg = s.recv(1024)    #接收数据（字节数）
            msg =  msg.decode('utf-8')   #解码
            if(msg=="1"):
                print('I received:', msg)
                start = 1   # start = 1为启动树莓派开始拍照的信号
                print(start)
            elif(msg=="3"):
                ###choose
                picl = []
                score = []
                path = '/home/pi/Desktop/dataset'
                path_list = os.listdir(path)
                for filename in path_list:
                    picl.append(filename)
                    score.append(algorhtmReq(path,filename))
                print("the final choose picture is:",picl[score.index(max(score))])
                finalpic = picl[score.index(max(score))]
                ###network
                im = io.imread(path+"/"+finalpic).astype(np.float32) / 255
                xmin, ymin, xmax, ymax = auto_cropping([im - 0.5])[0]
                final = (255*im[ymin:ymax, xmin:xmax]).astype(np.uint8)
                io.imsave('/home/pi/Desktop/dataset/final.jpg', final)
                orii = cv2.imread(path+"/"+finalpic)
                a=(xmin,ymin)
                b=(xmax,ymax)
                cv2.rectangle(orii, a, b, (0, 255, 0), 2)
                cv2.imwrite('/home/pi/Desktop/dataset/origin_cut.jpg',orii)
                print("congratulation!!!")
                Ishow = cv2.imread('/home/pi/Desktop/dataset/final.jpg')
                cv2.imshow("fianl",Ishow)
                cv2.waitKey(0)
        elif(end==0 and start==1):
            video_capture = cv2.VideoCapture(0)
            while count<numberphoto:
                if not video_capture.isOpened():
                    print('Unable to load camera.')
                    sleep(5)
                    pass
                    # 读视频帧
                ret, img = video_capture.read()
                print("I'm catching the picture")
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                (found,weight) = hog.detectMultiScale(img,winStride=(2,4),padding=(8,8),scale=1.2,useMeanshiftGrouping=False)
                #print('found', type(found), enumerate(found),found)
                if turntoBoxselectioln(found) == "turn":#此时还未检测到人，进行一次大范围的扰动
                    print("try to find destination")
                    level_now += 5
                    #vertical_now +=5
                    if(level_now>=145):
                        level_now = 35
                    pwm.setRotationAngle(1, level_now) #定义舵机水平转动
                    pwm.setRotationAngle(0, vertical_now)#定义舵机垂直转动
                else:
                    print("I find it!!!")
                    #video_capture.release()
                    #cv2.destroyAllwindows()
                    pos_idea_x = np.random.randint(640/6*2,640/6*5/2)
                    pos_idea_y = np.random.randint(480/6*2,480/6*5/2)
                    print(pos_idea_x,pos_idea_y)
                    turn_to_position(video_capture,pos_idea_x,pos_idea_y,pwm)#调动位置
                    #拍摄图片
                    #cap=video_capture
                    ret,catch = video_capture.read()
                    count +=1
                    print("dataset/User." + str(position) + '.' + str(count) + ".jpg")
                    cv2.imwrite("dataset/User." + str(position) + '.' + str(count) + ".jpg",catch)
                    #cv2.imshow("capture",catch)
                    #cap.release()
                    #cv2.destroyAllwindows()
            video_capture.release()
            position += 1#下一个拍照的位置地点
            end = 1#代表1次拍照结束
            start = 0
            print('start:',start,'\n')
            print('end:',end,'\n')
        # setRotationAngle(0/1, i)    0是竖直转动舵机 1是水平转动舵机
        elif(end==1 and start==0):
            print("i finished the work")
            s.send("2".encode('utf-8'))#go to next position
            end = 0
            #continue
def turn_to_position(video_capture,x_p,y_p,pwm):
    global level_now,vertical_now
    #函数功能：把摄像头捕捉得到的框图的中心点定位到x_P,y_p
    #舵机角度初始化
    #pwm.setRotationAngle(1, 90) #水平方向将舵机水平旋转到0度角
    #pwm.setRotationAngle(0, 20) #竖直方向将舵机垂直旋转到0度角
    #增量式PID算法的设计
    #水平舵机转动，设置如下4个变量
    need_level_angle = 0 # 定位为水平需要转动的小角度
    thisError_x=0 #定义为此次水平方向的误差
    lastError_x=0 #定义为上次水平方向的误差
    #垂直舵机转动，设置如下4个变量
    need_vertical_angle = 0 # 定位为水平需要转动的小角度
    thisError_y=0 #定义为此次水平方向的误差
    lastError_y=0 #定义为上次水平方向的误差
    #video_capture = cv2.VideoCapture(0)
    nn_max = 0#the max time to turn
    while nn_max<=25:
        # 打开视频捕获设备
        #video_capture = cv2.VideoCapture(0)
        '''
        if not video_capture.isOpened():
            print('Unable to load camera.try again')
            sleep(5)
            break
        else:
        # 读视频帧
    '''
        ret, img = video_capture.read()
        #print('work')
        #video_capture.release()
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        (found,weight) = hog.detectMultiScale(img,winStride=(2,4),padding=(8,8),scale=1.2,useMeanshiftGrouping=False)
        #video_capture.release()
        for (x,y,w,h) in found:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.resizeWindow('Video', 640,480 )
            #cv2.imshow("hog-detector",img)
        #定义found_filtered为检测到的矩形框，若识别到的有重合，则删去
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                # r在q内？
                if ri != qi and is_inside(r, q):
                    break
                else:
                    found_filtered.append(r)
        #这里暂时认为只识别出一个框图(此处由于开始时摄像头调为中心位置，故一般情况下会检测到人)
        #print(found_filtered)
        if found_filtered==[]:
            print("try again")
            kkk = random.randint(-8,8)
            level_now += kkk #定义舵机水平转动的角度
            mmm = random.randint(-2,2)
            vertical_now = mmm #定义舵机垂直转动的角度
            pwm.setRotationAngle(1, level_now) #定义舵机水平转动
            print("turn angle of level(+ mean left -mean right):",kkk)
            pwm.setRotationAngle(0, vertical_now)#定义舵机垂直转动
            print("turn angle of vertical(+ mean down -mean up)",mmm)
            nn_max+=1
        else:
            n_x,n_y,n_w,n_h = found_filtered[0]
            c_x = n_x+n_w/2  #center of x position
            c_y = n_y+n_h/2  #center of y position
            thisError_x = x_p-c_x
            thisError_y = c_y-y_p#
            #设置需要调整舵机的环节
            if abs(thisError_x)<2 and abs(thisError_y)<2:
                break
            else:
                #设置P值为0.001，D值为0.001（P，D为系数）
                need_level_angle = thisError_x*0.025+0.013*(thisError_x-lastError_x)
                need_vertical_angle = thisError_y*0.024+0.012*(thisError_y-lastError_y)
                #更新上次的水平、垂直error变量
                lastError_x = thisError_x
                lastError_y = thisError_y
                #定义水平、垂直转动角度
                level_now += need_level_angle 
                vertical_now += need_vertical_angle 
            #设置舵机水平转动的阈值为（45，135）
                if level_now>135:
                    level_now=135
                elif level_now<45:
                    level_now=45
            #设置舵机垂直转动的阈值为（10，30）
                if vertical_now>20:
                    vertical_now=20
                elif vertical_now<1:
                    vertical_now=1
                pwm.setRotationAngle(1, level_now) #水平方向将舵机水平旋转到0度角
                print("turn angle of level(+ mean left -mean right):",need_level_angle)
                pwm.setRotationAngle(0, vertical_now) #竖直方向将舵机垂直旋转到0度角
                print("turn angle of vertical(+ mean down -mean up)",need_vertical_angle)
                nn_max+=1
    print("the total times to turn:",nn_max)
             #video_capture.release()
#s.close


def themainwork():
    while True:
        global start,end,position,level_now,vertical_now#调用全局变量
        count = 0
        numberphoto =1#定义为一个定点拍摄图片的张数
        try:
            #print ("This is an PCA9685 routine")
            pwm = PCA9685()
            pwm.setPWMFreq(50)
            #pwm.setServoPulse(1,500) 
            pwm.setRotationAngle(1, 90) #水平方向将舵机水平旋转到0度角
            pwm.setRotationAngle(0, 20) #竖直方向将舵机水平旋转到0度角
            # setRotationAngle(0/1, i)    0是竖直转动舵机 1是水平转动舵机
        except:
            pwm.exit_PCA9685()
            print ("\nProgram end")
            exit()
        if start == 1:
            video_capture = cv2.VideoCapture(0)
            while count<numberphoto:
                if not video_capture.isOpened():
                    print('Unable to load camera.')
                    sleep(5)
                    pass
                    # 读视频帧
                ret, img = video_capture.read()
                print(img)
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                (found,weight) = hog.detectMultiScale(img,winStride=(2,4),padding=(8,8),scale=1.2,useMeanshiftGrouping=False)
                #video_capture.release()
                #print('found', type(found), enumerate(found),found)
                if turntoBoxselectioln(found) == "turn":#此时还未检测到人，进行一次大范围的
                    print("try to find destination")
                    level_now = random.randint(60,120) #定义舵机水平转动的角度
                    vertical_now = random.randint(10,30) #定义舵机垂直转动的角度
                    pwm.setRotationAngle(1, level_now) #定义舵机水平转动
                    pwm.setRotationAngle(0, vertical_now)#定义舵机垂直转动
                else:
                    #video_capture.release()
                    #cv2.destroyAllwindows()
                    print("I have found the destination")
                    pos_idea_x = np.random.randint(640/6*2,640/6*5/2)
                    pos_idea_y = np.random.randint(480/6*2,480/6*5/2)
                    print(pos_idea_x,pos_idea_y)
                    turn_to_position(video_capture,pos_idea_x,pos_idea_y,pwm)#调动位置
                    #拍摄图片
                    #cap=video_capture
                    ret,catch = video_capture.read()
                    count +=1
                    print("dataset/User." + str(position) + '.' + str(count) + ".jpg")
                    cv2.imwrite("dataset/User." + str(position) + '.' + str(count) + ".jpg",catch)
                    #cv2.imshow("capture",catch)
                    #cap.release()
                    #cv2.destroyAllwindows()
            video_capture.release()
            position += 1#下一个拍照的位置地点
            print('end:',end,'\n')
            Lock.acquire()
            end = 1#代表1次拍照结束
            #delete --just experience
            print('end:',end,'\n')
            start = 0
            Lock.release()
            print('start:',start,'\n')
        else:
            pass
if __name__=='__main__':
    contoraspberry()