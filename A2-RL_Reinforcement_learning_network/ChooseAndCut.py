from __future__ import absolute_import
###############choose photo#################
import requests
import base64
import json
import os
import cv2
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
    AIBeauty_url = "https://openapi.mtlab.meitu.com/v2/AsestheticsAssess?api_key=JDPaT3Q6vILO64zofYv9eWlw3KwJSJfQ&api_secret=yxtRRWxW1E4JnNiRxrDFsDNll273YcOq"
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

if __name__ == '__main__':
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
    #show how to cut
    orii = cv2.imread(path+"/"+finalpic)
    a=(xmin,ymin)
    b=(xmax,ymax)
    cv2.rectangle(orii, a, b, (0, 255, 0), 2)
    cv2.imwrite('/home/pi/Desktop/dataset/origin_cut.jpg',orii)
    print("congratulation!!!")

