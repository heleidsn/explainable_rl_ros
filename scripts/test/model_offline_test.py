'''
Date: 2020-11-17 18:49:33
LastEditors: Lei He
LastEditTime: 2020-11-17 19:04:51
FilePath: /explainable_rl_ros/scripts/test/model_offline_test.py
'''
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

import sys
sys.path.append('/home/helei/catkin_py3/src/explainable_rl_ros/scripts')
from scripts_final.td3 import TD3


def read_img_from_jpg():
    img = cv2.imread('scripts/test/depth_92.jpg', 0)

    return img

def get_obs():
    # load jpg as input

    image_obs = read_img_from_jpg()

    # get state features
    state_feature_array = np.zeros((80, 100))

    state_feature = np.zeros(6)

    state_feature_array[0, 0:6] = state_feature

    image_with_state = np.array([image_obs, state_feature_array])
    image_with_state = image_with_state.swapaxes(0, 2)
    image_with_state = image_with_state.swapaxes(0, 1)

    return image_with_state

def main():
    # get obs
    obs = get_obs()

    # load model
    model_path = '/home/helei/catkin_py3/src/explainable_rl_ros/scripts/models_real/model_1_1115_2d.zip'
    model = TD3.load(model_path)

    # predict
    action_real, _ = model.predict(obs)
    print(action_real)

if __name__ == "__main__":
    main()