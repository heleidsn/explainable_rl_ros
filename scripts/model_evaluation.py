#!/usr/bin/env python3
'''
Author: Lei He
Date: 2020-08-29 14:51:26
LastEditTime: 2020-08-31 23:20:36
LastEditors: Please set LastEditors
Description: model evaluation in ros environment
FilePath: /explainable_rl_ros/scripts/model_evaluation.py
'''
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import sys
import cv2
import rospy
from mavros_msgs.msg import State
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import math

from configparser import ConfigParser

from scripts_final.td3 import TD3
# from stable_baselines import TD3


class ModelEvalNode():
    def __init__(self):
        
        self.bridge = CvBridge()

        self.cfg = ConfigParser()
        self.cfg.read('/home/helei/catkin_ws_rl/src/explainable_rl_ros/configs/config.ini')
        self.set_config(self.cfg)

        self.set_sub_pub()
    
        # set goal pose
        self._set_goal_pose(20, 0, 5)

        self._depth_image_meter = np.zeros((self.image_height, self.image_width))
        self._depth_image_gray = np.zeros((self.image_height, self.image_width))

        # load model
        # print(sys.path)
        self.model = TD3.load('/home/helei/catkin_ws_rl/src/explainable_rl_ros/scripts/models/test_model')
        print('model load success')

        self._check_all_systems_ready()
       
        while not rospy.is_shutdown():
            self.check_sensor_data()
            obs = self.get_obs()
            action_real, _ = self.model.predict(obs)
            self.set_action(action_real)
            
            if self.control_rate:
                rospy.sleep(1 / self.control_rate)
            else:
                rospy.sleep(1.0)

    def set_config(self, cfg):
        self.control_rate = cfg.getint('gazebo', 'control_rate')

        self.goal_distance = 0
        self.max_depth_meter = cfg.getfloat('gazebo', 'max_depth_meter')

        self.image_height = cfg.getint('gazebo', 'image_height')
        self.image_width = cfg.getint('gazebo', 'image_width')

        self.state_feature_length = cfg.getint('gazebo', 'state_feature_length')

        self.max_vertical_difference = cfg.getfloat('control', 'max_vertical_difference')

        # uav_model
        self.acc_lim_x = cfg.getfloat('uav_model', 'acc_lim_x')
        self.acc_lim_z = cfg.getfloat('uav_model', 'acc_lim_z')
        self.acc_lim_yaw_deg = cfg.getfloat('uav_model', 'acc_lim_yaw_deg')
        self.acc_lim_yaw_rad = math.radians(self.acc_lim_yaw_deg)

        self.max_vel_x = cfg.getfloat('uav_model', 'max_vel_x')
        self.min_vel_x = cfg.getfloat('uav_model', 'min_vel_x')
        self.max_vel_z = cfg.getfloat('uav_model', 'max_vel_z')
        self.max_vel_yaw_rad = math.radians(cfg.getfloat('uav_model', 'max_vel_yaw_deg'))
        

    def set_sub_pub(self):
        # --------------Subscribers-------------------
        # state
        self._mavros_state = State()
        rospy.Subscriber('mavros/state', State, callback=self._stateCb, queue_size=10)
        # depth image
        rospy.Subscriber('/camera/depth/image_raw', Image, callback=self._imageCb, queue_size=10)
        # local odometry
        self._local_odometry = Odometry()
        rospy.Subscriber('/mavros/local_position/odom', Odometry, callback=self._local_odomCb, queue_size=10)

        # --------------Publishers--------------------
        # current goal position 
        self._goal_pose = PoseStamped()
        self._goal_pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # control command
        self._local_pose_setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local',PoseStamped, queue_size=10)
        self._setpoint_marker_pub = rospy.Publisher('network/setpoint', Marker, queue_size=10)


# call back functions
    def _stateCb(self, msg):
        self._mavros_state = msg

    def _poseCb(self, msg):
        self._current_local_pose = msg

    def _imageCb(self, msg):
        depth_image_msg = msg
        try:
            # tranfer image from ros msg to opencv image encode F32C1
            cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)

        (rows,cols) = cv_image.shape
        image = np.array(cv_image, dtype=np.float32)
        self._depth_image_meter = np.copy(image)
        # deal with nan
        image[np.isnan(image)] = self.max_depth_meter
        # transfer to uint8
        image_gray = image / self.max_depth_meter * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = np.copy(image_gray_int)

    def _local_odomCb(self, msg):
        self._local_odometry = msg

# check system 
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers, services and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        # self._check_all_publishers_ready()
        return True

    def _check_all_sensors_ready(self):
        rospy.logdebug("CHECK ALL SENSORS CONNECTION:")
        self._check_depth_image_ready()
        self._check_local_odometry_ready()
        self._check_local_odometry_ready()
        rospy.logdebug("All Sensors CONNECTED and READY!")

    def _check_depth_image_ready(self):
        self._current_pose = None
        rospy.logdebug("Waiting for /camera/depth/image_raw to be READY...")
        while self._depth_image_meter is None and not rospy.is_shutdown():
            try:
                self._depth_image_meter = rospy.wait_for_message("/camera/depth/image_raw", PoseStamped, timeout=5.0)
                rospy.logdebug("Current /camera/depth/image_raw READY=>")
            except:
                rospy.logdebug("Current /camera/depth/image_raw not ready, retrying for getting depth image")
        return self._depth_image_meter

    def _check_local_odometry_ready(self):
        self._local_odometry = None
        rospy.logdebug("Waiting for /mavros/local_position/odom to be READY...")
        while self._local_odometry is None and not rospy.is_shutdown():
            try:
                self._local_odometry = rospy.wait_for_message("/mavros/local_position/odom", PoseStamped, timeout=5.0)
                rospy.logdebug("Current mavros/local_position/pose READY=>")
            except:
                rospy.logdebug("Current /mavros/local_position/odom not ready, retrying for getting local odom")
        return self._local_odometry

    
# functions

    def get_obs(self):
        image = self._depth_image_gray.copy() # Note: check image format. Now is 0-black near 255-wight far

        # transfer image to image obs according to 0-far  255-nears
        image_obs = 255 - image

        state_feature_array = np.zeros((self.image_height, self.image_width))

        state_feature = self._get_state_feature(self._local_odometry, self._goal_pose)
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        image_with_state = np.array([image_obs, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

    def check_sensor_data(self):
        pass

    def set_action(self, action):
        '''
        generate control command from action
        action_real: forward speed, climb speed, yaw speed
        '''
        # get yaw and yaw setpoint 
        current_yaw = self.get_current_yaw()
        yaw_speed = action[2]
        yaw_setpoint = current_yaw + yaw_speed

        # transfer dx dy from body frame to local frame
        dx_body = action[0]
        dy_body = 0
        dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

        pose_setpoint = PoseStamped()
        self._current_pose = self._local_odometry.pose
        pose_setpoint.pose.position.x = self._current_pose.pose.position.x + dx_local
        pose_setpoint.pose.position.y = self._current_pose.pose.position.y + dy_local
        pose_setpoint.pose.position.z = self._current_pose.pose.position.z + action[1]

        orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
        pose_setpoint.pose.orientation.x = orientation_setpoint[0]
        pose_setpoint.pose.orientation.y = orientation_setpoint[1]
        pose_setpoint.pose.orientation.z = orientation_setpoint[2]
        pose_setpoint.pose.orientation.w = orientation_setpoint[3]
        
        marker_network_setpoint = Marker()
        marker_network_setpoint.header.stamp = rospy.Time.now()
        marker_network_setpoint.header.frame_id = 'local_origin'
        marker_network_setpoint.type = Marker.SPHERE
        marker_network_setpoint.action = Marker.ADD
        marker_network_setpoint.pose.position = pose_setpoint.pose.position
        marker_network_setpoint.pose.orientation.x = 0.0
        marker_network_setpoint.pose.orientation.y = 0.0
        marker_network_setpoint.pose.orientation.z = 0.0
        marker_network_setpoint.pose.orientation.w = 1.0
        marker_network_setpoint.scale.x = 0.3
        marker_network_setpoint.scale.y = 0.3
        marker_network_setpoint.scale.z = 0.3
        marker_network_setpoint.color.a = 0.8
        marker_network_setpoint.color.r = 0.0
        marker_network_setpoint.color.g = 0.0
        marker_network_setpoint.color.b = 0.0

        self._local_pose_setpoint_pub.publish(pose_setpoint)
        self._setpoint_marker_pub.publish(marker_network_setpoint)
    

# utils

    def _get_state_feature(self, current_odom, goal_pose):
        '''
        Airsim pose use NED SYSTEM
        Gazebo pose z-axis up is positive different from NED
        Gazebo twist using body frame
        '''
        # get distance and angle in polar coordinate
        # transfer to 0~255 image formate for cnn
        current_pose = current_odom.pose
        relative_pose_x = goal_pose.pose.position.x - current_pose.pose.position.x
        relative_pose_y = goal_pose.pose.position.y - current_pose.pose.position.y
        relative_pose_z = goal_pose.pose.position.z - current_pose.pose.position.z
        distance = math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2))
        relative_yaw = self._get_relative_yaw(current_pose, goal_pose)

        distance_norm = distance / self.goal_distance * 255
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255
        
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255

        # current speed and angular speed
        current_vel_local = current_odom.twist.twist
        linear_velocity_xy = current_vel_local.linear.x  # forward velocity
        linear_velocity_norm = linear_velocity_xy / self.max_vel_x * 255
        linear_velocity_z = current_vel_local.linear.z  #  vertical velocity
        linear_velocity_z_norm = (linear_velocity_z / self.max_vel_z / 2 + 0.5) * 255
        angular_velocity_norm = (-current_vel_local.angular.z / self.max_vel_yaw_rad / 2 + 0.5) * 255  # TODO: check the sign of the 

        self.state_raw = np.array([distance, relative_pose_z, relative_yaw, linear_velocity_xy, linear_velocity_z, -current_vel_local.angular.z])
        state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm, linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
        state_norm = np.clip(state_norm, 0, 255)
        self.state_norm = state_norm

        return state_norm

    def _get_relative_yaw(self, current_pose, goal_pose):
        # get relative angle
        relative_pose_x = goal_pose.pose.position.x - current_pose.pose.position.x
        relative_pose_y = goal_pose.pose.position.y - current_pose.pose.position.y
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        explicit_quat = [current_pose.pose.orientation.x, current_pose.pose.orientation.y, \
                                current_pose.pose.orientation.z, current_pose.pose.orientation.w]
        
        yaw_current = euler_from_quaternion(explicit_quat)[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def _set_goal_pose(self, x, y, z):
        self._goal_pose.pose.position.x = x
        self._goal_pose.pose.position.y = y
        self._goal_pose.pose.position.z = z

        self.goal_distance = math.sqrt(pow(x, 2) + pow(y, 2))

        self._goal_pose_pub.publish(self._goal_pose)

    def get_current_yaw(self):
        orientation = self._local_odometry.pose.pose.orientation

        current_orientation = [orientation.x, orientation.y, \
                                orientation.z, orientation.w]
            
        current_attitude = euler_from_quaternion(current_orientation)
        current_yaw = euler_from_quaternion(current_orientation)[2]

        return current_yaw

    def point_transfer(self, x, y, theta):
        # transfer x, y to another frame
        x1 = x * math.cos(theta) + y * math.sin(theta)
        x2 = - x * math.sin(theta) + y * math.cos(theta)

        return x1, x2

if __name__ == "__main__":
    try:
        print('init model eval node')
        rospy.init_node('model_eval', anonymous=True)
        node_me = ModelEvalNode()
    except rospy.ROSInterruptException:
        print('ros node model-evaluation terminated...')