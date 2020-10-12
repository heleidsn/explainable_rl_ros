#!/usr/bin/env python3
'''
Author: Lei He
Date: 2020-08-29 14:51:26
LastEditTime: 2020-10-12 22:56:38
LastEditors: Please set LastEditors
Description: ROS node pack for model evaluation in ros environment
FilePath: /explainable_rl_ros/scripts/model_evaluation.py
'''
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import sys
import cv2
import rospy
from mavros_msgs.msg import State, PositionTarget
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker
from explainable_rl_ros.msg import vel_cmd

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import math

from configparser import ConfigParser

from scripts_final.td3 import TD3
# from stable_baselines import TD3


class ModelEvalNode():
    def __init__(self):
        rospy.logdebug('Start init ModelEvalNode...')
        
        self.bridge = CvBridge()
        self.cfg = ConfigParser()

        # settings
        self.goal_height = 5
        config_path = '/home/helei/catkin_py3/src/explainable_rl_ros/configs/config.ini'
        model_path = '/home/helei/catkin_py3/src/explainable_rl_ros/scripts/models/2020_10_06_00_07_same with i5_200000.zip'
        self.image_source = 'gazebo'   # choose image source. gazebo or realsense
        
        self.cfg.read(config_path)
        self.set_config(self.cfg)

        self.set_sub_pub()
    
        # state variables
        self._mavros_state = State()
        self._goal_pose = PoseStamped()

        self.pose_local = PoseStamped()
        self.vel_local = TwistStamped()

        self.state_feature_raw = np.zeros(6)
        self.state_feature_norm = np.zeros(6)

        self._depth_image_meter = None
        self._depth_image_gray = None

        

        self.action_last = np.array([0, 0, 0])
        self.filter_alpha = 0.5

        # load model
        self.model = TD3.load(model_path)
        rospy.logdebug('model load success')

        self._check_all_systems_ready()

        # set goal pose
        self._set_goal_pose(20, 0, self.goal_height)
       
        rospy.logdebug('neural network controller start...')
        while not rospy.is_shutdown():
            # self.check_sensor_data()
            obs = self.get_obs()
            action_real, _ = self.model.predict(obs)
            rospy.logdebug('state_raw: ' + np.array2string(self.state_feature_raw, formatter={'float_kind':lambda x: "%.2f" % x}))
            rospy.logdebug('state_norm: ' + np.array2string(self.state_feature_norm, formatter={'float_kind':lambda x: "%.2f" % x}))
            rospy.logdebug('action real: ' + np.array2string(action_real, formatter={'float_kind':lambda x: "%.2f" % x}))
            # self.set_action_pose(action_real)
            self.set_action_vel(action_real)
            
            if self.control_rate:
                rospy.sleep(1 / self.control_rate)
            else:
                rospy.sleep(1.0)

    def set_config(self, cfg):
        self.control_rate = cfg.getint('gazebo', 'control_rate')
        rospy.logdebug('control_rate: {:d}'.format(self.control_rate))

        self.goal_distance = None
        self.max_depth_meter = cfg.getfloat('gazebo', 'max_depth_meter')
        self.max_depth_meter_gazebo = cfg.getfloat('gazebo', 'max_depth_meter_gazebo')

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
        # --------------Publishers--------------------
        # current goal position
        self._goal_pose_pub = rospy.Publisher('/network/goal', PoseStamped, queue_size=10)
        self._goal_pose_marker_pub = rospy.Publisher('network/marker_goal', Marker, queue_size=10)

        # control command
        self._local_pose_setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local',PoseStamped, queue_size=10)
        self._setpoint_marker_pub = rospy.Publisher('network/marker_pose_setpoint', Marker, queue_size=10)
        self._setpoint_raw_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        
        # image
        self._depth_image_gray_input = rospy.Publisher('network/depth_image_input', Image, queue_size=10)

        # debug info
        self._action_msg_pub = rospy.Publisher('/network/debug/action', vel_cmd, queue_size=10)
        self._state_vel_msg_pub = rospy.Publisher('/network/debug/state', vel_cmd, queue_size=10)
        
        # --------------Subscribers------------------
        rospy.Subscriber('mavros/state', State, callback=self._stateCb, queue_size=10)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self._local_pose_Cb, queue_size=10)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback=self._local_vel_Cb, queue_size=10)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback=self._click_goalCb, queue_size=1)  # clicked goal pose from rviz
        # get depth image from different sources
        if self.image_source == 'gazebo':
            rospy.Subscriber('/camera/depth/image_raw', Image, callback=self._image_gazebo_Cb, queue_size=10)
        elif self.image_source == 'realsense':
            rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback=self._image_realsense_Cb, queue_size=10)
        else:
            rospy.logerr("image_source error")
        
        
# call back functions
    def _stateCb(self, msg):
        self._mavros_state = msg

    def _local_pose_Cb(self, msg):
        self.pose_local = msg
    
    def _local_vel_Cb(self, msg):
        self.vel_local = msg

    def _image_gazebo_Cb(self, msg):
        depth_image_msg = msg

        # transfer image from msg to cv2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding=depth_image_msg.encoding)
        except CvBridgeError as e:
            print(e)
        
        # get image in meters
        image = np.array(cv_image, dtype=np.float32)

        # deal with nan
        image[np.isnan(image)] = self.max_depth_meter_gazebo
        image_small = cv2.resize(image, (100, 80), interpolation = cv2.INTER_AREA)
        self._depth_image_meter = np.copy(image_small)

        # get image gray (0-255)
        image_gray = self._depth_image_meter / self.max_depth_meter_gazebo * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = np.copy(image_gray_int)
        
        # publish image topic
        image_msg = self.bridge.cv2_to_imgmsg(self._depth_image_gray)
        self._depth_image_gray_input.publish(image_msg)

    def _image_realsense_Cb(self, msg):
        depth_image_msg = msg

        # get depth image in mm
        try:
            # tranfer image from ros msg to opencv image encode F32C1
            cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding=depth_image_msg.encoding)
        except CvBridgeError as e:
            print(e)
        
        # rescale image to 100 80
        image = np.array(cv_image, dtype=np.float32)
        image_small = cv2.resize(image, (100, 80), interpolation = cv2.INTER_AREA)

        # get depth image in meter
        image_small_meter = image_small / 1000  # transter depth from mm to meter
        image_small_meter[image_small_meter == 0] = self.max_depth_meter_realsense
        self._depth_image_meter = np.copy(image_small_meter)

        # get depth image in gray (0-255)
        image_gray = self._depth_image_meter / self.max_depth_meter_realsense * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = np.copy(image_gray_int)

        # publish image topic
        image_msg = self.bridge.cv2_to_imgmsg(self._depth_image_gray)
        self._depth_image_gray_input.publish(image_msg)

    def _click_goalCb(self, msg):
        clicked_goal_pose = msg
        goal_x = clicked_goal_pose.pose.position.x
        goal_y = clicked_goal_pose.pose.position.y
        goal_z = self.goal_height
        rospy.logdebug('recieved clicked goal pose: {:.2f} {:.2f} {:.2f}'.format(goal_x, goal_y, goal_z))
        self._set_goal_pose(goal_x, goal_y, goal_z)

# check system 
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers, services and other simulation systems are
        operational.
        """
        self._check_all_topics_ready()
        # self._check_all_sensors_ready()
        # self._check_all_publishers_ready()
        return True

    def _check_all_topics_ready(self):
        rospy.logdebug('CHECK ALL TOPICS CONNECTION:')
        rospy.logdebug('wait for image')
        rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=5.0)
        rospy.logdebug('image ready')

        rospy.logdebug('wait for pose')
        rospy.wait_for_message("/mavros/local_position/pose", PoseStamped, timeout=5.0)
        rospy.logdebug('pose ready')

        rospy.logdebug('wait for velocity_local')
        rospy.wait_for_message("/mavros/local_position/velocity_local", TwistStamped, timeout=5.0)
        rospy.logdebug('velocity_local ready')

        rospy.logdebug('ALL TOPICS READY!!!')

    def _check_all_sensors_ready(self):
        rospy.logdebug("CHECK ALL SENSORS CONNECTION:")
        self._check_depth_image_ready()
        self._check_local_odometry_ready()
        rospy.logdebug("All Sensors CONNECTED and READY!")

    def _check_depth_image_ready(self):
        self._depth_image_meter = None
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

        state_feature = self._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        image_with_state = np.array([image_obs, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

    def check_sensor_data(self):
        pass

    def set_action_pose(self, action):
        '''
        generate control command from action
        action_real: forward speed, climb speed, yaw speed
        '''

        # test1: use first order low pass filter to actions
        action_smooth = self.filter_alpha * action + (1 - self.filter_alpha) * self.action_last
        rospy.logdebug('action smooth: ' + np.array2string(action_smooth, formatter={'float_kind':lambda x: "%.2f" % x}))
        self.action_last = action_smooth
        # get yaw and yaw setpoint 
        current_yaw = self.get_current_yaw()
        yaw_speed = action_smooth[2]
        yaw_setpoint = current_yaw + yaw_speed

        # yaw speed fov limitation
        speed_scale = 1- abs(math.degrees(action_smooth[2])) / 60
        # transfer dx dy from body frame to local frame
        dx_body = action_smooth[0] * speed_scale
        dy_body = 0
        dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

        pose_setpoint = PoseStamped()
        current_pose = self.pose_local
        pose_setpoint.pose.position.x = current_pose.pose.position.x + dx_local
        pose_setpoint.pose.position.y = current_pose.pose.position.y + dy_local
        # pose_setpoint.pose.position.z = current_pose.pose.position.z + action_smooth[1]
        pose_setpoint.pose.position.z = self._goal_pose.pose.position.z

        orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
        pose_setpoint.pose.orientation.x = orientation_setpoint[0]
        pose_setpoint.pose.orientation.y = orientation_setpoint[1]
        pose_setpoint.pose.orientation.z = orientation_setpoint[2]
        pose_setpoint.pose.orientation.w = orientation_setpoint[3]
        
        self._local_pose_setpoint_pub.publish(pose_setpoint)

        # publish visualization markers
        self.publish_marker_setpoint_pose(pose_setpoint)
        # print(self._goal_pose)
        self.publish_marker_goal_pose(self._goal_pose)

    def set_action_vel(self, action):
        
        control_msg = PositionTarget()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = 'local_origin'

        # BODY_NED
        control_msg.coordinate_frame = 8
        # use vx, vz, yaw_rate
        # control_msg.type_mask = int('111010111110', 2)
        control_msg.type_mask = int('011111000111', 2)

        # yaw speed fov limitation
        speed_scale = 1- abs(math.degrees(action[2])) / 60

        control_msg.velocity.x = action[0] * speed_scale / 4
        control_msg.velocity.y = 0
        control_msg.velocity.z = action[1]
        
        control_msg.yaw_rate = action[2]

        self._setpoint_raw_pub.publish(control_msg)

        self.publish_marker_goal_pose(self._goal_pose)

        # publish action and state
        action_msg = vel_cmd()
        action_msg.vel_xy = control_msg.velocity.x
        action_msg.vel_z = control_msg.velocity.z
        action_msg.yaw_rate = control_msg.yaw_rate
        self._action_msg_pub.publish(action_msg)

        state_vel_msg = vel_cmd()
        state_vel_msg.vel_xy = self.state_feature_raw[3]
        state_vel_msg.vel_z = self.state_feature_raw[4]
        state_vel_msg.yaw_rate = self.state_feature_raw[5]
        self._state_vel_msg_pub.publish(state_vel_msg)


# utils
    def publish_vel_raw(self, vel_cmd_enu):
        '''
        publish velocity control command in ENU coordinate
        '''
        control_msg = PositionTarget()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = 'local_origin'

        # BODY_NED
        control_msg.coordinate_frame = 8
        # use vx, vz, yaw_rate
        # control_msg.type_mask = int('111010111110', 2)
        control_msg.type_mask = int('011111000111', 2)

        control_msg.velocity.x = vel_cmd_enu[0]
        control_msg.velocity.y = 0
        control_msg.velocity.z = vel_cmd_enu[1]
        
        control_msg.yaw_rate = vel_cmd_enu[2]

        self._setpoint_raw_pub.publish(control_msg)
        
    def publish_marker_goal_pose(self, goal_pose):
        # publish goal pose marker
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'local_origin'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = goal_pose.pose.position
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        self._goal_pose_marker_pub.publish(marker)

    def publish_marker_setpoint_pose(self, pose_setpoint):
        
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'local_origin'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = pose_setpoint.pose.position
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 0.8
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self._setpoint_marker_pub.publish(marker)

    def _get_state_feature(self):
        '''
        Airsim pose use NED SYSTEM
        Gazebo pose z-axis up is positive different from NED
        Gazebo twist using body frame
        '''
        goal_pose = self._goal_pose
        current_pose = self.pose_local
        current_vel = self.vel_local
        # get distance and angle in polar coordinate
        # transfer to 0~255 image formate for cnn
        relative_pose_x = goal_pose.pose.position.x - current_pose.pose.position.x
        relative_pose_y = goal_pose.pose.position.y - current_pose.pose.position.y
        relative_pose_z = goal_pose.pose.position.z - current_pose.pose.position.z
        distance = math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2))
        relative_yaw = self._get_relative_yaw(current_pose, goal_pose)

        distance_norm = distance / self.goal_distance * 255
        vertical_distance_norm = (-relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255
        
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255

        # current speed and angular speed
        current_vel_local = current_vel.twist
        linear_velocity_xy = current_vel_local.linear.x  # forward velocity
        linear_velocity_xy = math.sqrt(pow(current_vel_local.linear.x, 2) + pow(current_vel_local.linear.y, 2))
        linear_velocity_norm = linear_velocity_xy / self.max_vel_x * 255
        linear_velocity_z = current_vel_local.linear.z  #  vertical velocity
        linear_velocity_z_norm = (linear_velocity_z / self.max_vel_z / 2 + 0.5) * 255
        angular_velocity_norm = (-current_vel_local.angular.z / self.max_vel_yaw_rad / 2 + 0.5) * 255  # TODO: check the sign of the 

        self.state_feature_raw = np.array([distance, relative_pose_z, relative_yaw, linear_velocity_xy, linear_velocity_z, -current_vel_local.angular.z])
        state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm, linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
        state_norm = np.clip(state_norm, 0, 255)
        self.state_feature_norm = state_norm / 255

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

        # publish goal pose
        self._goal_pose.pose.position.x = x
        self._goal_pose.pose.position.y = y
        self._goal_pose.pose.position.z = z

        self._goal_pose_pub.publish(self._goal_pose)
        rospy.logdebug('set goal pose to {:.2f} {:.2f} {:.2f}'.format(x, y, z))

        # update goal distance
        self.goal_distance = self.get_dist_from_pose_2d(self._goal_pose, self.pose_local)

    def get_dist_from_pose_2d(self, pose1, pose2):
        # calculate distance from two pose
        x1 = pose1.pose.position.x
        x2 = pose2.pose.position.x

        y1 = pose1.pose.position.y
        y2 = pose2.pose.position.y

        distance = math.sqrt(pow((x1-x2), 2) + pow((y1-y2), 2))

        return distance

    def get_current_yaw(self):
        orientation = self.pose_local.pose.orientation

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
        print('start model eval node')
        rospy.init_node('model_eval', anonymous=True, log_level=rospy.DEBUG)
        node_me = ModelEvalNode()
    except rospy.ROSInterruptException:
        print('ros node model-evaluation terminated...')