#!/usr/bin/env python3
'''
Author: Lei He
Date: 2020-08-29 14:51:26
LastEditTime: 2020-10-12 16:37:15
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
from mavros_msgs.msg import State, PositionTarget
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

        # state variables
        self.current_odom = Odometry()
        self.current_pose = PoseStamped()
        self.velocity_local = TwistStamped()
        self.goal_pose_rviz = PoseStamped()
        self.goal_pose_current = PoseStamped()
        
        self._depth_image_meter = np.zeros((self.image_height, self.image_width))
        self._depth_image_gray = np.zeros((self.image_height, self.image_width))

        self._check_all_systems_ready()

        # load model
        self.model = TD3.load('/home/helei/catkin_ws_rl/src/explainable_rl_ros/scripts/models/test_model')
        print('model load success')

        # set goal pose
        self.set_init_goal_pose()
       
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
        self.time_for_control_second = 1 / self.control_rate

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

        self.goal_accept_radius = cfg.getfloat('gazebo', 'goal_accept_radius')
        self.takeoff_hight = cfg.getfloat('gazebo', 'takeoff_hight')

        self.goal_init_x = cfg.getfloat('gazebo', 'goal_init_x')
        self.goal_init_y = cfg.getfloat('gazebo', 'goal_init_y')

        self.fov_h_degrees = cfg.getfloat('input_image', 'fov_horizontal_degrees')

    def set_sub_pub(self):
        # --------------Subscribers-------------------
        # state
        self._mavros_state = State()
        rospy.Subscriber('mavros/state', State, callback=self._stateCb, queue_size=10)
        # depth image
        rospy.Subscriber('/camera/depth/image_raw', Image, callback=self._imageCb, queue_size=10)
        # local odometry
        rospy.Subscriber('/mavros/local_position/odom', Odometry, callback=self._local_odomCb, queue_size=10)
        # local velocity
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback=self._velocity_localCb, queue_size=10)
        # rviz goal pose 
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback=self._update_goal_pose_rvizCb, queue_size=10)

        # --------------Publishers--------------------
        self._local_pose_setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local_test',PoseStamped, queue_size=10)
        self._setpoint_velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self._setpoint_raw_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self._setpoint_marker_pub = rospy.Publisher('/network/setpoint', Marker, queue_size=10)
        self._goal_pose_pub = rospy.Publisher('/network/current_goal_pose', PoseStamped, queue_size=10)
        self._marker_goal_pub = rospy.Publisher('/network/goal_marker', Marker, queue_size=10)

        # for test
        self._test_vel_cmd_pub = rospy.Publisher('/network/test/vel_cmd', TwistStamped, queue_size=10)
        self._test_vel_state_pub = rospy.Publisher('/network/test/vel_state', TwistStamped, queue_size=10)


# call back functions
    def _stateCb(self, msg):
        self._mavros_state = msg

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
        self.current_odom = msg
        self.current_pose = self.current_odom.pose

    def _update_goal_pose_rvizCb(self, msg):
        # print('update_goal_pose')
        self.goal_pose_rviz = msg
        self.goal_pose_rviz.pose.position.z = self.takeoff_hight
        self._update_goal_pose(self.goal_pose_rviz)
        
    def _velocity_localCb(self, msg):
        self.velocity_local = msg

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
        rospy.logdebug("All Sensors CONNECTED and READY!")

    def _check_depth_image_ready(self):
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

    
# main functions

    def get_obs(self):
        image = self._depth_image_gray.copy() # Note: check image format. Now is 0-black near 255-wight far

        # transfer image to image obs according to 0-far  255-nears
        image_obs = 255 - image

        state_feature_array = np.zeros((self.image_height, self.image_width))

        state_feature = self._get_state_feature(self.current_odom, self.goal_pose_current, self.velocity_local)
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

        pose_setpoint = PoseStamped()

        current_distance = self.get_dist_from_pose_2d(self.current_pose, self.goal_pose_current)
        # print(current_distance)

        if current_distance < self.goal_accept_radius:
            pose_setpoint = self.goal_pose_current
        else:
            # 1. get actions
            cmd_vel_x = float(action[0])
            cmd_vel_z = float(action[1])
            cmd_yaw_rate = float(action[2])

            # 2. get current state
            velocity_local = self.velocity_local
            forward_speed = math.sqrt(pow(velocity_local.twist.linear.x, 2) + pow(velocity_local.twist.linear.y, 2))
            current_vel_x = forward_speed
            current_vel_z = velocity_local.twist.linear.z
            current_vel_yaw = velocity_local.twist.angular.z

            current_state = np.array([current_vel_x, current_vel_z, current_vel_yaw])
            self.publish_vel_state(current_state)

            # acc limitation
            vel_x_range = np.array([current_vel_x - self.acc_lim_x * self.time_for_control_second, current_vel_x + self.acc_lim_x * self.time_for_control_second])
            vel_z_range = np.array([current_vel_z - self.acc_lim_z * self.time_for_control_second, current_vel_z + self.acc_lim_z * self.time_for_control_second])
            vel_yaw_range = np.array([current_vel_yaw - self.acc_lim_yaw_rad * self.time_for_control_second, current_vel_yaw + self.acc_lim_yaw_rad * self.time_for_control_second])
            cmd_vel_x_new = np.clip(cmd_vel_x, vel_x_range[0], vel_x_range[1])
            cmd_vel_z_new = np.clip(cmd_vel_z, vel_z_range[0], vel_z_range[1])
            cmd_yaw_rate_new = np.clip(cmd_yaw_rate, vel_yaw_range[0], vel_yaw_range[1])

            

            # velocity limitation
            cmd_vel_x_final = np.clip(cmd_vel_x_new, self.min_vel_x, self.max_vel_x)
            cmd_vel_z_final = np.clip(cmd_vel_z_new, -self.max_vel_z, self.max_vel_z)
            cmd_yaw_rate_final = np.clip(cmd_yaw_rate_new, -self.max_vel_yaw_rad, self.max_vel_yaw_rad)

            # print('curr: {:.2f} cmd: {:.2f} new: {:.2f} final: {:.2f}'.format(current_vel_yaw, cmd_yaw_rate, cmd_yaw_rate_new, cmd_yaw_rate_final))

            # FoV limitation
            speed_scale = (self.fov_h_degrees - math.degrees(abs(cmd_yaw_rate_final))) / self.fov_h_degrees
            speed_scale = max(0, speed_scale)
            # print('cmd: {:.2f} scale: {:.2f}'.format(cmd_yaw_rate_final, speed_scale))
            speed_scale = 1
            cmd_vel_x_final = cmd_vel_x_final * speed_scale

            self.vel_cmd_final = np.array([cmd_vel_x_final, cmd_vel_z_final, cmd_yaw_rate_final])
            # self.publish_vel_setpoint(self.vel_cmd_final)
            # self.publish_vel_setpoint(action)
            # self.vel_cmd_final = np.array([5, 0, 0])
            print('comm: {:.2f} {:.2f} {:.2f}'.format(action[0], action[1], action[2]))
            print('cont: {:.2f} {:.2f} {:.2f}'.format(self.vel_cmd_final[0], self.vel_cmd_final[1], self.vel_cmd_final[2]))
            print('real: {:.2f} {:.2f} {:.2f}'.format(current_vel_x, current_vel_z, current_vel_yaw))
            self.publish_vel_raw(self.vel_cmd_final)
            # self.publish_vel_raw(action)

            # get yaw and yaw setpoint 
            current_yaw = self.get_current_yaw()
            yaw_speed = self.vel_cmd_final[2]
            yaw_setpoint = current_yaw + yaw_speed

            # transfer dx dy from body frame to local frame
            dx_body = action[0]
            dy_body = 0
            dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

            pose_setpoint = PoseStamped()
            pose_setpoint.pose.position.x = self.current_pose.pose.position.x + dx_local
            pose_setpoint.pose.position.y = self.current_pose.pose.position.y + dy_local
            pose_setpoint.pose.position.z = self.current_pose.pose.position.z + action[1]

            orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
            pose_setpoint.pose.orientation.x = orientation_setpoint[0]
            pose_setpoint.pose.orientation.y = orientation_setpoint[1]
            pose_setpoint.pose.orientation.z = orientation_setpoint[2]
            pose_setpoint.pose.orientation.w = orientation_setpoint[3]

        self.publish_wp_setpoint(pose_setpoint)


    def publish_wp_setpoint(self, pose_setpoint):
        '''
        publish waypoint setpoint and it's markers
        '''

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


    def publish_vel_setpoint(self, vel_cmd):
        # get control cmd
        cmd_vel_x_final = vel_cmd[0]
        cmd_vel_z_final = vel_cmd[1]
        cmd_vel_yaw_final = vel_cmd[2]
        
        # generate topic msg
        vel_setpoint = TwistStamped()
        # vel_setpoint.twist.linear.x = cmd_vel_x_final
        vel_setpoint.twist.linear.x = cmd_vel_x_final
        vel_setpoint.twist.linear.y = 0
        vel_setpoint.twist.linear.z = cmd_vel_z_final
        vel_setpoint.twist.angular.x = 0
        vel_setpoint.twist.angular.y = 0
        vel_setpoint.twist.angular.z = cmd_vel_yaw_final

        self._setpoint_velocity_pub.publish(vel_setpoint)

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

    def publish_vel_state(self, vel_state):
        '''
        publish current velocity state in local frame 
        '''
        msg = TwistStamped()
        msg.twist.linear.x = vel_state[0]
        msg.twist.linear.z = vel_state[1]
        msg.twist.angular.z = vel_state[2]

        self._test_vel_state_pub.publish(msg)
# utils

    def _get_state_feature(self, current_odom, goal_pose, velocity_local):
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
        current_vel_local = velocity_local.twist
        linear_velocity_xy = math.sqrt(pow(current_vel_local.linear.x, 2) + pow(current_vel_local.linear.y, 2)) # forward velocity
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

    def _update_goal_pose(self, goal_pose):
        '''
        update goal pose
        update goal_distance once have new goal
        publish current goal pose and the marker
        '''
        self.goal_pose_current = goal_pose
        self.goal_distance = self.get_dist_from_pose_2d(self.goal_pose_current, self.current_pose)

        self._publish_goal_pose(self.goal_pose_current)

    def get_dist_from_pose_2d(self, pose1, pose2):
        # calculate distance from two pose
        x1 = pose1.pose.position.x
        x2 = pose2.pose.position.x

        y1 = pose1.pose.position.y
        y2 = pose2.pose.position.y

        distance = math.sqrt(pow((x1-x2), 2) + pow((y1-y2), 2))

        return distance

    def _publish_goal_pose(self, goal_pose):
        '''
        publish goal pose and marker
        '''
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = 'local_origin'
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = 4
        m.scale.y = 4
        m.scale.z = 4
        m.color.a = 0.2
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.pose = goal_pose.pose
        
        self._marker_goal_pub.publish(m)
        self._goal_pose_pub.publish(self.goal_pose_current)

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

    def set_init_goal_pose(self):
        '''
        set init goal pose from config
        '''
        # print('set init pose')
        goal_pose = PoseStamped()
        goal_pose.pose.position.x = self.goal_init_x
        goal_pose.pose.position.y = self.goal_init_y
        goal_pose.pose.position.z = self.takeoff_hight

        self._update_goal_pose(goal_pose)

if __name__ == "__main__":
    try:
        print('init model eval node')
        rospy.init_node('model_eval')
        node_me = ModelEvalNode()
    except rospy.ROSInterruptException:
        print('ros node model-evaluation terminated...')