#!/usr/bin/env python3
'''
Author: Lei He
Date: 2022-10-08 09:36:02
LastEditTime: 2022-10-10 22:59:03
LastEditors: Please set LastEditors
Description: ROS node pack for model evaluation in ros environment
FilePath: /explainable_rl_ros/scripts/model_evaluation.py
'''

import math
from configparser import ConfigParser

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import PositionTarget, State
from sensor_msgs.msg import CompressedImage, Image
from stable_baselines3 import SAC
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker


class ModelEvalNode():
    def __init__(self, config_path):
        rospy.logdebug('Start init ModelEvalNode...')

        self.bridge = CvBridge()
        self.cfg = ConfigParser()

        # load settings
        self.cfg.read(config_path)
        self.set_config(self.cfg)

        # set sub and pub
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

        # load model
        self.model = SAC.load(self.model_path)
        rospy.logdebug('model load success')

        # get action num. 2 for 2d, 3 for 3d
        self.action_num = self.model.action_space.shape[0]

        if self.action_num == 2:
            self.action_last = np.array([0, 0])
        elif self.action_num == 3:
            self.action_last = np.array([0, 0, 0])
        else:
            rospy.logerr("action num error: action num should be 2 or 3, now is {}".format(self.action_num))

        # set init goal pose
        self.set_goal_pose(self.goal_init_x, self.goal_init_y, self.goal_init_z)
        rospy.loginfo('init goal set to x:{}, y:{}, z:{}'.format(self.goal_init_x, self.goal_init_y, self.goal_init_z))

    def start_controller(self):
        rospy.logdebug('neural network controller start...')

        rate = rospy.Rate(self.control_rate)  # 10Hz
        rospy.loginfo('control rate: {}'.format(self.control_rate))

        self.check_topics_ready()
        
        record_data = False
        step = 0
        action_list = []
        state_raw_list = []
        obs_list = []

        while not rospy.is_shutdown():
            # 1. get obs
            step = step + 1
            print(step)
            obs = self.get_obs()

            # 2. get action
            action_real, _ = self.model.predict(obs, deterministic=True)
            # norm action
            # action_real[0] = action_real[0] / 5
            # action_real[1] = action_real[1] / 2
            # action_real[2] = action_real[2] / math.degrees(50)

            # action_real = np.array([3, 0, 0])
            # 3. set action
            self.set_action(action_real)
            
            # record data for 200 steps
            
            if step < 600 and record_data:
                obs_list.append(obs)
                # traj_list.append(pose)
                action_list.append(action_real)
                state_raw_list.append(self.state_feature_raw)
            elif step == 600 and record_data:
                np.save('action_eval_ros', action_list)
                np.save('state_eval_ros', state_raw_list)
                np.save('obs_eval_ros', obs_list)
            else:
                action_list = []
                state_raw_list = []
                obs_list = []

            # 4. log some data
            rospy.logdebug('state_raw: ' + np.array2string(self.state_feature_raw, formatter={'float_kind': lambda x: "%.2f" % x}))
            rospy.logdebug('state_norm: ' + np.array2string(self.state_feature_norm, formatter={'float_kind': lambda x: "%.2f" % x}))
            rospy.logdebug('action real: ' + np.array2string(action_real, formatter={'float_kind': lambda x: "%.2f" % x}))
            # rospy.logdebug('state_raw: ' + np.array2string(self.state_feature_raw))
            # rospy.logdebug('state_norm: ' + np.array2string(self.state_feature_norm))
            # rospy.logdebug('action real: ' + np.array2string(action_real))
            rospy.logdebug('obs min: {:.2f} max: {:.2f}'.format(self._depth_image_meter.min(), self._depth_image_meter.max()))

            rate.sleep()

    def set_config(self, cfg):
        # sim_setting section
        self.control_rate = cfg.getint('sim_setting', 'control_rate')
        self.image_source = cfg.get('sim_setting', 'image_source')
        assert self.image_source == 'gazebo' or self.image_source == 'realsense' or \
            self.image_source == 'rosbag' or self.image_source == 'airsim', 'image_source setting error: should be gazebo or realsense'

        self.control_method = cfg.get('sim_setting', 'control_method')
        assert self.control_method == 'velocity' or self.control_method == 'position', \
            'control_method setting error: should be velocity or position'

        self.model_path = cfg.get('sim_setting', 'model_path')

        # filter alpha for action low pass filter
        self.filter_alpha = self.cfg.getfloat('sim_setting', 'filter_alpha')

        self.goal_init_x = cfg.getfloat('sim_setting', 'goal_x')
        self.goal_init_y = cfg.getfloat('sim_setting', 'goal_y')
        self.goal_init_z = cfg.getfloat('sim_setting', 'goal_z')

        self.goal_distance = None
        self.max_depth_meter = cfg.getfloat('sim_setting', 'max_depth_meter')
        self.min_depth_meter = cfg.getfloat('sim_setting', 'min_depth_meter')

        self.image_height = cfg.getint('sim_setting', 'image_height')
        self.image_width = cfg.getint('sim_setting', 'image_width')

        self.state_feature_length = cfg.getint('sim_setting', 'state_feature_length')

        # uav_model section
        self.acc_lim_x = cfg.getfloat('uav_model', 'acc_lim_x')
        self.acc_lim_z = cfg.getfloat('uav_model', 'acc_lim_z')
        self.acc_lim_yaw_deg = cfg.getfloat('uav_model', 'acc_lim_yaw_deg')
        self.acc_lim_yaw_rad = math.radians(self.acc_lim_yaw_deg)

        self.max_vel_x = cfg.getfloat('uav_model', 'max_vel_x')
        self.min_vel_x = cfg.getfloat('uav_model', 'min_vel_x')
        self.max_vel_z = cfg.getfloat('uav_model', 'max_vel_z')
        self.max_vel_yaw_rad = math.radians(cfg.getfloat('uav_model', 'max_vel_yaw_deg'))

        self.max_vertical_difference = cfg.getfloat('uav_model', 'max_vertical_difference')

    def set_sub_pub(self):
        # --------------Publishers--------------------
        # current goal position
        self._goal_pose_pub = rospy.Publisher('/network/goal', PoseStamped, queue_size=10)
        self._goal_pose_marker_pub = rospy.Publisher('network/marker_goal', Marker, queue_size=10)

        # control command
        self._local_pose_setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self._setpoint_marker_pub = rospy.Publisher('network/marker_pose_setpoint', Marker, queue_size=10)
        self._setpoint_raw_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)

        # image
        self._depth_image_gray_input = rospy.Publisher('network/depth_image_input', Image, queue_size=10)

        # # debug info
        # self._action_msg_pub = rospy.Publisher('/network/debug/action', vel_cmd, queue_size=10)
        # self._state_vel_msg_pub = rospy.Publisher('/network/debug/state', vel_cmd, queue_size=10)

        # --------------Subscribers------------------
        # sub topic name
        self.sub_topic_state = '/mavros/state'
        self.sub_topic_local_pose = '/mavros/local_position/pose'
        self.sub_topic_local_vel = '/mavros/local_position/velocity_local'

        # mav state
        rospy.Subscriber('/mavros/state', State, callback=self._stateCb, queue_size=10)
        # local pose
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self._local_pose_Cb, queue_size=10)
        # local velocity
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback=self._local_vel_Cb, queue_size=10)
        # clicked goal pose from rviz
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback=self._click_goalCb, queue_size=1)

        # depth image from different sources
        if self.image_source == 'gazebo':
            self.sub_topic_depth_image = '/camera/depth/image_raw'
            rospy.Subscriber(self.sub_topic_depth_image, Image, callback=self._image_gazebo_Cb, queue_size=10)
        elif self.image_source == 'realsense':
            self.sub_topic_depth_image = '/camera/aligned_depth_to_color/image_raw'
            rospy.Subscriber(self.sub_topic_depth_image, Image, callback=self._image_realsense_Cb, queue_size=10)
        elif self.image_source == 'rosbag':
            self.sub_topic_depth_image = '/camera/aligned_depth_to_color/image_raw/compressedDepth'
            rospy.Subscriber(self.sub_topic_depth_image, CompressedImage, callback=self._image_rosbag_Cb, queue_size=10)
        elif self.image_source == 'airsim':
            self.sub_topic_depth_image = '/airsim_node/drone_1/camera_1/DepthVis'
            rospy.Subscriber(self.sub_topic_depth_image, Image, callback=self._image_airsim_Cb, queue_size=10)
        else:
            rospy.logerr("image_source error")

# call back functions
    def _stateCb(self, msg):
        self._mavros_state = msg

    def _local_pose_Cb(self, msg):
        self.pose_local = msg

    def _local_vel_Cb(self, msg):
        self.vel_local = msg

    def _image_rosbag_Cb(self, msg):
        # 'msg' as type CompressedImage
        depth_fmt, compress_type = msg.format.split(';')
        # remove white space
        depth_fmt = depth_fmt.strip()
        compress_type = compress_type.strip()
        if compress_type != "compressedDepth":
            raise Exception("Compression type is not 'compressedDepth'."
                            "You probably subscribed to the wrong topic.")

        # remove header from raw data
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]

        depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)

        if depth_img_raw is None:
            # probably wrong header size
            raise Exception("Could not decode compressed depth image."
                            "You may need to change 'depth_header_size'!")

        cv_image = depth_img_raw

        # rescale image to 100 80
        image = np.array(cv_image, dtype=np.float32)
        image_small = cv2.resize(image, (100, 80), interpolation=cv2.INTER_NEAREST)

        # get depth image in meter
        image_small_meter = image_small / 1000  # transfer depth from mm to meter

        # deal with zero values
        image_small_meter[image_small_meter == 0] = self.max_depth_meter
        image_small_meter = np.clip(image_small_meter, self.min_depth_meter, self.max_depth_meter)

        self._depth_image_meter = np.copy(image_small_meter)

        # get depth image in gray (0-255)
        image_gray = self._depth_image_meter / self.max_depth_meter * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = image_gray_int

        obs = 255 - image_gray_int
        cv2.imshow('obs', obs)
        cv2.waitKey(1)

    def _image_gazebo_Cb(self, msg):
        depth_image_msg = msg

        # transfer image from msg to cv2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg,
                                                 desired_encoding=depth_image_msg.encoding)
        except CvBridgeError as e:
            print(e)

        # get image in meters
        image = np.array(cv_image, dtype=np.float32)

        # rescale
        image_small = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # deal with nan
        image_small[np.isnan(image_small)] = self.max_depth_meter
        image_small = np.clip(image_small, self.min_depth_meter, self.max_depth_meter)
        self._depth_image_meter = np.copy(image_small)

        # get image gray (0-255)
        image_gray = self._depth_image_meter / self.max_depth_meter * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = np.copy(image_gray_int)

    def _image_realsense_Cb(self, msg):
        depth_image_msg = msg

        # get depth image in mm
        try:
            # transfer image from ros msg to opencv image encode F32C1
            cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg,
                                                 desired_encoding=depth_image_msg.encoding)
        except CvBridgeError as e:
            print(e)

        # rescale image to 100 80
        image = np.array(cv_image, dtype=np.float32)
        image_small = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        # get depth image in meter
        image_small_meter = image_small / 1000  # transfer depth from mm to meter

        # deal with zero values
        image_small_meter[image_small_meter == 0] = self.max_depth_meter
        image_small_meter = np.clip(image_small_meter, self.min_depth_meter, self.max_depth_meter)

        self._depth_image_meter = np.copy(image_small_meter)

        # get depth image in gray (0-255)
        image_gray = self._depth_image_meter / self.max_depth_meter * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = image_gray_int

    def _image_airsim_Cb(self, msg):
        data = msg

        # depth input is 0-20 meter
        depth_input = self.bridge.imgmsg_to_cv2(data, desired_encoding=data.encoding)
        # cv2.imshow('test', depth_input)
        # cv2.waitKey(1)

        image_small = cv2.resize(depth_input, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST) * 100

        depth_input = np.clip(image_small, 0, self.max_depth_meter)

        self._depth_image_meter = np.copy(depth_input)

        depth_input = depth_input / self.max_depth_meter * 255
        image_gray_int = depth_input.astype(np.uint8)
        self._depth_image_gray = np.copy(image_gray_int)

    def _click_goalCb(self, msg):
        clicked_goal_pose = msg
        goal_x = clicked_goal_pose.pose.position.x
        goal_y = clicked_goal_pose.pose.position.y
        goal_z = self.goal_init_z
        rospy.logdebug('received clicked goal pose: {:.2f} {:.2f} {:.2f}'.format(goal_x, goal_y, goal_z))
        self.set_goal_pose(goal_x, goal_y, goal_z)

# check system
    def check_topics_ready(self):
        rospy.logdebug('waiting for msgs')
        # wait for latest msgs
        image_msg = rospy.wait_for_message(self.sub_topic_depth_image, Image, timeout=1.0)
        local_pose_msg = rospy.wait_for_message(self.sub_topic_local_pose, PoseStamped, timeout=1.0)
        local_vel_msg = rospy.wait_for_message(self.sub_topic_local_vel, TwistStamped, timeout=1.0)

        rospy.logdebug('update state variables')
        if self.image_source == 'gazebo':
            self._image_gazebo_Cb(image_msg)
        elif self.image_source == 'realsense':
            self._image_realsense_Cb(image_msg)
        elif self.image_source == 'airsim':
            self._image_airsim_Cb(image_msg)
        else:
            print('image_source error')

        rospy.logdebug('update')
        self._local_pose_Cb(local_pose_msg)
        self._local_vel_Cb(local_vel_msg)

        rospy.logdebug('topics check finished')

# gym functions
    def get_obs(self):
        '''
        1. get depth image obs from gazebo msg
        '''
        # get depth image from current topic
        image = self._depth_image_gray.copy()  # Note: check image format. Now is 0-black near 255-wight far

        # transfer image to image obs according to 0-far  255-nears
        image_obs = 255 - image

        # publish image_obs
        image_obs_msg = self.bridge.cv2_to_imgmsg(image_obs)
        self._depth_image_gray_input.publish(image_obs_msg)

        '''
        2. get state feature
        '''
        state_feature_array = np.zeros((self.image_height, self.image_width))

        state_feature = self._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        image_with_state = np.array([image_obs, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state.astype(np.uint8)

    def _get_state_feature(self):
        '''
        get state feature with velocity!
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
        relative_yaw = -self._get_relative_yaw(current_pose, goal_pose)

        distance_norm = distance / self.goal_distance * 255
        vertical_distance_norm = (-relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) * 255

        # current speed and angular speed
        current_vel_local = current_vel.twist
        linear_velocity_xy = current_vel_local.linear.x  # forward velocity
        linear_velocity_xy = math.sqrt(pow(current_vel_local.linear.x, 2) + pow(current_vel_local.linear.y, 2))

        linear_velocity_z = current_vel_local.linear.z  # vertical velocity

        state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm])

        state_norm = np.clip(state_norm, 0, 255)
        # self.state_feature_norm = state_norm / 255

        self.state_feature_norm = state_norm
        self.state_feature_raw = np.array([distance, relative_pose_z, relative_yaw, linear_velocity_xy, linear_velocity_z, current_vel_local.angular.z])

        return state_norm

    def set_action(self, action):
        '''
        set action with real action command
        '''
        if self.control_method == 'velocity':
            self.set_action_vel(action)
        elif self.control_method == 'position':
            self.set_action_pose(action)

    def set_action_pose(self, action):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        pose_setpoint = PoseStamped()

        # get distance to goal
        goal_pose = self._goal_pose
        current_pose = self.pose_local
        # current_vel = self.vel_local
        # get distance and angle in polar coordinate
        # transfer to 0~255 image formate for cnn
        relative_pose_x = goal_pose.pose.position.x - current_pose.pose.position.x
        relative_pose_y = goal_pose.pose.position.y - current_pose.pose.position.y
        # relative_pose_z = goal_pose.pose.position.z - current_pose.pose.position.z
        distance = math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2))

        if distance < 3:
            # near the goal, set pose cmd to goal pose and current yaw
            pose_setpoint.pose.position = self._goal_pose.  pose.position
            current_yaw = self.get_current_yaw()
            orientation_setpoint = quaternion_from_euler(0, 0, current_yaw)
            pose_setpoint.pose.orientation.x = orientation_setpoint[0]
            pose_setpoint.pose.orientation.y = orientation_setpoint[1]
            pose_setpoint.pose.orientation.z = orientation_setpoint[2]
            pose_setpoint.pose.orientation.w = orientation_setpoint[3]
        elif self._depth_image_meter.min() < 0.5:
            pose_setpoint = current_pose
            rospy.logerr("too close to the obs, < 1m!!!")
        else:
            # use first order low pass filter to actions
            action_smooth = self.filter_alpha * action + (1 - self.filter_alpha) * self.action_last
            rospy.logdebug('action smooth: ' + np.array2string(action_smooth, formatter={'float_kind': lambda x: "%.2f" % x}))
            self.action_last = action_smooth

            # get yaw and yaw setpoint
            current_yaw = self.get_current_yaw()
            yaw_speed = action_smooth[-1] * 0.2
            yaw_setpoint = current_yaw + yaw_speed

            # transfer dx dy from body frame to local frame
            dx_body = action_smooth[0] * 0.5
            dy_body = 0
            dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

            pose_setpoint.pose.position.x = self.pose_local.pose.position.x + dx_local
            pose_setpoint.pose.position.y = self.pose_local.pose.position.y + dy_local
            if self.action_num == 3:
                pose_setpoint.pose.position.z = self.pose_local.pose.position.z + action_smooth[1]
            elif self.action_num == 2:
                pose_setpoint.pose.position.z = self._goal_pose.pose.position.z

            orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
            pose_setpoint.pose.orientation.x = orientation_setpoint[0]
            pose_setpoint.pose.orientation.y = orientation_setpoint[1]
            pose_setpoint.pose.orientation.z = orientation_setpoint[2]
            pose_setpoint.pose.orientation.w = orientation_setpoint[3]

        self._local_pose_setpoint_pub.publish(pose_setpoint)

        # publish visualization markers
        self.publish_marker_setpoint_pose(pose_setpoint)
        self.publish_marker_goal_pose(self._goal_pose)

    def set_action_vel(self, action):

        control_msg = PositionTarget()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = 'local_origin'

        # BODY_NED
        control_msg.coordinate_frame = 8
        # use vx, vz, yaw_rate
        control_msg.type_mask = int('011111000111', 2)

        # yaw speed fov limitation
        # speed_scale = 1- abs(math.degrees(action[2])) / 60

        # add acc limitation
        # 2. get current state

        # current_vel_x = self.state_feature_raw[3]
        # current_vel_z = self.state_feature_raw[4]
        # current_vel_yaw = self.state_feature_raw[5]

        # dt = 1 / self.control_rate
        # vel_x_range = np.array([current_vel_x - self.acc_lim_x * dt, current_vel_x + self.acc_lim_x * dt])
        # vel_z_range = np.array([current_vel_z - self.acc_lim_z * dt, current_vel_z + self.acc_lim_z * dt])
        # vel_yaw_range = np.array([current_vel_yaw - self.acc_lim_yaw_rad * dt, current_vel_yaw + self.acc_lim_yaw_rad * dt])
        # cmd_vel_x_new = np.clip(action[0], vel_x_range[0], vel_x_range[1])
        # cmd_vel_z_new = np.clip(action[1], vel_z_range[0], vel_z_range[1])
        # cmd_yaw_rate_new = np.clip(action[2], vel_yaw_range[0], vel_yaw_range[1])

        # # velocity limitation
        # cmd_vel_x_final = np.clip(cmd_vel_x_new, self.min_vel_x, self.max_vel_x)
        # cmd_vel_z_final = np.clip(cmd_vel_z_new, -self.max_vel_z, self.max_vel_z)
        # cmd_yaw_rate_final = np.clip(cmd_yaw_rate_new, -self.max_vel_yaw_rad, self.max_vel_yaw_rad)

        # control_msg.velocity.x = cmd_vel_x_final
        # control_msg.velocity.y = 0
        # control_msg.velocity.z = cmd_vel_z_final

        # control_msg.yaw_rate = cmd_yaw_rate_final

        action_smooth = self.filter_alpha * action + (1 - self.filter_alpha) * self.action_last
        self.action_last = action_smooth

        # direct control
        control_msg.velocity.x = action_smooth[0]
        control_msg.velocity.y = 0

        if self.action_num == 2:
            pose_z = self.pose_local.pose.position.z
            print(pose_z)
            z_error = 5 - pose_z
            control_msg.velocity.z = z_error * 5
        elif self.action_num == 3:
            control_msg.velocity.z = action_smooth[1]
        else:
            rospy.logerr("action num error: action num should be 2 or 3, now is {}".format(self.action_num))

        control_msg.yaw_rate = -action_smooth[-1]
        

        # control_msg.yaw_rate = 1

        rospy.logdebug('action_final: {:.2f} {:.2f} {:.2f}'.format(control_msg.velocity.x, control_msg.velocity.z, math.degrees(control_msg.yaw_rate)))

        # publish setpoint marker
        setpoint_pose = self.get_pose_from_velocity(action_smooth)
        self.publish_marker_setpoint_pose(setpoint_pose)

        self._setpoint_raw_pub.publish(control_msg)

        self.publish_marker_goal_pose(self._goal_pose)

        # # publish action and state
        # action_msg = vel_cmd()
        # action_msg.vel_xy = control_msg.velocity.x
        # action_msg.vel_z = control_msg.velocity.z
        # action_msg.yaw_rate = control_msg.yaw_rate
        # self._action_msg_pub.publish(action_msg)

        # state_vel_msg = vel_cmd()
        # state_vel_msg.vel_xy = self.state_feature_raw[3]
        # state_vel_msg.vel_z = self.state_feature_raw[4]
        # state_vel_msg.yaw_rate = self.state_feature_raw[5]
        # self._state_vel_msg_pub.publish(state_vel_msg)

    def get_pose_from_velocity(self, action):
        '''
        description: get pose according to the velocity action real
        '''
        # get current pose
        current_pose = self.pose_local
        # get setpoint pose
        dx_body = action[0]
        dy_body = 0

        current_yaw = self.get_current_yaw()
        yaw_speed = -action[-1]
        yaw_setpoint = current_yaw + yaw_speed

        dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

        # publish setpoint
        pose_setpoint = PoseStamped()
        current_pose = self.pose_local
        pose_setpoint.pose.position.x = current_pose.pose.position.x + dx_local
        pose_setpoint.pose.position.y = current_pose.pose.position.y + dy_local
        if self.action_num == 2:
            pose_setpoint.pose.position.z = self._goal_pose.pose.position.z
        elif self.action_num == 3:
            pose_setpoint.pose.position.z = current_pose.pose.position.z + action[1]
        else:
            rospy.logerr("action num error: action num should be 2 or 3, now is {}".format(self.action_num))

        orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
        pose_setpoint.pose.orientation.x = orientation_setpoint[0]
        pose_setpoint.pose.orientation.y = orientation_setpoint[1]
        pose_setpoint.pose.orientation.z = orientation_setpoint[2]
        pose_setpoint.pose.orientation.w = orientation_setpoint[3]

        return pose_setpoint

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

    def _get_relative_yaw(self, current_pose, goal_pose):
        # get relative angle
        relative_pose_x = goal_pose.pose.position.x - current_pose.pose.position.x
        relative_pose_y = goal_pose.pose.position.y - current_pose.pose.position.y
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        explicit_quat = [current_pose.pose.orientation.x, current_pose.pose.orientation.y,
                         current_pose.pose.orientation.z, current_pose.pose.orientation.w]

        yaw_current = euler_from_quaternion(explicit_quat)[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def set_goal_pose(self, x, y, z):

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

        current_orientation = [orientation.x, orientation.y, orientation.z, orientation.w]

        # current_attitude = euler_from_quaternion(current_orientation)
        current_yaw = euler_from_quaternion(current_orientation)[2]

        return current_yaw

    def point_transfer(self, x, y, theta):
        # transfer x, y to another frame
        x1 = x * math.cos(theta) + y * math.sin(theta)
        x2 = - x * math.sin(theta) + y * math.cos(theta)

        return x1, x2

    def wrapAngleToPlusMinusPI(self, angle):
        '''
        description: change angle to (-pi, pi)
        '''
        return angle - 2.0 * math.pi * math.floor(angle / (2.0 * math.pi) + 0.5)


if __name__ == "__main__":
    try:
        print('start model eval node')

        # print(sys.path)
        rospy.init_node('model_eval', anonymous=True, log_level=rospy.DEBUG)

        config_path = '/home/helei/catkin_py3/src/explainable_rl_ros/scripts/real_test_sb3/config/config_airsim.ini'
        node_me = ModelEvalNode(config_path)
        node_me.start_controller()

    except rospy.ROSInterruptException:
        print('ros node model-evaluation terminated...')
