'''
Author: Lei He
Date: 2020-10-30 09:56:57
LastEditTime: 2020-10-30 10:26:51
LastEditors: Lei He
Description: Used for smooth waypoint generated by network controller
FilePath: /catkin_py3/src/explainable_rl_ros/scripts/utils/__pycache__/waypoint_smoother.py
'''
import numpy as np

class WpSmoother():
    def __init__(self):
        super().__init__()
        self.waypoint_origin = np.zeros(3)

        # waypoint smooth
        self.setpoint_yaw_rad = 0
        self.setpoint_yaw_velocity = 0
        self.last_speed = 0

        self.goto_position = np.zeros(3)
        self.adapted_goto_position = np.zeros(3)
        self.smoothed_goto_position = np.zeros(3)

    def smooth_wp(self):
        return np.zeros(3)

    def set_action_smooth(self, action):
        '''
        description: set action using smooth method according to PX4 avoidance
        '''
        # generate goto_position
        dx_body = action[0]
        dy_body = 0

        current_yaw = self.get_current_yaw()
        yaw_speed = action[2]
        yaw_setpoint = current_yaw + yaw_speed
        
        dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

        current_pose = self.pose_local
        goto_x = current_pose.pose.position.x + dx_local
        goto_y = current_pose.pose.position.y + dy_local
        goto_z = current_pose.pose.position.z + action[1]

        self.goto_position = np.array([goto_x, goto_y, goto_z])

    def next_smooth_yaw(self, wp_sp_ori):
        '''
        description: smooth yaw_rate command to get better pose setpoint
        not finished
        '''
        # get desired_sp_yaw_rad from current pose and wp_sp_ori
        current_pose = self.pose_local

        # if new way point is near to current way point, keep current yaw
        if self.get_dist_from_pose_2d(current_pose, wp_sp_ori) > 0.1:
            desired_setpoint_yaw_rad = 0
        else:
            desired_setpoint_yaw_rad = self.get_current_yaw()

        P_constant_xy = smoothing_speed_xy
        D_constant_xy = 2.0 * math.sqrt(P_constant_xy) # critically damped

        desired_yaw_velocity = 0.0

        yaw_diff = desired_setpoint_yaw_rad - self.setpoint_yaw_rad

        p = yaw_diff * P_constant_xy
        d = (desired_yaw_velocity - self.setpoint_yaw_velocity) * D_constant_xy

        dt = 1 / self.control_rate
        self.setpoint_yaw_velocity += (p + d) * dt

        self.setpoint_yaw_rad += self.setpoint_yaw_velocity * dt

        self.setpoint_yaw_rad = self.wrapAngleToPlusMinusPI(self.setpoint_yaw_rad)

    def adapt_speed(self, action):
        # low pass filter
        dt = 1 / self.control_rate
        filter_time_constant = 0.9
        alpha = dt / (filter_time_constant + dt)
        speed_sp = alpha * speed_sp + (1- alpha) * self.last_speed

        # reduce speed while high yaw rate command
        scale = 1 - action[2] / self.max_vel_yaw_rad
        speed_sp *= scale

        self.last_speed = speed_sp

        return speed_sp

    def smooth_waypoint(self):

        P_constant = np.array([self.smoothing_speed_xy, self.smoothing_speed_xy, self.smoothing_speed_z])
        D_constant = 2 * math.sqrt(P_constant)

        desired_location = self.adapted_goto_position

        location_diff = desired_location - smoothed_goto_location_

        velocity_diff = desired_velocity - smoothed_goto_location_velocity_

        # p = 
        # d = 

        smoothed_goto_location_velocity_ += (p + d) * dt

        smoothed_goto_location_ += smoothed_goto_location_velocity_ * dt
        