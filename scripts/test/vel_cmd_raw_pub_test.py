#!/usr/bin/env python3

import rospy
import math
from mavros_msgs.msg import State, PositionTarget

class VelCmdPubTest():
    def __init__(self):
        super().__init__()
        rospy.init_node('vel_cmd_pub_test_node')

        self.setpoint_raw_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)

        self.control_rate = 10

        while not rospy.is_shutdown():
            # create 
            msg = self.create_cmd_msg()
            # publish vel raw cmd
            self.setpoint_raw_pub.publish(msg)

            rospy.sleep(1 / self.control_rate)

    def create_cmd_msg(self):

        control_msg = PositionTarget()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = 'local_origin'
        # BODY_NED
        control_msg.coordinate_frame = 8
        # use vx, vz, yaw_rate
        control_msg.type_mask = int('011111000111', 2)

        control_msg.velocity.x = 5
        control_msg.velocity.y = 0
        control_msg.velocity.z = 0
        control_msg.yaw_rate = math.radians(0)

        return control_msg


if __name__ == "__main__":
    VelCmdPubTest()