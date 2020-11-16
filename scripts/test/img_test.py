'''
Date: 2020-11-16 14:29:00
LastEditors: Lei He
LastEditTime: 2020-11-16 20:34:47
FilePath: /explainable_rl_ros/scripts/test/img_test.py
'''
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class ImageSubTest():
    '''
    read and decode of the compressed depth image
    '''
    def __init__(self):
        super().__init__()
        rospy.init_node("image sub test")

        self.bridge = CvBridge()

        depth_image_topic = '/camera/aligned_depth_to_color/image_raw/compressedDepth'

        rospy.Subscriber(depth_image_topic, CompressedImage, callback=self.image_Cb, queue_size=10)

        self.control_rate = 10
        
        while not rospy.is_shutdown():

            rospy.sleep(1 / self.control_rate)
    
    def image_Cb(self, msg):
        '''
        ref: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/
        '''
        path_depth = 'test_dir'
        # 'msg' as type CompressedImage
        depth_fmt, compr_type = msg.format.split(';')
        # remove white space
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()
        if compr_type != "compressedDepth":
            raise Exception("Compression type is not 'compressedDepth'."
                            "You probably subscribed to the wrong topic.")

        # remove header from raw data
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]

        depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED).astype('uint8')
        if depth_img_raw is None:
            # probably wrong header size
            raise Exception("Could not decode compressed depth image."
                            "You may need to change 'depth_header_size'!")

        if depth_fmt == "16UC1":
            # write raw image data
            path = os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png")
            # cv2.imwrite(path, depth_img_raw)
            cv2.imshow('depth', depth_img_raw)
            cv2.waitKey(1)
        elif depth_fmt == "32FC1":
            raw_header = msg.data[:depth_header_size]
            # header: int, float, float
            [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
            depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32)-depthQuantB)
            # filter max values
            depth_img_scaled[depth_img_raw==0] = 0

            # depth_img_scaled provides distance in meters as f32
            # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
            depth_img_mm = (depth_img_scaled*1000).astype(np.uint16)
            cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_mm)
        else:
            raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")


if __name__ == "__main__":
    os.makedirs('test_dir', exist_ok=True)
    ist = ImageSubTest()