'''
Date: 2020-11-16 14:29:00
LastEditors: Lei He
LastEditTime: 2020-11-19 10:03:28
FilePath: /explainable_rl_ros/scripts/test/img_test.py
'''
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os

class ImageSubTest():
    '''
    read and decode of the compressed depth image from rosbag
    Step 1: play rosbag using rqt
    Step 2: run this script to create a new ros node to show image
    '''
    def __init__(self):
        super().__init__()
        rospy.init_node("image sub test")

        self.bridge = CvBridge()

        # depth_image_topic = '/camera/aligned_depth_to_color/image_raw/compressedDepth'
        depth_image_topic = '/camera/depth/image_raw'
        rospy.Subscriber(depth_image_topic, Image, callback=self.image_gazebo_Cb, queue_size=10)

        # rospy.Subscriber(depth_image_topic, CompressedImage, callback=self.image_Cb, queue_size=10)

        self.control_rate = 10
        self.max_depth_meter_realsense = 10

        self.depth_img_num = 0
        
        while not rospy.is_shutdown():

            rospy.sleep(1 / self.control_rate)

    def image_gazebo_Cb(self, msg):
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
        # cv2.imshow('depth gray', image_gray_int)
        # cv2.waitKey(1)
    
    def image_compressed_Cb(self, msg):
        print('get depth image')
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

        depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
        
        if depth_img_raw is None:
            # probably wrong header size
            raise Exception("Could not decode compressed depth image."
                            "You may need to change 'depth_header_size'!")
        
        cv_image = depth_img_raw
        # rescale image to 100 80
        image = np.array(cv_image, dtype=np.float32)
        np.save('depth_raw_rs', image)
        image_small = cv2.resize(image, (100, 80), interpolation = cv2.INTER_NEAREST)

        # get depth image in meter
        image_small_meter = image_small / 1000  # transter depth from mm to meter
        image_small_meter[image_small_meter == 0] = self.max_depth_meter_realsense
        image_small_meter = np.clip(image_small_meter, 0, self.max_depth_meter_realsense)
        self._depth_image_meter = np.copy(image_small_meter)

        # get depth image in gray (0-255)
        image_gray = self._depth_image_meter / self.max_depth_meter_realsense * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = np.copy(image_gray_int)

        obs = 255 - self._depth_image_gray
        # cv2.imwrite('depth_{}.jpg'.format(self.depth_img_num), obs)
        print(self.depth_img_num)
        self.depth_img_num += 1

        cv2.imshow('obs', obs)
        cv2.imshow('depth', self._depth_image_gray)
        cv2.waitKey(1)

        # if depth_fmt == "16UC1":
        #     # write raw image data
        #     path = os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png")
        #     # cv2.imwrite(path, depth_img_raw)
        #     cv2.imshow('depth', depth_img_raw)
        #     cv2.waitKey(1)
        # elif depth_fmt == "32FC1":
        #     raw_header = msg.data[:depth_header_size]
        #     # header: int, float, float
        #     [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
        #     depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32)-depthQuantB)
        #     # filter max values
        #     depth_img_scaled[depth_img_raw==0] = 0

        #     # depth_img_scaled provides distance in meters as f32
        #     # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
        #     depth_img_mm = (depth_img_scaled*1000).astype(np.uint16)
        #     cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_mm)
        # else:
        #     raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")


if __name__ == "__main__":
    os.makedirs('test_dir', exist_ok=True)
    ist = ImageSubTest()