# explainable_rl_ros
 ros package for explainable RL paper


## install
- create new workspace for python3
    ```
    mkdir -p ~/catkin_py3/src
    cd ~/catkin_py3
    catkin init
    ```

- config catkin workspace using python3 
  - for jetson nano
    ```
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so
    ```
  - for other ubuntu machine (they use different path for libpython3.6m.so)
    ```
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
    ```



- install python3-catkin-pkg
  - `sudo apt-get install python3-catkin-pkg-modules`
  
- build **geometry2**, **vision_opencv** and **explainable_rl_ros** with python3
    ```
    cd src
    git clone https://github.com/ros/geometry2
    cd geometry2
    git checkout -b melodic-devel origin/melodic-devel
    cd ..

    git clone https://github.com/ros-perception/vision_opencv.git
    cd vision_opencv
    git checkout -b melodic origin/melodic
    cd ..
    
    git clone https://github.com/heleidsn/explainable_rl_ros.git
    cd ..
    catkin build 
    source devel/setup.bash
    ```

- run 
  ```
  rosrun explainable_rl_ros model_evaluation.py 
  ```

## install dependencies

- Tensorflow
  - 1.14.0 (Stable-Baselines only supports Tensorflow versions from 1.8.0 to 1.14.0)
  - Install Tensorflow on Jetson Nano: https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
  - check jetpack version
    - `dpkg-query --show nvidia-l4t-core`
    - `nvidia-l4t-core	32.4.3-20200625213809`
    - L4T 32.4.3 using JetPack 4.4
    - `sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'`

- stable baselines
  - https://forums.developer.nvidia.com/t/pip-install-cant-resolve-opencv-dependency-in-jetpack4-3/120723/3
  - `pip3 install stable-baselines`

- numpy
  - `pip3 uninstall numpy`
  - `pip3 install numpy==1.17.0`


## problems

### for jetson nano

- vision_opencv problem

  ```
  [ERROR] [1601915497.691958]: bad callback: <bound method ModelEvalNode._imageCb of <__main__.ModelEvalNode object at 0x7f876cd588>>
  Traceback (most recent call last):
    File "/opt/ros/melodic/lib/python2.7/dist-packages/rospy/topics.py", line 750, in _invoke_callback
      cb(msg)
    File "/home/helei/catkin_py3/src/explainable_rl_ros/scripts/model_evaluation.py", line 130, in _imageCb
      cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding='passthrough')
    File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 163, in imgmsg_to_cv2
      dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
    File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 99, in encoding_to_dtype_with_channels
      return self.cvtype2_to_dtype_with_channels(self.encoding_to_cvtype2(encoding))
    File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 91, in encoding_to_cvtype2
      from cv_bridge.boost.cv_bridge_boost import getCvType
  ImportError: dynamic module does not define module export function (PyInit_cv_bridge_boost)
  ```
- vision_opencv needs to be compiled with python3, but jetson nano using opencv4, vision_opencv needs opencv3....
- https://github.com/ros-perception/vision_opencv/issues/272

- Problem solve
  - download vision_opencv
  - `git clone https://github.com/ros-perception/vision_opencv.git`
  - `git checkout noetic`
  - change requement to fit python 3.6.9 on Jetson Nano
    - change **cv_bridge/CMakeList.txt** and build with python3
    - ```# if(NOT ANDROID)
      #   find_package(PythonLibs)
      #   if(PYTHONLIBS_VERSION_STRING VERSION_LESS "3.8")
      #     # Debian Buster
      #     find_package(Boost REQUIRED python37)
      #   else()
      #     # Ubuntu Focal
      #     find_package(Boost REQUIRED python)
      #   endif()
      # else()
      # find_package(Boost REQUIRED)
      # endif()

      if(NOT ANDROID)
        find_package(PythonLibs)
        if(PYTHONLIBS_VERSION_STRING VERSION_LESS 3)
          find_package(Boost REQUIRED python)
        else()
          find_package(Boost REQUIRED python3)
        endif()
      else()
      find_package(Boost REQUIRED)
      endif()
      ```

### for other ubuntu machine

#### install opencv for python
- `pip3 install --upgrade pip`
- `pip3 install rospkg`
- `pip3 install opencv-python`
- `pip3 install tensorflow-gpu==1.14.0`

## sub and pub

- sub1: /camera/aligned_depth_to_color/image_raw
  - `rostopic hz /camera/aligned_depth_to_color/image_raw`
  - `average rate: 14.073
	min: 0.063s max: 0.079s std dev: 0.00454s window: 28`
- sub2: mavros/state
  - `average rate: 1.058
	min: 0.514s max: 1.006s std dev: 0.15268s window: 10`
  - `header: 
  seq: 259
  stamp: 
    secs: 1601977531
    nsecs: 907081432
  frame_id: ''
connected: True
armed: False
guided: False
manual_input: True
mode: "MANUAL"
system_status: 3`

- sub3: /camera/depth/image_rect_raw
  - `average rate: 17.087
	min: 0.001s max: 0.084s std dev: 0.02623s window: 102`

- sub4: /mavros/local_position/odom
  - `average rate: 29.983
	min: 0.029s max: 0.037s std dev: 0.00186s window: 88`
  ```
  header: 
  seq: 15160
  stamp: 
    secs: 1601977778
    nsecs: 436370688
  frame_id: "local_origin"
  child_frame_id: "fcu"
  pose: 
    pose: 
      position: 
        x: 0.0
        y: 0.0
        z: -2.09963297844
      orientation: 
        x: 0.0159018854295
        y: 0.00122475778322
        z: -0.475007078863
        w: -0.879837421173
    covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  twist: 
    twist: 
      linear: 
        x: 0.00101573157178
        y: -0.00570582863205
        z: 1.021144044e-05
      angular: 
        x: -0.000607163237873
        y: -2.01841248781e-05
        z: -0.000148270453792
    covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ```

## logs

### 2020-11-14 test new AirSim model in ros env

对在AirSim中训练的模型`2020_11_13_08_29_remove_reach_and_crash_reward_-1_1`进行测试。
由于之前测试过每次prediction之前获取msg，发现获取msg的过程需要大概0.1s，所以还是改成callback的方式，直接调用。

简单测试之后发现一个问题：一直在爬升。。。

换用位置控制看看: 位置控制表现更差

下一步： 直接在gazebo环境中进行训练，实现real-time，放弃transfer learning
