
# 解决ROS和python3的问题

ROS原生不支持python3,因此个别包需要在python3环境下进行重新编译,生成对应的文件.

比如在`import cv_bridge`的时候会出现报错：

```
Traceback (most recent call last):
  File "/home/helei/catkin_ws_rl/src/explainable_rl_ros/scripts/model_evaluation.py", line 71, in <module>
    node_me = ModelEvalNode()
  File "/home/helei/catkin_ws_rl/src/explainable_rl_ros/scripts/model_evaluation.py", line 23, in __init__
    self.bridge = CvBridge()
  File "/opt/ros/kinetic/lib/python2.7/dist-packages/cv_bridge/core.py", line 67, in __init__
    import cv2
ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type
```

## 解决方案
1. 新建workspace专门用于python3的编译
2. 进行`catkin config`

```
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin config --install
```
3. 拉取vision_opencv代码

```
mkdir src
cd src
git clone -b kinetic https://github.com/ros-perception/vision_opencv.git
```

4. 编译

```
cd ..
catkin build cv_bridge
```

## 问题解决

### 解决编译中出现的Boost问题

```
Errors     << cv_bridge:cmake /home/helei/catkin_py3/logs/cv_bridge/build.cmake.000.log    
CMake Error at /usr/local/share/cmake-3.16/Modules/FindPackageHandleStandardArgs.cmake:146 (message):
  Could NOT find Boost (missing: python3) (found version "1.58.0")
Call Stack (most recent call first):
  /usr/local/share/cmake-3.16/Modules/FindPackageHandleStandardArgs.cmake:393 (_FPHSA_FAILURE_MESSAGE)
  /usr/local/share/cmake-3.16/Modules/FindBoost.cmake:2165 (find_package_handle_standard_args)
  CMakeLists.txt:11 (find_package)
```

解决办法：
```
//到libboost_python-py35所在文件夹下，建立软连接
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s libboost_python-py35.so libboost_python3.so
sudo ln -s libboost_python-py35.a libboost_python3.a
```

ref: https://blog.csdn.net/qq_42138662/article/details/105677869

### libgcc_s.so.1问题

`libgcc_s.so.1 must be installed for pthread_cancel to work`

使用调试发现，问题仍然出在`import cv2`这个语句中。
在命令行中`echo $PYTHONPATH`可以发现：

```
/home/helei/catkin_py3/devel/lib/python3/dist-packages:/home/helei/catkin_ws_rl/devel/lib/python3/dist-packages:/home/helei/catkin_avoidance/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
```

也就是ros安装时候的`python2.7/dist-packages`仍然在环境中，而里面的`cv2.so`是python2.7的

解决方法：
如果以后不需要用到python2.7，则可以将2.7中的cv2.so删除
`rm /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so`

或者重命名：

```
cd /opt/ros/kinetic/lib/python2.7/dist-packages
mv cv2.so cv2_backup.so
```

然后需要的时候再重命名回来就好。