#!/bin/bash

catkin_make -DCMAKE_BUILD_TYPE=Debug && source devel/setup.bash #&& roslaunch pointcloud_to_rangeimage compression.launch

