#!/bin/bash

while true; do
    rosbag play --clock -r 0.1 /catkin_ws/rosbags/evaluation_frames.bag
done

