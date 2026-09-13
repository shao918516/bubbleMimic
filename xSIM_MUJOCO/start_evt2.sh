#!/bin/bash
# 彻底清空所有虚拟环境、IsaacSim残留变量
deactivate 2>/dev/null
unset VIRTUAL_ENV
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH

# 加载纯净ROS Humble
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0

# 仅添加MuJoCo仿真代码目录，不引入isaac库路径
export PYTHONPATH="/home/felix/isaac/xMimic/xSIM_MUJOCO"

# 使用系统python3.10运行，完全隔离env_isaaclab
python3 scripts/simulator_view_asyn.py -m evt2

