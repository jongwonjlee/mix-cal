# Extrinsic Calibration of Multiple Inertial Sensors from Arbitrary Trajectories

This program serves the extrinsic calibration of multiple 6DOF inertial measurement unit (IMU) with arbitrary measurements. This program, wrapped with [ROS](https://www.ros.org/) as of now, requires the installation of [Ceres Solver](http://ceres-solver.org/). Some code has been adapted from the [OpenVINS](https://docs.openvins.com/), a open-sourced visual-inertial simulator.


## Dependencies

The codebase has a dependency to the following libraries and tools:
- Ceres Solver: http://ceres-solver.org/
- Eigen: https://eigen.tuxfamily.org/
- ROS: https://www.ros.org/
<!--- - OpenVINS: https://docs.openvins.com/ -->


## Dataset [[Link](https://uofi.box.com/s/nemjm0v2q05hmgzg6ewef4iwqpwh1tkd)]

The dataset consists of (1) two IMUs, (2) one camera, and (3) trajectory captured by VICON for ground-truth, under three different perceptual conditions: (1) baseline, (2) ill-lit, and (3) blurry scenes. Collected while the camera heads toward the fiducal marker, this dataset is organized to our framework's performance to [Kalibr](https://github.com/ethz-asl/kalibr), a camera-IMU calibration toolbox widely used.


## How to execute

**Disclaimer**: The command below shows an example of how to build the program and execute it by `roslaunch`. The details on running it with the aforementioned dataset will be appeared shortly.

```
# setup your own workspace
mkdir -p ${YOUR_WORKSPACE}/catkin_ws/src/
cd ${YOUR_WORKSPACE}/catkin_ws
catkin init
# repositories to clone
cd src
git clone https://github.com/jongwonjlee/mix-cal.git
# go back to root and build
cd ..
catkin build -j4
# run the calibration
source devel/setup.bash
roslaunch imucalib run_record.launch csv_filepath:="${IMU_DATA_PATH}/" csv_filename:="results.csv"
```


## Credit / Licensing

The program is licensed under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.txt) inherited from that of [OpenVINS](https://github.com/rpng/open_vins) where some part of it is adapted. If you use code in this program pertaining to [OpenVINS](https://github.com/rpng/open_vins), please cite the following:

```
@Conference{Geneva2020ICRA,
  Title      = {OpenVINS: A Research Platform for Visual-Inertial Estimation},
  Author     = {Patrick Geneva and Kevin Eckenhoff and Woosik Lee and Yulin Yang and Guoquan Huang},
  Booktitle  = {Proc. of the IEEE International Conference on Robotics and Automation},
  Year       = {2020},
  Address    = {Paris, France},
  Url        = {\url{https://github.com/rpng/open_vins}}
}
```
