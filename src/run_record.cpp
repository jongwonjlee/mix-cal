/*
 *    Extrinsic calibration of multiple inertial sensors from in-flight data
 *    Copyright (c) 2021 Jongwon Lee (jongwon5@illinois.edu)
 *    http://www.github.com/jongwonjlee/mixcal
 *    This program is free software: you can redistribute it and/or modify 
 *    it under the terms of the GNU General Public License as published by 
 *    the Free Software Foundation, either version 3 of the License, or 
 *    (at your option) any later version. 
 * 
 *    This program is distributed in the hope that it will be useful, 
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of 
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 *    GNU General Public License for more details. 
 * 
 *    You should have received a copy of the GNU General Public License 
 *    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <iomanip>   // for setiosflags
#include <csignal>
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <fstream>

#include "residuals.h"
#include "estimator.h"
#include "utils/parse_ros.h"
#include "utils/calibration_options.h"
#include "utils/utils.h"
#include "utils/read_imu_csv.h"
#include "utils/eval.h"

// Main function
int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    // Read in our parameters
    CalibrationOptions params;
    ros::init(argc, argv, "run_record");
    ros::NodeHandle nh("~");
    params = parse_ros_nodehandler(nh);
    
    // Step through the rosbag
    signal(SIGINT, signal_callback_handler);

    std::vector<std::map<size_t, Eigen::Vector3d> > am_arr, wm_arr;

    // ##### FROM HERE ADDED BLOCKS
    // load data
    std::string filename_imu0 = params.filepath_csv + "imu0.csv";
    std::string filename_imu1 = params.filepath_csv + "imu1.csv";
    std::vector<Eigen::Vector3d> am_imu0, am_imu1, wm_imu0, wm_imu1;
    read_imu_csv(filename_imu0, am_imu0, wm_imu0);
    read_imu_csv(filename_imu1, am_imu1, wm_imu1);
    printf("Read csv done.\n");

    // crop data
    if (params.len_sequence != -1) {
        am_imu0.resize(params.len_sequence);
        wm_imu0.resize(params.len_sequence);
        am_imu1.resize(params.len_sequence);
        wm_imu1.resize(params.len_sequence);
    }
    
    // create arbitrary map data
    std::map<size_t, std::vector<Eigen::Vector3d> > am_map, wm_map;
    am_map.insert({0, am_imu0});
    am_map.insert({1, am_imu1});
    wm_map.insert({0, wm_imu0});
    wm_map.insert({1, wm_imu1});

    // swap data
    swap_imu_readings(am_map, am_arr);
    swap_imu_readings(wm_map, wm_arr);

    int count = am_arr.size();
    printf("[COUNT]: %d \n", count);

    // ##### HERE ADDED BLOCKS END

    // Initialize pose information for accelerometers
    std::map<size_t,Eigen::VectorXd> imu_pose_initial = params.imu_extrinsics;
    for (int n = 1; n < params.num_imus; n ++) {
        Eigen::VectorXd zero_vec(7); zero_vec << 1,0,0,0,0,0,0;
        imu_pose_initial.at(n) = zero_vec;
        // add_noise(imu_pose_initial.at(n), params.pos_offset_mu, params.pos_offset_sd, params.ori_offset_mu, params.ori_offset_sd);
    }

    // Initialize wd_inB_init, an initial guess for angular acceleration seen by base IMU
    std::vector<Eigen::Vector3d> wd_inI_init(count);
    for (int t = 1; t < count-1; t ++) wd_inI_init.at(t) = 0.5 * (wm_arr.at(t+1).at(0) - wm_arr.at(t-1).at(0)) * params.sim_freq_imu;
    
    // Container for pose estimation
    std::map<size_t,Eigen::VectorXd> imu_pose_estimated;

    // Create estimator
    Ours::Estimator estimator(params, imu_pose_initial);
    estimator.feed_init(wd_inI_init);
    // Do calibration
    estimator.construct_problem(am_arr, wm_arr);
    estimator.solve_problem();
    estimator.get_extrinsics(imu_pose_estimated);
    estimator.show_results();
    
    // Print all pose estimation result
    print_results(params.imu_extrinsics, imu_pose_initial, imu_pose_estimated);

    // export estimated pose to file
    std::string export_filename = params.filepath_csv + params.filename_csv;
    int time_count_ms = estimator.print_timer();
    export_pose(export_filename, imu_pose_estimated.at(1), time_count_ms);
    
    // erase A0 pose information
    imu_pose_initial.erase(0);
    imu_pose_estimated.erase(0);
    params.imu_extrinsics.erase(0);
    
    // Print estimation error
    print_rmse(params.imu_extrinsics, imu_pose_initial, imu_pose_estimated);
    
    // Done!
    return EXIT_SUCCESS;

}
