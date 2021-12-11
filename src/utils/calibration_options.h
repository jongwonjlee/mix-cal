/*
 *    Extrinsic calibration of multiple inertial sensors from in-flight data
 *    Copyright (c) 2021 Jongwon Lee (jongwon5@illinois.edu)
 *    http://www.github.com/jongwonjlee/mixcal
 * 
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
#ifndef CALIBRATION_OPTIONS_H
#define CALIBRATION_OPTIONS_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <Eigen/Dense>

#include "math.h"


using namespace std;
// using namespace ov_core;


# if defined(OV_MSCKF_VIOMANAGEROPTIONS_H)  // OpenVINS wrapper
struct CalibrationOptions : ov_msckf::VioManagerOptions {
# else      // without OpenVINS
struct CalibrationOptions {
# endif

    // ESTIMATOR ===============================

    /// Frequency (Hz) that we will simulate our inertial measurement unit
    double sim_freq_imu = 400.0;

    /// IMU noise (gyroscope and accelerometer)
            /**
     * @brief Struct of our imu noise parameters
     */
    struct NoiseManager {

        /// Gyroscope white noise (rad/s/sqrt(hz))
        double sigma_w = 1.6968e-04;

        /// Gyroscope white noise covariance
        double sigma_w_2 = pow(1.6968e-04, 2);

        /// Gyroscope random walk (rad/s^2/sqrt(hz))
        double sigma_wb = 1.9393e-05;

        /// Gyroscope random walk covariance
        double sigma_wb_2 = pow(1.9393e-05, 2);

        /// Accelerometer white noise (m/s^2/sqrt(hz))
        double sigma_a = 2.0000e-3;

        /// Accelerometer white noise covariance
        double sigma_a_2 = pow(2.0000e-3, 2);

        /// Accelerometer random walk (m/s^3/sqrt(hz))
        double sigma_ab = 3.0000e-03;

        /// Accelerometer random walk covariance
        double sigma_ab_2 = pow(3.0000e-03, 2);

        /// Nice print function of what parameters we have loaded
        void print() {
            printf("\t- gyroscope_noise_density: %.6f\n", sigma_w);
            printf("\t- accelerometer_noise_density: %.5f\n", sigma_a);
            printf("\t- gyroscope_random_walk: %.7f\n", sigma_wb);
            printf("\t- accelerometer_random_walk: %.6f\n", sigma_ab);
        }

    };
    NoiseManager imu_noises;

    /**
     * @brief This function will print out all noise parameters loaded.
     * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
     */
    void print_noise() {
        printf("NOISE PARAMETERS:\n");
        imu_noises.print();
    }

    /// newly added
    int num_imus = 1;
    int len_sequence = 500;
    int num_sequence = 1;

    // Seed for initial states pertaining to calibration (i.e. initial guess for sensor pose)
    int sim_seed_calibration = 0;

    // for run_sim_only
    int playback_speed = 1;

    bool show_report = true;
    bool show_timer = true;
    bool show_covariance = true;
    /// Map between imuid and imu extrinsics (q_ItoU, p_IinU).
    std::map<size_t,Eigen::VectorXd> imu_extrinsics;
    // Map btween imuid and gyro mislignment for each imu (R_Utog).
    int gyroscope_misalignment = 0;
    std::map<size_t,Eigen::Matrix3d> gyr_mis_extrinsics;
    
    void print_imus() {
        printf("IMU PARAMETERS:\n");
        assert((int)imu_extrinsics.size() == num_imus);
        for(int n=0; n<num_imus; n++) {
            std::cout << "imu_" << n << "_extrinsic(0:3):" << endl << imu_extrinsics.at(n).block(0,0,4,1).transpose() << std::endl;
            std::cout << "imu_" << n << "_extrinsic(4:6):" << endl << imu_extrinsics.at(n).block(4,0,3,1).transpose() << std::endl;
            Eigen::Matrix4d T_UtoI = Eigen::Matrix4d::Identity();
            T_UtoI.block(0,0,3,3) = quat_2_Rot(imu_extrinsics.at(n).block(0,0,4,1)).transpose();
            T_UtoI.block(0,3,3,1) = -T_UtoI.block(0,0,3,3)*imu_extrinsics.at(n).block(4,0,3,1);
            std::cout << "T_U" << n << "toI:" << endl << T_UtoI << std::endl << std::endl;
        }
    }
    
    double pos_offset_mu = 0;
    double pos_offset_sd = 0;
    double ori_offset_mu = 0;
    double ori_offset_sd = 0;
    double accel_transition = 0;
    double alpha_transition = 0;

    double ba_bound = 0.5;
    double bw_bound = 0.5;

    int fix_gyr_mis = 0;

    std::string filepath_csv;
    std::string filename_csv;

    /// Gravity in the global frame (i.e. should be [0, 0, 9.81] typically)
    Eigen::Vector3d gravity = {0.0, 0.0, 9.81};
};


#endif //CALIBRATION_OPTIONS_H