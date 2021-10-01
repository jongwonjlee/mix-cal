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
#ifndef EVAL_H
#define EVAL_H

#include <cmath>
#include <fstream>
#include <eigen3/Eigen/Dense>

#include "gen_noise.h"
#include "math.h"


class extrinsicError {
public:
    double pos_rmse;
    double ori_rmse;
};


void imu_extrinsics_error(const std::map<size_t,Eigen::VectorXd>& imu_extrinsics_gt, const std::map<size_t,Eigen::VectorXd>& imu_extrinsics_et, extrinsicError& result) {
    int num_sensors = imu_extrinsics_et.size();
    
    // pose difference
    std::vector<double> p_IinU_diff;
    std::vector<double> q_ItoU_diff;
    
    for (const auto& p : imu_extrinsics_et) {
        int m = p.first;     // used only to read sensor index
        Eigen::Vector3d p_IinU_et = imu_extrinsics_et.at(m).block(4,0,3,1);
        Eigen::Vector3d p_IinU_gt = imu_extrinsics_gt.at(m).block(4,0,3,1);

        Eigen::Vector4d quat_ItoU_et = imu_extrinsics_et.at(m).block(0,0,4,1);
        Eigen::Vector4d quat_ItoU_gt = imu_extrinsics_gt.at(m).block(0,0,4,1);
        Eigen::Map<Eigen::Quaterniond> q_ItoU_et(quat_ItoU_et.data());
        Eigen::Map<Eigen::Quaterniond> q_ItoU_gt(quat_ItoU_gt.data());
        
        p_IinU_diff.push_back((p_IinU_et - p_IinU_gt).array().abs().matrix().norm());
        q_ItoU_diff.push_back(abs(q_ItoU_et.angularDistance(q_ItoU_gt)) * (180/M_PI));
    }
    
    // calculate mean and standard deviation
    double pos_mse_sum = std::inner_product(p_IinU_diff.begin(), p_IinU_diff.end(), p_IinU_diff.begin(), 0.0);  // sum[X^2]
    double ori_mse_sum = std::inner_product(q_ItoU_diff.begin(), q_ItoU_diff.end(), q_ItoU_diff.begin(), 0.0);  // sum[X^2]
    
    result.pos_rmse = sqrt(pos_mse_sum / num_sensors);
    result.ori_rmse = sqrt(ori_mse_sum / num_sensors);
}


void print_imus(std::map<size_t,Eigen::VectorXd> imu_extrinsics) {
  // Print IMU poses
  printf("IMU PARAMETERS:\n");
  for(size_t n=0; n<imu_extrinsics.size(); n++) {
    std::cout << "imu_" << n << "_extrinsic(0:3):" << std::endl << imu_extrinsics.at(n).block(0,0,4,1).transpose() << std::endl;
    std::cout << "imu_" << n << "_extrinsic(4:6):" << std::endl << imu_extrinsics.at(n).block(4,0,3,1).transpose() << std::endl;
    Eigen::Matrix4d T_UtoI = Eigen::Matrix4d::Identity();
    T_UtoI.block(0,0,3,3) = quat_2_Rot(imu_extrinsics.at(n).block(0,0,4,1)).transpose();
    T_UtoI.block(0,3,3,1) = -T_UtoI.block(0,0,3,3)*imu_extrinsics.at(n).block(4,0,3,1);
    std::cout << "T_U" << n << "toI:" << std::endl << T_UtoI << std::endl << std::endl;
  }
}


void print_results(const std::map<size_t,Eigen::VectorXd>& imupose_gt, 
                   const std::map<size_t,Eigen::VectorXd>& imupose_init,
                   const std::map<size_t,Eigen::VectorXd>& imupose_estm) {
  // Print all IMU's GT, initial guess, and estimated pose, respectively 
  printf(" -- IMU EXTRINSICS (GROUND TRUTH): \n");
  print_imus(imupose_gt);
  printf(" -- IMU EXTRINSICS (INITIAL)     : \n");
  print_imus(imupose_init);
  printf(" -- IMU EXTRINSICS (ESTIMATED)   : \n");
  print_imus(imupose_estm);
}


void print_rmse(const std::map<size_t,Eigen::VectorXd>& imupose_gt, 
                  const std::map<size_t,Eigen::VectorXd>& imupose_init,
                  const std::map<size_t,Eigen::VectorXd>& imupose_estm) {
  // Print RMSE averaged over imus in concern
  extrinsicError rmse_bfore, rmse_after;
  imu_extrinsics_error(imupose_gt, imupose_init, rmse_bfore);
  imu_extrinsics_error(imupose_gt, imupose_estm, rmse_after);
  printf(" -- AVERAGE POSE RMSE ERROR : \n");
  printf("    TRANSLATION [mm] : %.4f  -->  %.4f\n", rmse_bfore.pos_rmse * 1e3, rmse_after.pos_rmse * 1e3);
  printf("    ROTATION   [deg] : %.4f  -->  %.4f\n", rmse_bfore.ori_rmse, rmse_after.ori_rmse);
}


void export_errors(const std::string filename,
                       const std::map<size_t,Eigen::VectorXd>& imu_extrinsics_gt, 
                       const std::map<size_t,Eigen::VectorXd>& imu_extrinsics_et,
                       const int consumed_time) {
  // export rotational and translational absolute errors as a file
  size_t num_imus = imu_extrinsics_et.size();

  std::ofstream f;
  if (!file_exists(filename)) {  // if no previous file exists
    f.open(filename, ios_base::out);
    for (size_t n = 0; n < num_imus; n ++) {
      f.clear();
      f << "dp_imu" << n << " dq_imu" << n << " "; // add index
    }
    f << "ms" << std::endl;
    f.close();
  }

  f.open(filename, ios_base::out | ios_base::app);  // open file
  
  for (size_t n = 0; n < num_imus; n++) {
    Eigen::Vector3d p_IinU_et = imu_extrinsics_et.at(n).block(4,0,3,1);
    Eigen::Vector3d p_IinU_gt = imu_extrinsics_gt.at(n).block(4,0,3,1);

    Eigen::Vector4d quat_ItoU_et = imu_extrinsics_et.at(n).block(0,0,4,1);
    Eigen::Vector4d quat_ItoU_gt = imu_extrinsics_gt.at(n).block(0,0,4,1);
    Eigen::Map<Eigen::Quaterniond> q_ItoU_et(quat_ItoU_et.data());
    Eigen::Map<Eigen::Quaterniond> q_ItoU_gt(quat_ItoU_gt.data());
    
    f << (p_IinU_et - p_IinU_gt).array().abs().matrix().norm() << " ";    // position error [m]
    f << abs(q_ItoU_et.angularDistance(q_ItoU_gt)) << " ";      // orientation error [rad]
  }
  f << consumed_time << std::endl;
  f.close();

}

void export_pose(const std::string filename,
                 const Eigen::VectorXd& imu_extrinsic, 
                 const int consumed_time) {
  // export (1) position of I1 w.r.t. I0 in I0 frame (^{I0}p_{I0}_{I1}) and (2) orientation axis of I1 w.r.t. I0 (^{I0}_{I1}R) as a file
  std::ofstream f;
  if (!file_exists(filename)) {  // if no previous file exists
    f.open(filename, ios_base::out);
    f.clear();
    f << "p_x p_y p_z q_x q_y q_z q_w" << " " << "ms" << std::endl; // add index
    f.close();
  }

  f.open(filename, ios_base::out | ios_base::app);  // open file
  
  Eigen::Vector3d p_UtoIinU = imu_extrinsic.block(4,0,3,1);
  Eigen::Vector4d quat_ItoU = imu_extrinsic.block(0,0,4,1);
  Eigen::Map<Eigen::Quaterniond> q_ItoU(quat_ItoU.data());
  Eigen::Matrix3d R_ItoU = q_ItoU.matrix();
  
  Eigen::Matrix3d R_UtoI = R_ItoU.transpose();
  Eigen::Vector3d p_ItoUinU = -p_UtoIinU;
  
  Eigen::Vector3d p_ItoUinI = R_UtoI * p_ItoUinU;
  Eigen::Quaterniond q_UtoI(R_UtoI);
  
  f << p_ItoUinI(0) << " " << p_ItoUinI(1) << " " << p_ItoUinI(2) << " ";    // position [m]
  f << q_UtoI.x() << " " << q_UtoI.y() << " " << q_UtoI.z() << " " << q_UtoI.w() << " ";      // orientation [quat]
  
  f << consumed_time << std::endl;
  f.close();

}


#endif /* EVAL_H */