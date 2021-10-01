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
#ifndef GEN_NOISE_H
#define GEN_NOISE_H

#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Geometry>

template <typename T>
T norm(std::vector<T> v) {
  return std::sqrt(pow(v.at(0), 2) + pow(v.at(1), 2) + pow(v.at(2), 2));
}


template <typename T>
std::vector<T> create_random_vector() {
  static std::random_device rd;
  static std::mt19937 rg(rd());
  
  static std::uniform_real_distribution<T> distr(-1000, 1000);
  
  std::vector<T> random_vec;
  
  for(size_t i = 0; i < 3; i ++) {
    random_vec.emplace_back(distr(rg));
  }

  return random_vec;

}


template <typename T>
std::vector<T> create_random_unit_vector() {
  std::vector<T> v = create_random_vector<T>();
  constexpr double eps = 0.01;
  T m = norm<T>(v);
  
  if (m > eps) {
    typename std::vector<T>::iterator it;
    for(it = v.begin(); it != v.end(); it ++ ) *it = *it/m;
    return v;
  }
  else {
    return create_random_unit_vector<T>();
  }
}


Eigen::Vector3d create_pos_noise(double mean, double stdv) {
  std::vector<double> v = create_random_unit_vector<double>();
  Eigen::Vector3d pos_noise{v.at(0), v.at(1), v.at(2)};

  std::random_device rd;
  std::mt19937 rg(rd());
  
  std::normal_distribution<double> dist(mean, stdv);
  double offset = dist(rg);

  return pos_noise * offset;
}


Eigen::Quaterniond create_ori_noise(double mean, double stdv) {
  std::vector<double> v = create_random_unit_vector<double>();
  Eigen::Quaterniond ori_noise;

  std::random_device rd;
  std::mt19937 rg(rd());
  
  std::normal_distribution<double> dist(mean, stdv);
  double offset = dist(rg);

  ori_noise.w() = cos(offset/2);
  ori_noise.x() = sin(offset/2) * v.at(0);
  ori_noise.y() = sin(offset/2) * v.at(1);
  ori_noise.z() = sin(offset/2) * v.at(2);

  return ori_noise;
}


Eigen::Matrix3d add_noise(const Eigen::Matrix3d& R_raw, double noise_angle) {
    Eigen::Quaterniond q_raw(R_raw);
    Eigen::Quaterniond q_noise;
    
    // Calculate direction vector of q_raw
    Eigen::Vector3d direction;
    if (fabs(q_raw.x()) <= 0.001 && fabs(q_raw.y()) <= 0.001 && fabs(q_raw.z()) <= 0.001) {
        direction.setOnes();
    } else {
        direction(0) = q_raw.x();
        direction(1) = q_raw.y();
        direction(2) = q_raw.z();
    }
    direction.normalize();
    
    // Create q_noise
    q_noise.w() = cos(noise_angle/2);
    q_noise.x() = direction(0) * sin(noise_angle/2);
    q_noise.y() = direction(1) * sin(noise_angle/2);
    q_noise.z() = direction(2) * sin(noise_angle/2);
    
    // Calculate noised q_raw
    Eigen::Matrix3d R;
    Eigen::Matrix3d R_noise(q_noise);
    R = R_noise * R_raw;

    return R;
}


void add_noise(Eigen::VectorXd& imu_extrinsic, const double pos_mu, const double pos_sd, const double ori_mu, const double ori_sd) {
    // add noise to imu orientation
    Eigen::Quaterniond ori_noise = create_ori_noise(ori_mu * M_PI/180, ori_sd * M_PI/180);
    Eigen::Vector4d quat_ItoU = imu_extrinsic.block(0,0,4,1);
    Eigen::Map<Eigen::Quaterniond> q_ItoU(quat_ItoU.data());
    q_ItoU = ori_noise * q_ItoU;
    Eigen::Map<Eigen::Vector4d> quat_ItoU_result(q_ItoU.coeffs().data());
    imu_extrinsic.block(0,0,4,1) = quat_ItoU_result;

    // add noise to imu position
    Eigen::Vector3d pos_noise = create_pos_noise(pos_mu, pos_sd);
    Eigen::Vector3d p_IinU = imu_extrinsic.block(4,0,3,1);
    p_IinU = p_IinU + pos_noise;
    imu_extrinsic.block(4,0,3,1) = p_IinU;
}


#endif /* GEN_NOISE_H */