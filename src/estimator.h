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
#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <assert.h>
#include <vector>
#include <map>
#include <eigen3/Eigen/Dense>
// timer
#include <algorithm>
#include <chrono>

#include "utils/calibration_options.h"

/* SUGGESTED SELF-CALIBRATION WITH BIAS ESTIMATION */
namespace Ours {

class Estimator
{
protected:
    /// True vio manager params (a copy of the parsed ones)
    CalibrationOptions params;

    // Parameter blocks
    std::map<size_t, Eigen::Vector3d> imu_pos;
    std::map<size_t, Eigen::Quaterniond> imu_ori;
    std::map<size_t, Eigen::Quaterniond> gyr_mis;   // gyroscope misalignment
    std::vector<Eigen::Vector3d> wd_inI;
    std::vector<std::map<size_t, Eigen::Vector3d> > acc_bias_arr;
    std::vector<std::map<size_t, Eigen::Vector3d> > gyr_bias_arr;

    // Define sampling time interval
    double dt;
    
    // Define variances for IMU bias and noise
    double var_a, var_w, var_ab, var_wb;

    // Define covariance matrices having above as diagonal components
    Eigen::Matrix3d Cov_a, Cov_w, Cov_ab, Cov_wb;
    std::map<size_t, Eigen::Matrix3d> Cov_a_arr, Cov_w_arr, Cov_ab_arr, Cov_wb_arr;

    // Define local parameterization to optimize quaternion on its own manifold (i.e. q \in R^4 where \norm(q) = 1)
    ceres::LocalParameterization* quat_loc_parameterization = new ceres::EigenQuaternionParameterization;

    // ceres::Problem object
    ceres::Problem problem;
    Solver::Options options;
    Solver::Summary summary;

    // Define timer
    std::chrono::time_point<std::chrono::high_resolution_clock> tic;
    std::chrono::time_point<std::chrono::high_resolution_clock> toc;
    
    // cost function value before and after optimization
    double initial_cost;
    double final_cost;

    // number of data to be proccessed
    int num_data;

public:
    Estimator(const CalibrationOptions& params,
              const std::map<size_t, Eigen::VectorXd>& imu_extrinsics);
    ~Estimator();
    void feed_init(const std::vector<Eigen::Vector3d>& wd_inI);
    void feed_bias(const std::vector<std::map<size_t, Eigen::Vector3d> >& acc_bias_arr,
                   const std::vector<std::map<size_t, Eigen::Vector3d> >& gyr_bias_arr);
    void construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements,
                           const std::vector<std::map<size_t, Eigen::Vector3d> >& w_measurements);
    void solve_problem();
    void get_extrinsics(std::map<size_t, Eigen::VectorXd>& imu_extrinsics_estimated);
    void get_gyr_mis(std::map<size_t, Eigen::Quaterniond>& gyr_mis_estimated);
    void show_results();

    // Internal member functions
    std::pair<double, double> print_covariance();
    int print_timer();
    void print_report();
};

Estimator::Estimator(const CalibrationOptions& params, const std::map<size_t, Eigen::VectorXd>& imu_extrinsics)
{
    // Store a copy of our params
    this->params = params;

    // Define sampling time interval
    dt = 1 / params.sim_freq_imu;
    // Define variances for IMU bias and noise
    var_a = pow(params.imu_noises.sigma_a, 2) / dt;
    var_w = pow(params.imu_noises.sigma_w, 2) / dt;
    var_ab = dt * pow(params.imu_noises.sigma_ab, 2);
    var_wb = dt * pow(params.imu_noises.sigma_wb, 2);

    for (int k = 0; k < 3; k ++) {
        Cov_a(k,k) = var_a;
        Cov_w(k,k) = var_w;
        Cov_ab(k,k) = var_ab;
        Cov_wb(k,k) = var_wb;
    }

    // ceres options
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 200;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = true;

    // Initialize imu_pos, imu_ori, and gyr_mis
    for (int n = 0; n < params.num_imus; n ++) {
        Eigen::Matrix3d R_BtoIn = quat_2_Rot(imu_extrinsics.at(n).block(0,0,4,1));
        Eigen::Quaterniond q_BtoIn(R_BtoIn);
        Eigen::Vector3d p_BinIn = imu_extrinsics.at(n).block(4,0,3,1);

        imu_pos.insert({n, p_BinIn});
        imu_ori.insert({n, q_BtoIn});

        Eigen::Quaterniond q_gyr = Eigen::Quaterniond::Identity();  // q_IntoGn
        gyr_mis.insert({n, q_gyr});
    }
    
}

Estimator::~Estimator() {}

void Estimator::feed_init(const std::vector<Eigen::Vector3d>& wd_inI) {
    // Initialize wd_inI
    this->wd_inI = wd_inI;

    // declare number of data used for construction
    num_data = this->wd_inI.size();

    // Initialize biases
    std::map<size_t,Eigen::Vector3d> bias_zeros;
    for (int n = 0; n < params.num_imus; n ++) bias_zeros.insert({n, Eigen::Vector3d::Zero()});
    acc_bias_arr.resize(num_data, bias_zeros);
    gyr_bias_arr.resize(num_data, bias_zeros);
}


void Estimator::feed_bias(const std::vector<std::map<size_t, Eigen::Vector3d> >& acc_bias_arr,
                          const std::vector<std::map<size_t, Eigen::Vector3d> >& gyr_bias_arr) {
    this->acc_bias_arr = acc_bias_arr;
    this->gyr_bias_arr = gyr_bias_arr;
}

void Estimator::construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements,
                                  const std::vector<std::map<size_t, Eigen::Vector3d> >& w_measurements) {
    // assert data length consistency
    assert((int)(a_measurements.size()) == num_data && (int)(w_measurements.size()) == num_data);

    // STEP 1-1 : ADD PARAMATERS TO BE OPTIMIZED
    // add all parameters in concern
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddParameterBlock(imu_pos.at(n).data(), 3);
        problem.AddParameterBlock(imu_ori.at(n).coeffs().data(), 4, quat_loc_parameterization);
        problem.AddParameterBlock(gyr_mis.at(n).coeffs().data(), 4, quat_loc_parameterization);
    }
    for (int t = 0; t < num_data; t ++) {
        problem.AddParameterBlock(wd_inI.at(t).data(), 3);
        for (int n = 0; n < params.num_imus; n ++) {
            problem.AddParameterBlock(acc_bias_arr.at(t).at(n).data(), 3);
            problem.AddParameterBlock(gyr_bias_arr.at(t).at(n).data(), 3);
        }
    }

    // STEP 1-2 : FIX CONSTANT PARAMATERS
    // fix base IMU pose
    problem.SetParameterBlockConstant(imu_pos.at(0).data());
    problem.SetParameterBlockConstant(imu_ori.at(0).coeffs().data());

    // fix gyroscope axis to be identical to that of imu if not applicable
    if (params.gyroscope_misalignment == 0) {
        for (int n = 0; n < params.num_imus; n ++) problem.SetParameterBlockConstant(gyr_mis.at(n).coeffs().data());
    }
    
    // STEP 1-3 : SET BOUNDS OF PARAMETERS
    // bound bias
    for (int t = 0; t < num_data; t ++) {
        for (int n = 0; n < params.num_imus; n ++) {
            for (int i = 0; i < 3; i ++) {
                problem.SetParameterLowerBound(acc_bias_arr.at(t).at(n).data(), i, -params.ba_bound);
                problem.SetParameterUpperBound(acc_bias_arr.at(t).at(n).data(), i, +params.ba_bound);
                problem.SetParameterLowerBound(gyr_bias_arr.at(t).at(n).data(), i, -params.bw_bound);
                problem.SetParameterUpperBound(gyr_bias_arr.at(t).at(n).data(), i, +params.bw_bound);
            }
        }
    }
    
    // STEP 2 : CONSTRUCT FACTORS
    for (int n = 1; n < params.num_imus; n ++) {
        for (int t = 0; t < num_data; t ++) {
            // Construct each residual's noise covariance matrices
            Eigen::Matrix3d Cov_acc = 2 * Cov_a + Cov_w * Cov_w;
            Eigen::Matrix3d Cov_gyr = 2 * Cov_w;
            Eigen::Matrix<double,3,3> sqrt_Info_acc = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_acc.inverse()).matrixL().transpose();
            Eigen::Matrix<double,3,3> sqrt_Info_gyr = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_gyr.inverse()).matrixL().transpose();

            // Add residual
            ceres::CostFunction* cost_acc = Ours::AcclResidual::Create(dt, a_measurements.at(t).at(n), w_measurements.at(t).at(0), a_measurements.at(t).at(0), sqrt_Info_acc);
            ceres::CostFunction* cost_gyr = Ours::AngvelResidual::Create(w_measurements.at(t).at(n), w_measurements.at(t).at(0), sqrt_Info_gyr);

            // Assign variable (subject to be optimized)
            problem.AddResidualBlock(cost_acc, new ceres::CauchyLoss(1.0), 
                                     imu_pos.at(n).data(), imu_ori.at(n).coeffs().data(), gyr_mis.at(0).coeffs().data(), wd_inI.at(t).data(),
                                     acc_bias_arr.at(t).at(n).data(), gyr_bias_arr.at(t).at(0).data(), acc_bias_arr.at(t).at(0).data());
            problem.AddResidualBlock(cost_gyr, new ceres::CauchyLoss(1.0), 
                                     imu_ori.at(n).coeffs().data(), gyr_mis.at(n).coeffs().data(), gyr_mis.at(0).coeffs().data(), 
                                     gyr_bias_arr.at(t).at(n).data(), gyr_bias_arr.at(t).at(0).data());
        }
    }

    // bias evolution
    for (int n = 0; n < params.num_imus; n ++) {
        for (int t = 0; t < num_data-1; t ++) {
            // biases
            Eigen::Matrix<double,3,3> sqrt_Info_ab = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_ab.inverse()).matrixL().transpose();
            Eigen::Matrix<double,3,3> sqrt_Info_wb = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_wb.inverse()).matrixL().transpose();

            problem.AddResidualBlock(Ours::BiasResidual::Create(sqrt_Info_ab), new ceres::CauchyLoss(1.0), 
                                     acc_bias_arr.at(t).at(n).data(), acc_bias_arr.at(t+1).at(n).data());
            problem.AddResidualBlock(Ours::BiasResidual::Create(sqrt_Info_wb), new ceres::CauchyLoss(1.0), 
                                     gyr_bias_arr.at(t).at(n).data(), gyr_bias_arr.at(t+1).at(n).data());
        }
    }
    
    /*
    // anchor the very first bias to avoid rank deficiency while calculating covariance matrices
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, acc_bias_arr.at(0).at(n).data());
        problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, gyr_bias_arr.at(0).at(n).data());
    }
    */
}

void Estimator::solve_problem()
{
    tic = std::chrono::high_resolution_clock::now();
    
    // Run the solver
    Solve(options, &problem, &summary);

    // Record initial and final cost
    initial_cost = summary.initial_cost;
    final_cost = summary.final_cost;
    summary.FullReport();
        
    toc = std::chrono::high_resolution_clock::now();
}

void Estimator::get_extrinsics(std::map<size_t, Eigen::VectorXd>& imu_extrinsics_estimated)
{
    // Export imu_ori and imu_pos as imu_extrinsics_estimated
    for (int n = 0; n < params.num_imus; n ++) {
        Eigen::VectorXd imu_eigen(7);
        // read estimated extrinsics (w.r.t. I); p_BinIn, q_BtoIn
        imu_eigen.block(0,0,4,1) = imu_ori.at(n).coeffs();
        imu_eigen.block(4,0,3,1) = imu_pos.at(n);
        imu_extrinsics_estimated.insert({n, imu_eigen});
    }
}

void Estimator::get_gyr_mis(std::map<size_t, Eigen::Quaterniond>& gyr_mis_estimated)
{
    gyr_mis_estimated = gyr_mis;
}

void Estimator::show_results()
{
    if (params.show_report) print_report();
    if (params.show_timer) print_timer();
    if (params.show_covariance) print_covariance();
}

std::pair<double, double> Estimator::print_covariance()
{
    Eigen::Matrix<double,3,3,Eigen::RowMajor> cov_pos = Eigen::Matrix<double,3,3,Eigen::RowMajor>::Zero();
    Eigen::Matrix<double,4,4,Eigen::RowMajor> cov_ori = Eigen::Matrix<double,4,4,Eigen::RowMajor>::Zero();

    ceres::Covariance::Options cov_options;
    ceres::Covariance covariance(cov_options);
    
    std::vector<std::pair<const double*, const double*> > covariance_blocks;

    double imu_pos_trace = 0;
    double imu_ori_trace = 0;

    for (int n = 1; n < params.num_imus; n ++) {
      covariance_blocks.push_back(std::make_pair(imu_pos.at(n).data(), imu_pos.at(n).data()));
      covariance_blocks.push_back(std::make_pair(imu_ori.at(n).coeffs().data(), imu_ori.at(n).coeffs().data()));
      covariance.Compute(covariance_blocks, &problem);
      
      covariance.GetCovarianceBlock(imu_pos.at(n).data(), imu_pos.at(n).data(), cov_pos.data());
      covariance.GetCovarianceBlock(imu_ori.at(n).coeffs().data(), imu_ori.at(n).coeffs().data(), cov_ori.data());
      
      // Since above covariance matrix is represented in quaternion, it implies nothing for human.
      // Therefore, we adopt a jacobian matrix of euler angle w.r.t. quaternion such that any covariance in quaternion can be converted into euler angle.
      // C_euler = J_dEdq * C_quat * J_dEdq;
      // This may result in the covariance along yaw (psi)- pitch (theta) - roll (phi)
      // For detailed explanation, please refer to https://www.ucalgary.ca/engo_webdocs/GL/96.20096.JSchleppe.pdf
      double q1 = imu_ori.at(n).x();
      double q2 = imu_ori.at(n).y();
      double q3 = imu_ori.at(n).z();
      double q4 = imu_ori.at(n).w();

      double D1 = pow(q3 + q2, 2) + pow(q4 + q1, 2);
      double D2 = pow(q3 - q2, 2) + pow(q4 - q1, 2);
      double Dt = pow(1 - 4 * pow(q2 * q3 + q1 * q4, 2), 0.5);
      Eigen::Matrix<double,3,4> J_dEdq;
      J_dEdq << -(q3 + q2)/D1+(q3-q2)/D2,  (q4+q1)/D1-(q4-q1)/D2,  (q4+q1)/D1-(q4-q1)/D2, -(q3+q2)/D1+(q3-q2)/D2,
                2*q4/Dt, 2*q3/Dt, 2*q2/Dt, 2*q1/Dt,
                -(q3 + q2)/D1+(q3-q2)/D2,  (q4+q1)/D1-(q4-q1)/D2,  (q4+q1)/D1-(q4-q1)/D2, -(q3+q2)/D1+(q3-q2)/D2;
      
      Eigen::Matrix3d cov_ori_3x3 = J_dEdq * cov_ori * J_dEdq.transpose();

      imu_pos_trace += cov_pos.trace();
      imu_ori_trace += cov_ori_3x3.trace();
    }

    printf(" -- COVARIANCE OF ESTIMATION: %.3g\n", imu_pos_trace + imu_ori_trace);

    return std::make_pair(imu_pos_trace, imu_ori_trace);
}

int Estimator::print_timer()
{
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    std::cout << "Time taken by function: " << duration.count() << " [ms]" << std::endl;
    return duration.count();
}

void Estimator::print_report()
{
    // Report optimization process
    std::cout << summary.FullReport() << std::endl << std::endl;
}

}

// SUGGESTED SELF-CALIBRATION WITH CONSTANT BIAS ESTIMATION
namespace OursConstBias {

class Estimator : public Ours::Estimator
{
protected:
    // parameter blocks
    std::map<size_t, Eigen::Vector3d> acc_bias, gyr_bias;

public:
    Estimator(const CalibrationOptions& params,
              const std::map<size_t, Eigen::VectorXd>& imu_extrinsics);
    ~Estimator();
    void feed_init(const std::vector<Eigen::Vector3d>& wd_inI);
    void construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements,
                           const std::vector<std::map<size_t, Eigen::Vector3d> >& w_measurements);
    
    // For Schneider et al.
    void get_biases(std::map<size_t, Eigen::Vector3d>& acc_bias_estimated,
                    std::map<size_t, Eigen::Vector3d>& gyr_bias_estimated) {
        acc_bias_estimated = acc_bias;
        gyr_bias_estimated = gyr_bias;
    }
    void get_wd_inB(std::vector<Eigen::Vector3d>& wd_inI_estimated) {
        wd_inI_estimated = wd_inI;
    };
};

Estimator::Estimator(const CalibrationOptions& params, const std::map<size_t, Eigen::VectorXd>& imu_extrinsics)
                     : Ours::Estimator::Estimator(params, imu_extrinsics) {}

Estimator::~Estimator() {}

void Estimator::feed_init(const std::vector<Eigen::Vector3d>& wd_inI) {
    // Initialize wd_inI
    this->wd_inI = wd_inI;

    // declare number of data used for construction
    num_data = this->wd_inI.size();

    // Initialize biases
    for (int n = 0; n < params.num_imus; n ++) {
        acc_bias.insert({n, Eigen::Vector3d::Zero()});
        gyr_bias.insert({n, Eigen::Vector3d::Zero()});
    }
}

void Estimator::construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements,
                                  const std::vector<std::map<size_t, Eigen::Vector3d> >& w_measurements) {
    // assert data length consistency
    assert((int)(a_measurements.size()) == num_data && (int)(w_measurements.size()) == num_data);

    // STEP 1 : ADD PARAMATERS TO BE OPTIMIZED
    // add all parameters in concern
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddParameterBlock(imu_pos.at(n).data(), 3);
        problem.AddParameterBlock(imu_ori.at(n).coeffs().data(), 4, quat_loc_parameterization);
        problem.AddParameterBlock(gyr_mis.at(n).coeffs().data(), 4, quat_loc_parameterization);
        problem.AddParameterBlock(acc_bias.at(n).data(), 3);
        problem.AddParameterBlock(gyr_bias.at(n).data(), 3);
    }
    for (int t = 0; t < num_data; t ++) problem.AddParameterBlock(wd_inI.at(t).data(), 3);

    // fix base IMU pose
    problem.SetParameterBlockConstant(imu_pos.at(0).data());
    problem.SetParameterBlockConstant(imu_ori.at(0).coeffs().data());
    
    // fix gyroscope axis to be identical to that of imu if not applicable
    if (params.gyroscope_misalignment == 0 || params.fix_gyr_mis) {
        for (int n = 0; n < params.num_imus; n ++) problem.SetParameterBlockConstant(gyr_mis.at(n).coeffs().data());
    }

    // STEP 2 : CONSTRUCT FACTORS
    for (int n = 1; n < params.num_imus; n ++) {
        for (int t = 0; t < num_data-1; t ++) {
            // Construct each residual's noise covariance matrices
            Eigen::Matrix3d Cov_acc = 2 * (Cov_a + t * Cov_ab) + (Cov_w + t * Cov_wb) * (Cov_w + t * Cov_wb);
            Eigen::Matrix3d Cov_gyr = 2 * (Cov_w + t * Cov_wb);
            Eigen::Matrix<double,3,3> sqrt_Info_acc = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_acc.inverse()).matrixL().transpose();
            Eigen::Matrix<double,3,3> sqrt_Info_gyr = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_gyr.inverse()).matrixL().transpose();
            
            // Add residual
            ceres::CostFunction* cost_acc = Ours::AcclResidual::Create(dt, a_measurements.at(t).at(n), w_measurements.at(t).at(n), a_measurements.at(t).at(0), sqrt_Info_acc);
            ceres::CostFunction* cost_gyr = Ours::AngvelResidual::Create(w_measurements.at(t).at(n), w_measurements.at(t).at(0), sqrt_Info_gyr);

            // Assign variable (subject to be optimized)
            problem.AddResidualBlock(cost_acc, NULL, imu_pos.at(n).data(), imu_ori.at(n).coeffs().data(), gyr_mis.at(0).coeffs().data(), wd_inI.at(t).data(),
                                                     acc_bias.at(n).data(), gyr_bias.at(0).data(), acc_bias.at(0).data());
            problem.AddResidualBlock(cost_gyr, NULL, imu_ori.at(n).coeffs().data(), gyr_mis.at(n).coeffs().data(), gyr_mis.at(0).coeffs().data(), 
                                                     gyr_bias.at(n).data(), gyr_bias.at(0).data());
        }
    }
    
    // anchor biases
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, acc_bias.at(n).data());
        problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, gyr_bias.at(n).data());
    }
}

}

// SUGGESTED SELF-CALIBRATION AS INTRODUCED IN SCHNEIDER ET AL.
namespace Schneider {

class Estimator : public Ours::Estimator {
public:
    Estimator(const CalibrationOptions& params,
              const std::map<size_t, Eigen::VectorXd>& imu_extrinsics);
    ~Estimator();
    void feed_init(const std::map<size_t, std::vector<Eigen::Vector3d> >& wd_inI,
                   const std::map<size_t, std::map<size_t, Eigen::Vector3d> >& acc_bias_map,
                   const std::map<size_t, std::map<size_t, Eigen::Vector3d> >& gyr_bias_map);
    void construct_problem(const std::map<size_t, std::vector<std::map<size_t, Eigen::Vector3d> > >& a_measurements,
                           const std::map<size_t, std::vector<std::map<size_t, Eigen::Vector3d> > >& w_measurements);
    
protected:
    // Parameter blocks
    std::map<size_t, std::vector<Eigen::Vector3d> > wd_inI_arr;
    std::map<size_t, std::map<size_t, Eigen::Vector3d> > acc_bias_map, gyr_bias_map;

    int num_segments;
};

Estimator::Estimator(const CalibrationOptions& params, const std::map<size_t, Eigen::VectorXd>& imu_extrinsics)
                     : Ours::Estimator::Estimator(params, imu_extrinsics) {}

Estimator::~Estimator() {}

void Estimator::feed_init(const std::map<size_t, std::vector<Eigen::Vector3d> >& wd_inI,
                          const std::map<size_t, std::map<size_t, Eigen::Vector3d> >& acc_bias_map,
                          const std::map<size_t, std::map<size_t, Eigen::Vector3d> >& gyr_bias_map) {
    // assert consistency of the number of sensors
    for (const auto& kv : acc_bias_map) {
        size_t k = kv.first;
        assert(acc_bias_map.at(k).size() == gyr_bias_map.at(k).size());
    }

    // Initialize wd_inI_arr
    this->wd_inI_arr = wd_inI_arr;

    // declare number of data used for construction
    num_segments = wd_inI_arr.size();

    // Initialize biases
    this->acc_bias_map = acc_bias_map;
    this->gyr_bias_map = gyr_bias_map;
}

void Estimator::construct_problem(const std::map<size_t, std::vector<std::map<size_t, Eigen::Vector3d> > >& a_measurements,
                                  const std::map<size_t, std::vector<std::map<size_t, Eigen::Vector3d> > >& w_measurements) {
    // assert data length consistency
    assert((int)(a_measurements.size()) == num_segments && (int)(w_measurements.size()) == num_segments);
    
    // STEP 1 : ADD PARAMATERS TO BE OPTIMIZED
    // add all parameters in concern
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddParameterBlock(imu_pos.at(n).data(), 3);
        problem.AddParameterBlock(imu_ori.at(n).coeffs().data(), 4, quat_loc_parameterization);
        problem.AddParameterBlock(gyr_mis.at(n).coeffs().data(), 4, quat_loc_parameterization);
    }
    for (const auto& kv : wd_inI_arr) {
        size_t k = kv.first;
        num_data = (int)(wd_inI_arr.at(k).size());
        for (int n = 0; n < params.num_imus; n ++) {
            problem.AddParameterBlock(acc_bias_map.at(k).at(n).data(), 3);
            problem.AddParameterBlock(gyr_bias_map.at(k).at(n).data(), 3);
        }
        for (int t = 0; t < num_data; t ++) problem.AddParameterBlock(wd_inI_arr.at(k).at(t).data(), 3);
    }

    // fix base IMU pose
    problem.SetParameterBlockConstant(imu_pos.at(0).data());
    problem.SetParameterBlockConstant(imu_ori.at(0).coeffs().data());

    // fix gyroscope axis to be identical to that of imu if not applicable
    if (params.gyroscope_misalignment == 0) {
        for (int n = 0; n < params.num_imus; n ++) problem.SetParameterBlockConstant(gyr_mis.at(n).coeffs().data());
    }

    // STEP 2 : CONSTRUCT FACTORS
    for (const auto& kv : wd_inI_arr) {
        size_t k = kv.first;
        num_data = (int)(wd_inI_arr.at(k).size());

        for (int n = 1; n < params.num_imus; n ++) {
            for (int t = 0; t < num_data-1; t ++) {
                // Construct each residual's noise covariance matrices
                Eigen::Matrix3d Cov_acc = 2 * (Cov_a + t * Cov_ab) + (Cov_w + t * Cov_wb) * (Cov_w + t * Cov_wb);
                Eigen::Matrix3d Cov_gyr = 2 * (Cov_w + t * Cov_wb);
                Eigen::Matrix<double,3,3> sqrt_Info_acc = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_acc.inverse()).matrixL().transpose();
                Eigen::Matrix<double,3,3> sqrt_Info_gyr = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov_gyr.inverse()).matrixL().transpose();
                
                // Add residual
                ceres::CostFunction* cost_acc = Ours::AcclResidual::Create(dt, a_measurements.at(k).at(t).at(n), w_measurements.at(k).at(t).at(n), a_measurements.at(k).at(t).at(0), sqrt_Info_acc);
                ceres::CostFunction* cost_gyr = Ours::AngvelResidual::Create(w_measurements.at(k).at(t).at(n), w_measurements.at(k).at(t).at(0), sqrt_Info_gyr);

                problem.AddResidualBlock(cost_acc, NULL, imu_pos.at(n).data(), imu_ori.at(n).coeffs().data(), gyr_mis.at(0).coeffs().data(), wd_inI_arr.at(k).at(t).data(),
                                                          acc_bias_map.at(k).at(n).data(), gyr_bias_map.at(k).at(0).data(), acc_bias_map.at(k).at(0).data());
                problem.AddResidualBlock(cost_gyr, NULL, imu_ori.at(n).coeffs().data(), gyr_mis.at(n).coeffs().data(), gyr_mis.at(0).coeffs().data(), 
                                                          gyr_bias_map.at(k).at(n).data(), gyr_bias_map.at(k).at(0).data());
            }
        }
        
        // anchor biases
        for (int n = 0; n < params.num_imus; n ++) {
            problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, acc_bias_map.at(k).at(n).data());
            problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, gyr_bias_map.at(k).at(n).data());
        }
    }
}

}


// SELF-CALIBRATION FROM BURGARD ET AL.
namespace Burgard {

class Estimator : public Ours::Estimator {

public:
    Estimator(const CalibrationOptions& params,
              const std::map<size_t, Eigen::VectorXd>& imu_extrinsics);
    ~Estimator();
    void feed_init(const std::vector<Eigen::Vector3d>& f_inI,
                   const std::vector<Eigen::Vector3d>& w_inI,
                   const std::vector<Eigen::Vector3d>& wd_inI);
    void construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements);

protected:
    // Parameter blocks
    std::vector<Eigen::Vector3d> f_inI;
    std::vector<Eigen::Vector3d> w_inI;
    std::vector<Eigen::Vector3d> wd_inI;
};

Estimator::Estimator(const CalibrationOptions& params, const std::map<size_t, Eigen::VectorXd>& imu_extrinsics)
                     : Ours::Estimator::Estimator(params, imu_extrinsics) {}

Estimator::~Estimator() {}

void Estimator::feed_init(const std::vector<Eigen::Vector3d>& f_inI,
                          const std::vector<Eigen::Vector3d>& w_inI,
                          const std::vector<Eigen::Vector3d>& wd_inI) {
    // assert data length consistency
    assert(f_inI.size() == w_inI.size() && w_inI.size() == wd_inI.size());

    // Copy initial guess of specific force, angular velocity, and angular acceleration
    this->f_inI = f_inI;
    this->w_inI = w_inI;
    this->wd_inI = wd_inI;

    // declare number of data used for construction
    num_data = f_inI.size();
}

void Estimator::construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements) 
{
    // assert data length consistency
    assert((int)(a_measurements.size()) == num_data);

    // STEP 1 : ADD PARAMATERS TO BE OPTIMIZED
    // add all parameters in concern
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddParameterBlock(imu_pos.at(n).data(), 3);
        problem.AddParameterBlock(imu_ori.at(n).coeffs().data(), 4, quat_loc_parameterization);
    }
    for (int t = 0; t < num_data; t ++) {
        problem.AddParameterBlock(f_inI.at(t).data(), 3);
        problem.AddParameterBlock(w_inI.at(t).data(), 3);
        problem.AddParameterBlock(wd_inI.at(t).data(), 3);
    }

    // fix base accelerometer pose
    problem.SetParameterBlockConstant(imu_pos.at(0).data());
    problem.SetParameterBlockConstant(imu_ori.at(0).coeffs().data());

    // STEP 2 : CONSTRUCT FACTORS
    // state observation error (i.e. spatial error)
    for (int n = 0; n < params.num_imus; n ++) {
        for (int t = 0; t < num_data; t ++) {
            // Assign measurement inputs (will remain in same)
            ceres::CostFunction* cost = Burgard::SpatialResidual::Create(a_measurements.at(t).at(n), params.imu_noises.sigma_a);
            // Assign variable (subject to be optimized): only the base pose should be obtained!
            problem.AddResidualBlock(cost, NULL, f_inI.at(t).data(), w_inI.at(t).data(), wd_inI.at(t).data(), imu_pos.at(n).data(), imu_ori.at(n).coeffs().data());
        }
    }

    // state transition error (i.e. temporal error)
    for (int t = 0; t < num_data-1; t ++) {
        ceres::CostFunction* cost = Burgard::TemporalResidual::Create(1/params.sim_freq_imu, params.accel_transition, params.alpha_transition);
        problem.AddResidualBlock(cost, NULL, f_inI.at(t).data(), f_inI.at(t+1).data(), w_inI.at(t).data(), w_inI.at(t+1).data(), wd_inI.at(t).data(), wd_inI.at(t+1).data());
    }
}

}


// SELF-CALIBRATION FROM KORTIER ET AL.
namespace Kortier {

class Estimator : public Ours::Estimator {

public:
    Estimator(const CalibrationOptions& params,
              const std::map<size_t, Eigen::VectorXd>& imu_extrinsics);
    ~Estimator();
    void feed_init(const std::vector<Eigen::Vector3d>& p_IinG,
                   const std::vector<Eigen::Vector3d>& v_IinG,
                   const std::vector<Eigen::Vector3d>& a_IinG,
                   const std::vector<Eigen::Quaterniond>& q_GtoI,
                   const std::vector<Eigen::Vector3d>& w_IinG,
                   const std::vector<Eigen::Vector3d>& wd_IinG);
    void construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements,
                           const std::vector<std::map<size_t, Eigen::Vector3d> >& w_measurements);
    
protected:
    // Parameter blocks
    std::vector<Eigen::Vector3d> p_IinG_arr;
    std::vector<Eigen::Vector3d> v_IinG_arr;
    std::vector<Eigen::Vector3d> a_IinG_arr;
    std::vector<Eigen::Quaterniond> q_GtoI_arr;
    std::vector<Eigen::Vector3d> w_IinG_arr;
    std::vector<Eigen::Vector3d> wd_IinG_arr;
    std::map<size_t, Eigen::Vector3d> acc_bias;
    std::map<size_t, Eigen::Vector3d> gyr_bias;
};

Estimator::Estimator(const CalibrationOptions& params, const std::map<size_t, Eigen::VectorXd>& imu_extrinsics)
                     : Ours::Estimator::Estimator(params, imu_extrinsics) {}

Estimator::~Estimator() {}

void Estimator::feed_init(const std::vector<Eigen::Vector3d>& p_IinG,
                          const std::vector<Eigen::Vector3d>& v_IinG,
                          const std::vector<Eigen::Vector3d>& a_IinG,
                          const std::vector<Eigen::Quaterniond>& q_GtoI,
                          const std::vector<Eigen::Vector3d>& w_IinG,
                          const std::vector<Eigen::Vector3d>& wd_IinG) {
    // assert data length consistency
    assert(p_IinG.size() == v_IinG.size() && v_IinG.size() == a_IinG.size() && a_IinG.size() == q_GtoI.size()
        && q_GtoI.size() == w_IinG.size() && w_IinG.size() == wd_IinG.size());

    // Copy initial guess of states
    this->p_IinG_arr = p_IinG;
    this->v_IinG_arr = v_IinG;
    this->a_IinG_arr = a_IinG;
    this->q_GtoI_arr = q_GtoI;
    this->w_IinG_arr = w_IinG;
    this->wd_IinG_arr = wd_IinG;

    // declare number of data used for construction
    num_data = p_IinG_arr.size();

    // Initialize biases
    for (int n = 0; n < params.num_imus; n ++) {
        acc_bias.insert({n, Eigen::Vector3d::Zero()});
        gyr_bias.insert({n, Eigen::Vector3d::Zero()});
    }
}

void Estimator::construct_problem(const std::vector<std::map<size_t, Eigen::Vector3d> >& a_measurements,
                                  const std::vector<std::map<size_t, Eigen::Vector3d> >& w_measurements) 
{
    // assert data length consistency
    assert((int)(a_measurements.size()) == num_data && (int)(w_measurements.size()) == num_data);

    // STEP 1 : ADD PARAMATERS TO BE OPTIMIZED
    // add all parameters in concern
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddParameterBlock(imu_pos.at(n).data(), 3);
        problem.AddParameterBlock(imu_ori.at(n).coeffs().data(), 4, quat_loc_parameterization);
        problem.AddParameterBlock(gyr_mis.at(n).coeffs().data(), 4, quat_loc_parameterization);
    }
    for (int t = 0; t < num_data; t ++) {
        problem.AddParameterBlock(p_IinG_arr.at(t).data(), 3);
        problem.AddParameterBlock(v_IinG_arr.at(t).data(), 3);
        problem.AddParameterBlock(a_IinG_arr.at(t).data(), 3);
        problem.AddParameterBlock(q_GtoI_arr.at(t).coeffs().data(), 4, quat_loc_parameterization);
        problem.AddParameterBlock(w_IinG_arr.at(t).data(), 3);
        problem.AddParameterBlock(wd_IinG_arr.at(t).data(), 3);
    }

    // fix base IMU pose
    problem.SetParameterBlockConstant(imu_pos.at(0).data());
    problem.SetParameterBlockConstant(imu_ori.at(0).coeffs().data());

    // fix gyroscope axis to be identical to that of imu if not applicable
    if (params.gyroscope_misalignment == 0) {
        for (int n = 0; n < params.num_imus; n ++) problem.SetParameterBlockConstant(gyr_mis.at(n).coeffs().data());
    }
    
    // STEP 2 : CONSTRUCT FACTORS
    // state transition error (i.e. temporal error)
    for (int t = 0; t < num_data-1; t ++) {
        ceres::CostFunction* trans_res = Kortier::TranslationalResidual::Create(1/params.sim_freq_imu, params.accel_transition);
        problem.AddResidualBlock(trans_res, NULL, 
                                 p_IinG_arr.at(t).data(), p_IinG_arr.at(t+1).data(), 
                                 v_IinG_arr.at(t).data(), v_IinG_arr.at(t+1).data(), 
                                 a_IinG_arr.at(t).data(), a_IinG_arr.at(t+1).data());
        
        ceres::CostFunction* rot_res = Kortier::RotationalResidual::Create(1/params.sim_freq_imu, params.alpha_transition);
        problem.AddResidualBlock(rot_res, NULL, 
                                 q_GtoI_arr.at(t).coeffs().data(), q_GtoI_arr.at(t+1).coeffs().data(), 
                                 w_IinG_arr.at(t).data(), w_IinG_arr.at(t+1).data(), 
                                 wd_IinG_arr.at(t).data(), wd_IinG_arr.at(t+1).data());
    }
    
    // measurement error
    for (int n = 0; n < params.num_imus; n ++) {
        for (int t = 0; t < num_data; t ++) {
            ceres::CostFunction* accl_res = Kortier::AcclResidual::Create(a_measurements.at(t).at(n), -params.gravity, params.imu_noises.sigma_a);
            ceres::CostFunction* gyro_res = Kortier::GyroResidual::Create(w_measurements.at(t).at(n), params.imu_noises.sigma_w);
            problem.AddResidualBlock(accl_res, NULL, 
                                     a_IinG_arr.at(t).data(),
                                     q_GtoI_arr.at(t).coeffs().data(), w_IinG_arr.at(t).data(), wd_IinG_arr.at(t).data(), 
                                     imu_pos.at(n).data(), imu_ori.at(n).coeffs().data(), acc_bias.at(n).data());
            problem.AddResidualBlock(gyro_res, NULL, 
                                     q_GtoI_arr.at(t).coeffs().data(), w_IinG_arr.at(t).data(), 
                                     imu_ori.at(n).coeffs().data(), gyr_mis.at(n).coeffs().data(), gyr_bias.at(n).data());
        }
    }

    // biases
    for (int n = 0; n < params.num_imus; n ++) {
        problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, acc_bias.at(n).data());
        problem.AddResidualBlock(Ours::FixVec3::Create(), NULL, gyr_bias.at(n).data());
    }
    
    
}

}


# endif // ESTIMATOR_H