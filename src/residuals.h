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
#ifndef RESIDUALS_H
#define RESIDUALS_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;



/*** RESIDUALS FOR SUGGESTED SELF-CALIBRATION WITH BIAS ESTIMATION ***/
namespace Ours {
class AcclResidual {
public:
  AcclResidual(const double dt_, const Eigen::Vector3d& f_inIn_, const Eigen::Vector3d& w_inI0_, const Eigen::Vector3d& f_inI0_, const Eigen::Matrix3d& sqrt_Info_)
  : dt(dt_), f_inIn(f_inIn_), w_inI0(w_inI0_), f_inI0(f_inI0_), sqrt_Info(sqrt_Info_) {}
  
  template <typename T>
  bool operator()(const T* const p_BinIn_ptr, const T* const q_BtoIn_ptr, const T* const q_I0tog_ptr, const T* const wd_inB_ptr, 
                  const T* const am_bias_In_ptr, const T* const wm_bias_I0_ptr, const T* const am_bias_I0_ptr, 
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_BinIn(p_BinIn_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_BtoIn(q_BtoIn_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_I0tog(q_I0tog_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > wd_inB(wd_inB_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > am_bias_In(am_bias_In_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > wm_bias_I0(wm_bias_I0_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > am_bias_I0(am_bias_I0_ptr);

    Eigen::Matrix<T, 3, 3> R_BtoIn(q_BtoIn);
    Eigen::Matrix<T, 3, 3> R_I0tog(q_I0tog);
    Eigen::Matrix<T, 3, 1> p_IninB = - R_BtoIn.transpose() * p_BinIn;

    Eigen::Matrix<T, 3, 1> w_inB = R_I0tog.transpose() * (w_inI0 - wm_bias_I0);

    Eigen::Matrix<T, 3, 1> accel_transverse  = wd_inB.cross(p_IninB);               // transverse acceleration
    Eigen::Matrix<T, 3, 1> accel_centripetal = w_inB.cross(w_inB.cross(p_IninB));  // centripetal acceleration

    Eigen::Matrix<T, 3, 1> f_IninB = (f_inI0 - am_bias_I0) + accel_transverse + accel_centripetal;
    Eigen::Matrix<T, 3, 1> f_IninIn = R_BtoIn * f_IninB;

    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = (f_inIn - am_bias_In) - f_IninIn;
    residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const double dt_, const Eigen::Vector3d& f_inIn_, const Eigen::Vector3d& w_inI0_, const Eigen::Vector3d& f_inI0_, const Eigen::Matrix3d& sqrt_Info_) {
    return new ceres::AutoDiffCostFunction<AcclResidual, 3, 3,4,4,3, 3,3,3>(new AcclResidual(dt_, f_inIn_, w_inI0_, f_inI0_, sqrt_Info_));
  }

private:
  const double dt;
  const Eigen::Vector3d f_inIn;
  const Eigen::Vector3d w_inI0;
  const Eigen::Vector3d f_inI0;
  const Eigen::Matrix3d sqrt_Info;
};

class AngvelResidual {
public:
  AngvelResidual(const Eigen::Vector3d& w_inIn_, const Eigen::Vector3d& w_inI0_, const Eigen::Matrix3d& sqrt_Info_)
  : w_inIn(w_inIn_), w_inI0(w_inI0_), sqrt_Info(sqrt_Info_) {}
  
  template <typename T>
  bool operator()(const T* const q_BtoIn_ptr, const T* const q_Intog_ptr, const T* const q_I0tog_ptr, 
                  const T* const wm_bias_In_ptr, const T* const wm_bias_I0_ptr, 
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T> > q_BtoIn(q_BtoIn_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_Intog(q_Intog_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_I0tog(q_I0tog_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > wm_bias_In(wm_bias_In_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > wm_bias_I0(wm_bias_I0_ptr);

    Eigen::Matrix<T, 3, 3> R_BtoIn(q_BtoIn);
    Eigen::Matrix<T, 3, 3> R_Intog(q_Intog);
    Eigen::Matrix<T, 3, 3> R_I0tog(q_I0tog);

    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = R_BtoIn.transpose() * R_Intog.transpose() * (w_inIn - wm_bias_In) - R_I0tog.transpose() * (w_inI0 - wm_bias_I0);
    residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& w_inIn_, const Eigen::Vector3d& w_inI0_, const Eigen::Matrix3d& sqrt_Info_) {
    return new ceres::AutoDiffCostFunction<AngvelResidual, 3, 4,4,4, 3,3>(new AngvelResidual(w_inIn_, w_inI0_, sqrt_Info_));
  }

private:
  const Eigen::Vector3d w_inIn;
  const Eigen::Vector3d w_inI0;
  const Eigen::Matrix3d sqrt_Info;
};

class BiasResidual {
public:
  BiasResidual(const Eigen::Matrix3d& sqrt_Info_) : sqrt_Info(sqrt_Info_) {}
  
  template <typename T>
  bool operator()(const T* const bias_curr_ptr, const T* const bias_next_ptr, T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > bias_curr(bias_curr_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > bias_next(bias_next_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = bias_next - bias_curr;
    residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& sqrt_Info_) {
    return new ceres::AutoDiffCostFunction<BiasResidual, 3, 3,3>(new BiasResidual(sqrt_Info_));
  }

private:
  const Eigen::Matrix3d sqrt_Info;
};

class FixVec3 {
public:
  FixVec3() {}

  template <typename T>
  bool operator()(const T* const p_ptr, T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p(p_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = p;

    return true;
  }

  static ceres::CostFunction* Create() {
    return new ceres::AutoDiffCostFunction<FixVec3, 3, 3>(new FixVec3());
  }
};

class FixQuat {
public:
  FixQuat() {}

  template <typename T>
  bool operator()(const T* const q_ptr, T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T> > q(q_ptr);

    Eigen::Map<Eigen::Matrix<T, 4, 1> > residuals(residuals_ptr);
    residuals.template block<4, 1>(0, 0) = q.coeffs();
    return true;
  }

  static ceres::CostFunction* Create() {
    return new ceres::AutoDiffCostFunction<FixQuat, 4, 4>(new FixQuat());
  }
};

}


/*** RESIDUALS FOR SELF-CALIBRATION OF BURGARD'S PAPER ***/
namespace Burgard {

class TemporalResidual {
  public:
    TemporalResidual(const double dt_, const double accel_transition_, const double alpha_transition_)
      : dt(dt_), accel_transition(accel_transition_), alpha_transition(alpha_transition_) {}

    template <typename T> 
    bool operator()(const T* const a_prev_ptr, const T* const a_curr_ptr, const T* const w_prev_ptr, const T* const w_curr_ptr, const T* const wd_prev_ptr, const T* const wd_curr_ptr, T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T,3,1> > a_prev(a_prev_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > a_curr(a_curr_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > w_prev(w_prev_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > w_curr(w_curr_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > wd_prev(wd_prev_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > wd_curr(wd_curr_ptr);

      Eigen::Matrix<T,9,1> x_prev;
      Eigen::Matrix<T,9,1> x_curr;
      
      x_prev.template block<3,1>(0,0) = a_prev;
      x_prev.template block<3,1>(3,0) = w_prev;
      x_prev.template block<3,1>(6,0) = wd_prev;
      x_curr.template block<3,1>(0,0) = a_curr;
      x_curr.template block<3,1>(3,0) = w_curr;
      x_curr.template block<3,1>(6,0) = wd_curr;

      Eigen::Matrix<T,9,9> F; 
      Eigen::Matrix<T,3,3> Fsub; 
      F.setIdentity();
      Fsub.setIdentity();
      F.template block<3, 3>(3, 6) = Fsub * dt;
      
      Eigen::Matrix<T, 9, 1> t_diff = x_curr - F * x_prev;
      
      Eigen::Map<Eigen::Matrix<T, 9, 1>> residuals(residuals_ptr);
      residuals.template block<9, 1>(0, 0) = t_diff;

      Eigen::Matrix<double,9,9> Cov = Eigen::Matrix<double,9,9>::Zero();
      for (int k=0; k<3; k++) Cov(k,k) = pow(accel_transition, 2);
      for (int k=3; k<6; k++) Cov(k,k) = pow((alpha_transition * dt), 2);
      for (int k=6; k<9; k++) Cov(k,k) = pow(alpha_transition, 2);
      Eigen::Matrix<double,9,9> sqrt_Info = Eigen::LLT<Eigen::Matrix<double,9,9> >(Cov.inverse()).matrixL().transpose();

      residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

      return true;
    }

    static ceres::CostFunction* Create(const double dt_, const double accel_transition_, const double alpha_transition_) {
      return new ceres::AutoDiffCostFunction<TemporalResidual,9,3,3,3,3,3,3>(new TemporalResidual(dt_, accel_transition_, alpha_transition_));
    }

  private:
    const double dt;
    const double accel_transition;
    const double alpha_transition;
};

class SpatialResidual {
public:
  SpatialResidual(const Eigen::Vector3d& f_inU_, const double sigma_a_)
  : f_inU(f_inU_), sigma_a(sigma_a_) {}
  
  template <typename T>
  bool operator()(const T* const f_IinI_ptr, const T* const w_IinI_ptr, const T* const wd_IinI_ptr, const T* const p_IinU_ptr, const T* const q_ItoU_ptr, T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > f_IinI(f_IinI_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > w_IinI(w_IinI_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > wd_IinI(wd_IinI_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_IinU(p_IinU_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_ItoU(q_ItoU_ptr);

    Eigen::Matrix<T, 3, 3> R_ItoU(q_ItoU);
    Eigen::Matrix<T, 3, 1> p_UinI = - R_ItoU.transpose() * p_IinU;

    Eigen::Matrix<T, 3, 1> accel_transverse  = wd_IinI.cross(p_UinI);             // transverse acceleration
    Eigen::Matrix<T, 3, 1> accel_centripetal = w_IinI.cross(w_IinI.cross(p_UinI));   // centripetal acceleration

    Eigen::Matrix<T, 3, 1> f_UinI = f_IinI + accel_transverse + accel_centripetal;
    Eigen::Matrix<T, 3, 1> f_UinU = R_ItoU * f_UinI;

    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = f_UinU - f_inU;
    
    Eigen::Matrix<double,3,3> Cov = Eigen::Matrix<double,3,3>::Zero();
    for (int k=0; k<3; k++) Cov(k,k) = pow(sigma_a, 2);
    Eigen::Matrix<double,3,3> sqrt_Info = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov.inverse()).matrixL().transpose();

    residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& f_inU_, const double sigma_a_) {
    return new ceres::AutoDiffCostFunction<SpatialResidual,3, 3,3,3,3,4>(new SpatialResidual(f_inU_, sigma_a_));
  }

private:
  const Eigen::Vector3d f_inU;
  const double sigma_a;
};

}


/*** RESIDUALS FOR SELF-CALIBRATION OF KORTIER'S PAPER ***/
namespace Kortier {

class TranslationalResidual {
  public:
    TranslationalResidual(double dt_, const double accel_transition_)
      : dt(dt_), accel_transition(accel_transition_) {}

    template <typename T> 
    bool operator()(const T* const p_IkinG_ptr, const T* const p_Ikp1inG_ptr, 
                    const T* const v_IkinG_ptr, const T* const v_Ikp1inG_ptr, 
                    const T* const a_IkinG_ptr, const T* const a_Ikp1inG_ptr, 
                    T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T,3,1> > p_IkinG(p_IkinG_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > v_IkinG(v_IkinG_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > a_IkinG(a_IkinG_ptr);

      Eigen::Map<const Eigen::Matrix<T,3,1> > p_Ikp1inG(p_Ikp1inG_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > v_Ikp1inG(v_Ikp1inG_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > a_Ikp1inG(a_Ikp1inG_ptr);
      
      Eigen::Map<Eigen::Matrix<T,9,1>> residuals(residuals_ptr);
      
      residuals.template block<3,1>(0,0) = p_Ikp1inG - (p_IkinG + v_IkinG * dt + a_IkinG * dt * dt * 0.5);
      residuals.template block<3,1>(3,0) = v_Ikp1inG - (v_IkinG + a_IkinG * dt);
      residuals.template block<3,1>(6,0) = a_Ikp1inG - a_IkinG;

      Eigen::Matrix<double,9,9> Cov = Eigen::Matrix<double,9,9>::Zero();
      for (int k=0; k<3; k++) Cov(k,k) = pow(accel_transition * dt * dt * 0.5, 2);
      for (int k=3; k<6; k++) Cov(k,k) = pow(accel_transition * dt, 2);
      for (int k=6; k<9; k++) Cov(k,k) = pow(accel_transition, 2);
      Eigen::Matrix<double,9,9> sqrt_Info = Eigen::LLT<Eigen::Matrix<double,9,9> >(Cov.inverse()).matrixL().transpose();

      residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

      return true;
    }

    static ceres::CostFunction* Create(const double dt_, const double accel_transition_) {
      return new ceres::AutoDiffCostFunction<TranslationalResidual,9, 3,3,3,3,3,3>(new TranslationalResidual(dt_, accel_transition_));
    }

  private:
    const double dt;
    const double accel_transition;
};

class RotationalResidual {
  public:
    RotationalResidual(double dt_, const double alpha_transition_)
      : dt(dt_), alpha_transition(alpha_transition_) {}

    template <typename T> 
    bool operator()(const T* const q_GtoIk_ptr, const T* const q_GtoIkp1_ptr, 
                    const T* const w_IkinG_ptr, const T* const w_Ikp1inG_ptr, 
                    const T* const wd_IkinG_ptr, const T* const wd_Ikp1inG_ptr, 
                    T* residuals_ptr) const {
      Eigen::Map<const Eigen::Quaternion<T> > q_GtoIk(q_GtoIk_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > w_IkinG(w_IkinG_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > wd_IkinG(wd_IkinG_ptr);

      Eigen::Map<const Eigen::Quaternion<T> > q_GtoIkp1(q_GtoIkp1_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > w_Ikp1inG(w_Ikp1inG_ptr);
      Eigen::Map<const Eigen::Matrix<T,3,1> > wd_Ikp1inG(wd_Ikp1inG_ptr);
      
      Eigen::Map<Eigen::Matrix<T,10,1>> residuals(residuals_ptr);

      Eigen::Matrix<T,3,1> w_IkinG_tmp = w_IkinG_tmp + wd_IkinG * dt;
      Eigen::Matrix<T,4,4> Omega = getOmega(w_IkinG_tmp);
      Eigen::Matrix<T,4,4> Exp = getExp(Omega);
      
      Eigen::Matrix<T,4,1> quat_GtoIk = q_GtoIk.coeffs();
      Eigen::Matrix<T,4,1> quat_GtoIkp1 = q_GtoIkp1.coeffs();

      residuals.template block<4,1>(0,0) = quat_GtoIkp1 - (- 0.5 * Exp * quat_GtoIk);
      residuals.template block<3,1>(4,0) = w_Ikp1inG - (w_IkinG + wd_IkinG * dt);
      residuals.template block<3,1>(7,0) = wd_Ikp1inG - wd_IkinG;

      Eigen::Matrix<double,10,10> Cov = Eigen::Matrix<double,10,10>::Zero();

      Eigen::Matrix<double,3,1> vec_cov;
      for (int k=0; k<3; k++) vec_cov(k,0) = pow(alpha_transition * dt, 2);
      Eigen::Matrix<double,4,4> Omega_cov = getOmega(vec_cov);
      Eigen::Matrix<double,4,4> Exp_cov = getExp(Omega_cov);

      Cov.block<4,4>(0,0) = Exp_cov;
      for (int k=4; k<7; k++) Cov(k,k) = pow(alpha_transition * dt, 2);
      for (int k=7; k<10; k++) Cov(k,k) = pow(alpha_transition, 2);
      Eigen::Matrix<double,10,10> sqrt_Info = Eigen::LLT<Eigen::Matrix<double,10,10> >(Cov.inverse()).matrixL().transpose();

      residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

      return true;
    }

    static ceres::CostFunction* Create(const double dt_, const double alpha_transition_) {
      return new ceres::AutoDiffCostFunction<RotationalResidual,10, 4,4,3,3,3,3>(new RotationalResidual(dt_, alpha_transition_));
    }

    template <typename T>
    Eigen::Matrix<T,4,4> getOmega(Eigen::Matrix<T,3,1> w) const {
      Eigen::Matrix<T,4,4> Omega = Eigen::Matrix<T,4,4>::Zero();
      Omega.template block<1,3>(0,1) = w.transpose();
      Omega.template block<3,1>(1,0) = -w;
      Omega.template block<3,3>(1,1) = getSkew(w);
      return Omega;
    }

    template <typename T>
    Eigen::Matrix<T,3,3> getSkew(Eigen::Matrix<T,3,1> w) const {
      Eigen::Matrix<T,3,3> Skew;// = Eigen::Matrix<T,3,3>::Zero();
      Skew(1,2) = -w(0,0);
      Skew(2,1) = w(0,0);
      Skew(0,2) = w(1,0);
      Skew(2,0) = -w(1,0);
      Skew(0,1) = -w(2,0);
      Skew(1,0) = w(2,0);
      return Skew;
    }

    template <typename T>
    Eigen::Matrix<T,4,4> getExp(Eigen::Matrix<T,4,4> Omega) const {
      Eigen::Matrix<T,4,4> Exp;
      Exp = Eigen::Matrix<T,4,4>::Identity() + Omega * dt * 0.5;
      return Exp;
    }

  private:
    const double dt;
    const double alpha_transition;
};

class AcclResidual {
public:
  AcclResidual(const Eigen::Vector3d& f_inU_, const Eigen::Vector3d& g_inG_, const double& sigma_a_)
  : f_inU(f_inU_), g_inG(g_inG_), sigma_a(sigma_a_) {}
  
  template <typename T>
  bool operator()(const T* const a_IinG_ptr,
                  const T* const q_GtoI_ptr, const T* const w_IinG_ptr, const T* const wd_IinG_ptr, 
                  const T* const p_IinU_ptr, const T* const q_ItoU_ptr, const T* const U_bias_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > a_IinG(a_IinG_ptr);
    
    Eigen::Map<const Eigen::Quaternion<T> > q_GtoI(q_GtoI_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > w_IinG(w_IinG_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > wd_IinG(wd_IinG_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_IinU(p_IinU_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_ItoU(q_ItoU_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > U_bias(U_bias_ptr);

    Eigen::Matrix<T, 3, 3> R_ItoU(q_ItoU);
    Eigen::Matrix<T, 3, 1> p_UinI = - R_ItoU.transpose() * p_IinU;

    Eigen::Matrix<T, 3, 3> R_GtoI(q_GtoI);
    Eigen::Matrix<T, 3, 1> w_IinI = R_GtoI * w_IinG;
    Eigen::Matrix<T, 3, 1> wd_IinI = R_GtoI * wd_IinG;

    Eigen::Matrix<T, 3, 1> accel_transverse  = wd_IinI.cross(p_UinI);             // transverse acceleration
    Eigen::Matrix<T, 3, 1> accel_centripetal = w_IinI.cross(w_IinI.cross(p_UinI));   // centripetal acceleration

    Eigen::Matrix<T, 3, 1> a_UinI = R_GtoI * a_IinG + accel_transverse + accel_centripetal;
    Eigen::Matrix<T, 3, 1> f_UinU = R_ItoU * (a_UinI - R_GtoI * g_inG);

    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = f_UinU - (f_inU - U_bias);

    Eigen::Matrix<double,3,3> Cov = Eigen::Matrix<double,3,3>::Zero();
    for (int k=0; k<3; k++) Cov(k,k) = sigma_a;
    Eigen::Matrix<double,3,3> sqrt_Info = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov.inverse()).matrixL().transpose();

    residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& f_inU_, const Eigen::Vector3d& g_inG_, const double& sigma_a_) {
    return new ceres::AutoDiffCostFunction<AcclResidual,3, 3,4,3,3,3,4,3>(new AcclResidual(f_inU_, g_inG_, sigma_a_));
  }

private:
  const Eigen::Vector3d f_inU;
  const Eigen::Vector3d g_inG;
  const double sigma_a;
};

class GyroResidual {
public:
  GyroResidual(const Eigen::Vector3d& w_inU_, const double& sigma_w_)
  : w_inU(w_inU_), sigma_w(sigma_w_) {}
  
  template <typename T>
  bool operator()(const T* const q_GtoI_ptr, const T* const w_IinG_ptr, 
                  const T* const q_ItoU_ptr, const T* const q_Utog_ptr, const T* const U_bias_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T> > q_GtoI(q_GtoI_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > w_IinG(w_IinG_ptr);
    
    Eigen::Map<const Eigen::Quaternion<T> > q_ItoU(q_ItoU_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_Utog(q_Utog_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > U_bias(U_bias_ptr);

    Eigen::Matrix<T, 3, 3> R_ItoU(q_ItoU);
    Eigen::Matrix<T, 3, 3> R_Utog(q_Utog);
    Eigen::Matrix<T, 3, 3> R_GtoI(q_GtoI);

    Eigen::Matrix<T, 3, 1> w_UinU = R_Utog * R_ItoU * R_GtoI * w_IinG;

    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = (w_inU - U_bias) - w_UinU;

    Eigen::Matrix<double,3,3> Cov = Eigen::Matrix<double,3,3>::Zero();
    for (int k=0; k<3; k++) Cov(k,k) = sigma_w;
    Eigen::Matrix<double,3,3> sqrt_Info = Eigen::LLT<Eigen::Matrix<double,3,3> >(Cov.inverse()).matrixL().transpose();

    residuals.applyOnTheLeft(sqrt_Info.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& w_inU_, const double& sigma_w_) {
    return new ceres::AutoDiffCostFunction<GyroResidual,3, 4,3, 4,4,3>(new GyroResidual(w_inU_, sigma_w_));
  }

private:
  const Eigen::Vector3d w_inU;
  const double sigma_w;
};

}


# endif // RESIDUALS_H