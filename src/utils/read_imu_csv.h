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
#ifndef READ_IMU_CSV_H
#define READ_IMU_CSV_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Core>

#include <string>
#include <vector>
#include <map>

#include <assert.h>

void swap_imu_readings(const std::map<size_t, std::vector<Eigen::Vector3d> >& src,
                       std::vector<std::map<size_t, Eigen::Vector3d> >& dst) {
    // clear data
    dst.clear();
    
    // make sure every vector has same length
    int len = 0;
    for (const auto& x : src) {
        if (len == 0) len = x.second.size();    // initialize len
        else assert(len == x.second.size());    // assert length consistency
    }

    // swap data
    for (int i = 0; i < len; i ++) {
        std::map<size_t, Eigen::Vector3d> element;
        for (const auto& x : src) element.insert({x.first, x.second.at(i)});
        dst.push_back(element);
    }
}

void read_imu_csv(const std::string filename, 
                  std::vector<Eigen::Vector3d>& am_arr,
                  std::vector<Eigen::Vector3d>& wm_arr)
{

    // clear data
    am_arr.clear();
    wm_arr.clear();

    // File pointer
    std::fstream fin;
  
    // Open an existing file
    fin.open(filename, std::ios::in);
    if (fin.fail()) {
        std::cout << "[ERROR] file could not be opened." << std::endl;
        std::cout << "filename: " << filename << std::endl;
        return;
    }
    else {
        std::cout << "opened " << filename << std::endl;
    }

    // Read the Data from the file as String Vector
    std::string line, cell;
    // pass header line
    std::getline(fin, line);
    
    // read remaining lines
    while (std::getline(fin, line)) {
        // used for breaking cells
        std::stringstream s(line);
  
        // read every column data of a row and store only accelerometer and gyroscope readings
        Eigen::Vector3d am, wm;
        int col = 0;
        while (std::getline(s, cell, ',')) {
            if (col == 17)      wm(0) = std::stod(cell);
            else if (col == 18) wm(1) = std::stod(cell);
            else if (col == 19) wm(2) = std::stod(cell);
            else if (col == 29) am(0) = std::stod(cell);
            else if (col == 30) am(1) = std::stod(cell);
            else if (col == 31) am(2) = std::stod(cell);
            
            col ++;
        }

        am_arr.push_back(am);
        wm_arr.push_back(wm);
    }

}

#endif // READ_IMU_CSV_H