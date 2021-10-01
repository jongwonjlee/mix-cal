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
#ifndef UTILS_H
#define UTILS_H

#include <fstream>

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) {
    std::exit(signum);
}


bool file_exists(const std::string filename) {
  std::ifstream f(filename);
  return f.good();
}


#endif /* UTILS_H */