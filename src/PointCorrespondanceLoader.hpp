#ifndef IMAGERECTIFICATION_POINTCORRESPONDANCELOADER_HPP
#define IMAGERECTIFICATION_POINTCORRESPONDANCELOADER_HPP

#include <vector>
#include <cstring>

#include <Eigen/Dense>

Eigen::MatrixXd load_point_corrspondances(const std::string &filename);


#endif //IMAGERECTIFICATION_POINTCORRESPONDANCELOADER_HPP
