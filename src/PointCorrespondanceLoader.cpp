#include <fstream>
#include <iostream>

#include "PointCorrespondanceLoader.hpp"

Eigen::MatrixXd load_point_corrspondances(const std::string &filename){

    //open the file
    std::ifstream myfile;
    myfile.open(filename, std::ios::in | std::ios::binary);

    if(!myfile.is_open()){
        std::cerr << "error: can not open file: " << filename << std::endl;
        return Eigen::MatrixXd(0,3);
    }

    // read the vector size
    unsigned int nbPoints;
    myfile >> nbPoints;

    // Matrix containing the points
    Eigen::MatrixXd points(nbPoints,3);

    // load points
    for(size_t i=0; i<nbPoints; ++i){
        myfile >> points(i,0);
        myfile >> points(i,1);
        myfile >> points(i,2);
    }

    myfile.close();

    return points;
}

