#ifndef MultiRectification_h
#define MultiRectification_h

#include <vector>

#include <Eigen/Dense>

#include "LevenbergMarquardt.hpp"


class MultiRectification : public LevenbergMarquardt{


protected :

  ///////////////////////////////////////////////////////////////////////
  /// a vector of input sorted point correspondences between each view
  std::vector<Eigen::MatrixXd> correspondences;

  ///////////////////////////////////////////////////////////////////////
  /// the images resolution
  Eigen::VectorXd imageWidth;
  Eigen::VectorXd imageHeight;
  
  ///////////////////////////////////////////////////////////////////////
  /// reference image
  unsigned int referenceImage;  

  ///////////////////////////////////////////////////////////////////////
  /// number of images
  unsigned int nbImages;


public :

  ///////////////////////////////////////////////////////////////////////
  /// constructor
  MultiRectification(const std::vector<Eigen::MatrixXd> &PointCorresp,
                     const std::vector<std::pair<uint, uint>> &imageSizes,
                     const unsigned int referenceImageId = 0);


  ///////////////////////////////////////////////////////////////////////
  /// destructor
  ~MultiRectification(){}

  ///////////////////////////////////////////////////////////////////////
  /// This function solves a non-linear system. 
  /// the x vector is the unknown
  /// The vector X corresponds to the state we want to reach : we want X=f(a,b).
  /// This function returns 1 if an acceptable solution has been found before the nbMaxIteration iterations.
  /// This algorithm is detailed in the "Multiple View Geometry", Hartley & Zisserman, 2nd edition, page 605.
  int minimize(std::vector<Eigen::Matrix3d> &H, std::vector<Eigen::Matrix<double,3,4>> &P, const unsigned int nbMaxIteration = 200, const bool statVerbose=true);

  ///////////////////////////////////////////////////////////////////////
  /// angles in radian
  Eigen::Matrix3d eulerAngle(const double &x, const double &y, const double &z) const;

protected :

  ///////////////////////////////////////////////////////////////////////
  /// error function to minimize : should be "surcharged" in the daugther class
  Eigen::VectorXd error(const Eigen::VectorXd &x) const;

  ///////////////////////////////////////////////////////////////////////
  void readData(const Eigen::VectorXd &x,
                std::vector<double> &Rx,
                std::vector<double> &Ry,
                std::vector<double> &Rz,
                std::vector<double> &alpha) const;

  ///////////////////////////////////////////////////////////////////////
  void getKR(const Eigen::VectorXd &x, 
             std::vector<Eigen::Matrix3d> &Koriginal,
             std::vector<Eigen::Matrix3d> &Knew,
             std::vector<Eigen::Matrix3d> &R) const;
 
  ///////////////////////////////////////////////////////////////////////
  void getH(const Eigen::VectorXd &x, std::vector<Eigen::Matrix3d> &H) const;

  ///////////////////////////////////////////////////////////////////////
  void getP(const Eigen::VectorXd &x, std::vector<Eigen::Matrix<double,3,4>> &P) const;

};

#endif

