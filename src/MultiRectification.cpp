#include "MultiRectification.hpp"

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>


///////////////////////////////////////////////////////////////////////
MultiRectification::MultiRectification(const std::vector<Eigen::MatrixXd> &PointCorresp,
                                       const std::vector<std::pair<uint, uint>> &imageSizes,
                                       const unsigned int referenceImageId)
                                       : correspondences(PointCorresp),
                                         imageWidth(correspondences.size()),
                                         imageHeight(correspondences.size()),
                                         referenceImage(referenceImageId), 
                                         nbImages(correspondences.size())
{
  // check the data
  if(correspondences.size() != imageSizes.size()){
    std::cerr << "error: MultiRectification: number of images and number of point correspondence not consistent" << std::endl;
    exit(EXIT_FAILURE);
  }     

  // check the data consistensy
  if(correspondences.size() < 2){
    std::cerr << "not enough images" << std::endl;
    exit(EXIT_FAILURE);
  }    

  // check the correspondences consistensy
  for(unsigned int i=1; i<correspondences.size(); ++i)
    if(correspondences[i].rows() != correspondences[0].rows()){
      std::cerr << "error: MultiRectification: the point correspondences should have the same size" << std::endl;
      exit(EXIT_FAILURE);
    }

  // reference image
  if(referenceImage > nbImages-1){
    std::cerr << "error: MultiRectification: invalid reference image id : " << referenceImageId << std::endl;
    exit(EXIT_FAILURE);
  }  

  // setup the image resolution data
  for(unsigned int i=0; i<nbImages; ++i){
    imageWidth[i]  = imageSizes[i].first;
    imageHeight[i] = imageSizes[i].second;
  }
}


///////////////////////////////////////////////////////////////////////
void MultiRectification::readData(const Eigen::VectorXd &x,
                                  std::vector<double> &Rx,
                                  std::vector<double> &Ry,
                                  std::vector<double> &Rz,
                                  std::vector<double> &alpha) const
{
  unsigned int index = 0;
  for(unsigned int i=0; i<nbImages; ++i){
    Rx[i]    = x[index++];
    Ry[i]    = x[index++];
    Rz[i]    = x[index++];
    alpha[i] = x[index++];    
  }
}

///////////////////////////////////////////////////////////////////////
void MultiRectification::getKR(const Eigen::VectorXd &x, 
                               std::vector<Eigen::Matrix3d> &Koriginal,
                               std::vector<Eigen::Matrix3d> &Knew,
                               std::vector<Eigen::Matrix3d> &R) const
{
  // extract the data from vector x
  std::vector<double> Rx(nbImages);
  std::vector<double> Ry(nbImages);
  std::vector<double> Rz(nbImages);
  std::vector<double> alpha(nbImages);
  readData(x,Rx,Ry,Rz,alpha);

  // build the internal parameters
  for(unsigned int i=0; i<nbImages; ++i){ 
    // Koriginal
    Koriginal[i].setIdentity();
    Koriginal[i](0,0) = Koriginal[i](1,1) = sqrt(pow(imageWidth[i],2) + pow(imageHeight[i],2));

    // Knew
    Knew[i] = Koriginal[i];
    if(i != referenceImage) Knew[i](0,0) = Knew[i](1,1) = Koriginal[i](0,0) * pow(3.0,alpha[i]);
  }

  // build the rotation matrices
  for(unsigned int i=0; i<nbImages;i++){ 
    if(i != referenceImage) R[i] = eulerAngle(Rx[i], Ry[i], Rz[i]);
    else R[i] = eulerAngle(0.0, Ry[i], Rz[i]);
  }
}

///////////////////////////////////////////////////////////////////////
void MultiRectification::getH(const Eigen::VectorXd &x, std::vector<Eigen::Matrix3d> &H) const
{
  // compute the internal parameters and rotation matrices
  std::vector<Eigen::Matrix3d> Koriginal(nbImages);
  std::vector<Eigen::Matrix3d> Knew(nbImages);
  std::vector<Eigen::Matrix3d> R(nbImages);
  getKR(x,Koriginal,Knew,R);

  // compute the centering homography
  std::vector<Eigen::Matrix3d> Hcenter(nbImages);    
  for(unsigned int i=0; i<nbImages; ++i){ 
    // Hcenter
    Hcenter[i].setIdentity();
    Hcenter[i](0,2) = -imageHeight[i] / 2.0;
    Hcenter[i](1,2) = -imageWidth[i]/ 2.0;
    ////  Hcenter[i](0,2) = -imageWidth[i] / 2.0;
    ////  Hcenter[i](1,2) = -imageHeight[i]/ 2.0;
  }
    
  // compute the reverse centering homography
  Eigen::Matrix3d HoutInv;    
  HoutInv.setIdentity();
  HoutInv(0,2) = imageHeight[referenceImage]  / 2.0;
  HoutInv(1,2) = imageWidth[referenceImage] / 2.0;
  ////HoutInv(0,2) = imageWidth[referenceImage]  / 2.0;
  ////HoutInv(1,2) = imageHeight[referenceImage] / 2.0;

  // compute the homographies
  Eigen::Matrix3d Koriginal_inv;
  for(unsigned int i=0; i<nbImages; ++i){
    Koriginal_inv = Koriginal[i].inverse();
    H[i] = HoutInv * Knew[i] * R[i] * Koriginal_inv * Hcenter[i];
  }
}


///////////////////////////////////////////////////////////////////////
void MultiRectification::getP(const Eigen::VectorXd &x, std::vector<Eigen::Matrix<double,3,4>> &P) const
{
  // compute the internal parameters and rotation matrices
  std::vector<Eigen::Matrix3d> Koriginal(nbImages);
  std::vector<Eigen::Matrix3d> Knew(nbImages);
  std::vector<Eigen::Matrix3d> R(nbImages);
  getKR(x,Koriginal,Knew,R);

  // compute each camera projection matrix
  Eigen::Vector4d C;
  C.fill(0.0);
  C(3) = 1.0;
  for(unsigned int i=0; i<P.size(); ++i){
    // compute the camera position
    C(0) = (double)i;

    // compute the camera matrix
    Koriginal[i](0,0) = Koriginal[i](1,1) =  Koriginal[i](0,0) * Koriginal[i](0,0) / Knew[i](0,0);
    Koriginal[i](0,2) = imageHeight[i] / 2.0;
    Koriginal[i](1,2) = imageWidth[i]/ 2.0;
    ////Koriginal[i](0,2) = imageWidth[i] / 2.0;
    ////Koriginal[i](1,2) = imageHeight[i]/ 2.0;

    // compute the projection matrix
    P[i].leftCols(3)  = R[i].transpose();
    P[i].rightCols(1) = - R[i].transpose() * C.head(3);
    P[i] = Koriginal[i] * P[i];
  }
}


///////////////////////////////////////////////////////////////////////
/// error function to minimize : should be "surcharged" in the daugther class
Eigen::VectorXd MultiRectification::error(const Eigen::VectorXd &x) const
{
  // compute the homographies from vector x
  std::vector<Eigen::Matrix3d> H(nbImages);
  getH(x,H);

  // compute the error vector
  Eigen::VectorXd error(1);
  error.fill(0.0);
  Eigen::Vector3d pt;
  for(unsigned int i=0; i<correspondences[0].rows(); ++i){
    // compute the average value of transformed y
    double average = 0.0;
    for(unsigned int j=0; j<nbImages; j++){
      Eigen::Vector3d tmp;
      tmp(0) = correspondences[j](i,0);
      tmp(1) = correspondences[j](i,1);
      tmp(2) = correspondences[j](i,2);
      pt = H[j] * tmp;
      pt /= pt(2);
      average += pt(1);
    }
 
    average /= (double)nbImages;

    // error = sum of the y-difference with the y-average
    for(unsigned int j=0; j<nbImages; j++){
      Eigen::Vector3d tmp;
      tmp(0) = correspondences[j](i,0);
      tmp(1) = correspondences[j](i,1);
      tmp(2) = correspondences[j](i,2);
      pt = H[j] * tmp;
      pt /= pt(2);
      error[0] += pow(average-pt(1),2);//fabs(average-pt(1));
    }
  }

  return error;
}


///////////////////////////////////////////////////////////////////////
int MultiRectification::minimize(std::vector<Eigen::Matrix3d> &H, std::vector<Eigen::Matrix<double,3,4>> &P, const unsigned int nbMaxIteration, const bool statVerbose)
{
  // initial solution vector
  Eigen::VectorXd x(4*nbImages);
  x.fill(0.0);
  
  // expected error vector
  Eigen::VectorXd X(1);
  X.fill(0.0);

  // minimization process
  bool restult = LevenbergMarquardt::minimize(x,X,nbMaxIteration,statVerbose);

  // get back the homographies
  if(H.size() != nbImages){
    H.clear();
    H = std::vector<Eigen::Matrix3d>(nbImages);
  }
  getH(x,H);

  // get back the projection matrix
  if(P.size() != nbImages){
    P.clear();
    P = std::vector<Eigen::Matrix<double,3,4>>(nbImages);
  }
  getP(x,P);

  //std::cout << "result: \n" << x <<  std::endl;

  return restult;
}


///////////////////////////////////////////////////////////////////////
// angles in radian
Eigen::Matrix3d MultiRectification::eulerAngle(const double &x, const double &y, const double &z) const 
{
  // x
  Eigen::Matrix3d Rx;
  Rx.setIdentity();
  Rx(1,1) =  Rx(2,2) = cos(x);
  Rx(1,2) = -sin(x);
  Rx(2,1) = -Rx(1,2);

  // y
  Eigen::Matrix3d Ry;
  Ry.setIdentity();
  Ry(0,0) =  Ry(2,2) = cos(y);
  Ry(0,2) =  sin(y);
  Ry(2,0) = -Ry(0,2);

  // z
  Eigen::Matrix3d Rz;
  Rz.setIdentity();
  Rz(0,0) = Rz(1,1) = cos(z);
  Rz(0,1) = -sin(z);
  Rz(1,0) = -Rz(0,1);

  // in the order Y-Z-X
  return Rx * Rz * Ry;
}


