#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>

#include "StringEnumerator.hpp"
#include "PointCorrespondanceLoader.hpp"
#include "MultiRectification.hpp"
#include <gflags/gflags.h>

const bool verboseLM = true;

DEFINE_bool(corrs, false, "compute correspondences");
DEFINE_string(calib_file, "", "calibration file path");

void read_calib_data(std::string calib_file_path, std::vector<cv::Mat>& K_mats_, std::vector<cv::Mat>& dist_coeffs_ );

int main(int argc, char ** argv)
{
    // parse arguments
    google::ParseCommandLineFlags(&argc, &argv, true);
    // vector of images
    std::vector<cv::Mat> images;

    // load images
    std::cout << "load images ..." << std::endl;
    images.push_back(cv::imread("../input/image0.jpg"));
    images.push_back(cv::imread("../input/image1.jpg"));
    images.push_back(cv::imread("../input/image2.jpg"));
    images.push_back(cv::imread("../input/image3.jpg"));

    // get image size
    std::vector<std::pair<uint,uint>> imageSize;
    for(const auto & im : images)
        imageSize.push_back(std::pair<uint,uint>(im.cols, im.rows)); // (width, height)
    if(verboseLM) for(uint i=0; i < imageSize.size(); ++i) std::cout << "   image " << i << " : " << imageSize[i].first << " x " << imageSize[i].second << std::endl;

    std::vector<Eigen::MatrixXd> correspondences;

    //load the calibration parameters
    std::vector<cv::Mat> K_mats, dist_coeffs;
    std::cout<<FLAGS_calib_file<<std::endl;
    std::cout<<FLAGS_corrs<<std::endl;
    //read_calib_data(FLAGS_calib_file, K_mats, dist_coeffs);
    if (FLAGS_corrs){
        int ij =0;
        for(const auto & im : images){

            int maxCorners = 500;
            int maxTrackbar = 25;
            std::vector<cv::Point2f> corners;
            double qualityLevel = 0.01;
            double minDistance = 10;
            int blockSize = 3, gradientSize = 3;
            bool useHarrisDetector = false;
            double k = 0.04;
            cv::Mat im_gray;
            if(im.channels() == 3)
                cvtColor( im, im_gray, cv::COLOR_BGR2GRAY );
            else
                im_gray = im;
            cv::goodFeaturesToTrack( im_gray,
                                 corners,
                                 maxCorners,
                                 qualityLevel,
                                 minDistance,
                                 cv::Mat(),
                                 blockSize,
                                 gradientSize,
                                 useHarrisDetector,
                                 k );
            std::cout << "** Number of corners detected: " << corners.size() << std::endl;
            int radius = 4;
            //cv::undistortPoints(corners, corners, K_mats[ij], dist_coeffs[ij]);

            std::cout<<corners.size()<<std::endl;
            // Matrix containing the points
            Eigen::MatrixXd points(corners.size(),3);

            for( size_t i = 0; i < corners.size(); i++ )
            {

                points(i,0) = corners[i].x;
                points(i,1) = corners[i].y;
                points(i,2) = 1;
                std::cout << "Here :"<<i<< std::endl;
            }
            correspondences.push_back(points);
            std::cout<<"came out"<<std::endl;
        }

    }
    else{
        // load (or compute by yourself) point correspondances
        std::cout << "load point correspondances ..." << std::endl;
        correspondences.push_back(load_point_corrspondances("../input/image0.list"));
        correspondences.push_back(load_point_corrspondances("../input/image1.list"));
        correspondences.push_back(load_point_corrspondances("../input/image2.list"));
        correspondences.push_back(load_point_corrspondances("../input/image3.list"));
    }

    if(verboseLM) for(const auto &vec_list : correspondences) std::cout << "----- nb points: " << vec_list.rows() << std::endl << vec_list << std::endl << std::endl;

    // display input images
    for(unsigned int im=0; im<images.size(); ++im){
        // draw points
        for(unsigned int i=0; i<correspondences[im].rows(); ++i)
            cv::circle(images[im], cv::Point2i(correspondences[im](i,0),correspondences[im](i,1)), 3, cv::Scalar(0,0,255), -1);
        // display image
        cv::imshow(stringEnumerator("image", im, 2), images[im]);
        //cv::imwrite(stringEnumerator("/tmp/image_", im, 2,".jpg"),images[im]);
    }
    cv::waitKey(100);

    // image rectification
    std::cout << correspondences[0].rows() << " correspondences" << std::endl;
    std::cout << "rectification computation ..." << std::endl;
    const uint referenceImage = images.size()/2; // center image is better, but any other image is fine
    MultiRectification multi(correspondences,imageSize,referenceImage);
    std::vector<Eigen::Matrix3d> H;            // rectifying homographies
    std::vector<Eigen::Matrix<double,3,4>> P;  // projection matrices
    multi.minimize(H,P,300000,verboseLM);

    // show result
    if(verboseLM) for(const auto &h: H) std::cout << "Homography:\n" << h << std::endl << std::endl;
    if(verboseLM) for(const auto &p: P) std::cout << "P :\n" << p << std::endl << std::endl;

    // apply homographies
    cv::Mat all;
    all.create(images[0].rows, images[0].cols * images.size(), CV_8UC3);
    cv::Size siz = cv::Size(images[0].cols, images[0].rows);

    for(uint im=0; im<images.size(); ++im){
        // convert homography from Eigen to OpenCV
        cv::Mat Hocv(3,3,CV_64F);
        for(uint i=0; i<3; ++i)
            for(uint j=0; j<3; ++j)
                Hocv.at<double>(i,j) = H[im](i,j);

/*#if 1 // optional (center final image in a bigger frame)
        const int scaleFactor = 2;
        cv::Mat centerTranslation = (cv::Mat_<double>(3,3) << 1, 0, 0.5*scaleFactor*images[im].cols/2, 0, 1, 0.5*scaleFactor*images[im].rows/2, 0 , 0, 1);
        Hocv = centerTranslation * Hocv;
        cv::warpPerspective(images[im], images[im], Hocv, images[im].size()*scaleFactor);
#else */
        cv::warpPerspective(images[im], images[im], Hocv, images[im].size() );
//#endif
        cv::imshow(stringEnumerator("image", im, 2), images[im]);
        cv::Mat part = all.colRange(siz.width*im, siz.width*(im+1));

        //cvtColor(images[im], part, cv::COLOR_GRAY2BGR);
        std::cout<<images[im].size<<","<<part.size<<std::endl;
        cv::Mat tmp = images[im].clone();
        cv::imshow(stringEnumerator("image", 2, 2), tmp);
        cvWaitKey(0);
        tmp.copyTo(part);
        cv::imwrite(stringEnumerator("imageRes_", im, 2,".jpg"),images[im]);
    }
    for (int j = 0; j < siz.height; j += 16)
        line(all, cv::Point(0, j), cv::Point(siz.width * images.size(), j),
             cv::Scalar(255));
    cv::imshow(stringEnumerator("image", 2, 2), all);
    cv::imwrite(stringEnumerator("imageRes_", 2, 2,".jpg"),all);
    cvWaitKey(0);


    return 0;
}

void read_calib_data(std::string calib_file_path, std::vector<cv::Mat>& K_mats_, std::vector<cv::Mat>& dist_coeffs_ ){


    std::cout << "Reading calibration data from ..."<<calib_file_path<<std::endl;


    cv::FileStorage fs(calib_file_path, cv::FileStorage::READ);
    cv::FileNode fn = fs.root();

    cv::FileNodeIterator fi = fn.begin(), fi_end = fn.end();
    int i=0;

    for (; fi != fi_end; ++fi, i++) {

        cv::FileNode f = *fi;

        // READING CAMERA PARAMETERS from here coz its only one time now due to static camera array
        // in future will have to subscribe from camera_info topic
        // Reading distortion coefficients
        std::vector<double> dc;
        cv::Mat_<double> dist_coeff = cv::Mat_<double>::zeros(1,5);
        f["distortion_coeffs"] >> dc;

        for (int j=0; j < dc.size(); j++)
             dist_coeff(0,j) = (double)dc[j];


        std::vector<int> ims;
        f["resolution"] >> ims;

        // Reading K (camera matrix)
        std::vector<double> intr;
        f["intrinsics"] >> intr;
        cv::Mat_<double> K_mat = cv::Mat_<double>::zeros(3,3);
        K_mat(0,0) = (double)intr[0]; K_mat(1,1) = (double)intr[1];
        K_mat(0,2) = (double)intr[2]; K_mat(1,2) = (double)intr[3];
        K_mat(2,2) = 1.0;


        dist_coeffs_.push_back(dist_coeff);
        K_mats_.push_back(K_mat);

    }

}