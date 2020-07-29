# Code
This code performs image rectification for 2 or more aligned cameras.

# Reference
If you perform any academic work with this program, please make a reference to the following paper in your publication:
@Article{nozick_AT_2013,
	author = "Vincent Nozick",
	title = "Camera array image rectification and calibration for stereoscopic and autostereoscopic displays",
	journal = "annals of telecommunications",
	month = "July",
	year = "2013",
	pages = "581--596",
	volume = "68",
	number = "11",
	abstract = "This paper presents an image rectification method for an arbitrary number of views with aligned camera center. This paper also describes how to extend this method to easily perform a robust camera calibration. These two techniques can be used for stereoscopic rendering to enhance the perception comfort or for depth from stereo. In this paper, we first expose why epipolar geometry is not suited to solve this problem. Second, we propose a nonlinear method that includes all the images in the rectification process. Then, we detail how to extract the rectification parameters to provide a quasi-Euclidean camera calibration. Our method only requires point correspondences between the views and can handle images with different resolutions. The tests show that it is robust to noise and to sparse point correspondences among the views."
}


# Dependencies
To run this program, you will need:
* cmake
* OpenCV 3.0
* Eigen 3.0
make sure to install them before to try the code.

# Usage for the example
## Linux and Macos
- cd multipleViewRectification
- mkdir build
- cd build
- cmake ..
- make
- ./bin/multipleRectification

## Windows
- use an IDE that uses cmake (or tell me how you managed it :)

# Input data
To use this program with you own data, you have to first extract point correspondences between your images and store them in homogeneous coordinates, like in the example.

# Help
If you need help, please contact me: Vincent Nozick (vincent.nozick@univ-eiffel.fr)
