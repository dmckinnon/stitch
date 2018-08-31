#include "Features.h"
#include <iostream>
#include <algorithm>
#include <Eigen/SVD>
#include "Compositor.h"
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;
using namespace Eigen;

/*
	Compositor function implementations
*/

/*
	Stitch two images together, given the homography from img2 to img1.
	We use img1 as the reference plane, and return a Mat that is bigger than
	both, containing the stitched images. 

	We do not assume that the images are the same size.

	Should this fail, we return a 1x1 Mat that contains the value 0. 
*/
// Support functions

// Actual function
cv::Mat Stitch(const cv::Mat& img1, const cv::Mat& img2, Eigen::Matrix3f H)
{
	// Normalise the points


	// To get the necessary size of the final Mat,
	// convert img2 corners to normalised projective coords, 
	// apply the homography, convert these to an img1 reference frame, 
	// and see where they lie relative to img1's 0,0
	int img1Width = img1.cols;
	int img1Height = img1.rows;
	Vector3f topRight(img2.cols, 0, 1);
	Vector3f bottomRight(img2.cols, img2.rows, 1);
	Vector3f bottomLeft(0, img2.rows, 1);
	Vector3f topLeft(0, 0, 1);

	auto topRightInImg1Space = H * topRight;
	auto bottomRightInImg1Space = H * bottomRight;
	auto bottomLeftInImg1Space = H * bottomLeft;
	auto topLeftInImg1Space = H * topLeft;

	std::cout << "Corners\n";
	cout << topRightInImg1Space << endl;
	cout << bottomRightInImg1Space << endl;
	cout << bottomLeftInImg1Space << endl;
	cout << topLeftInImg1Space << endl;

	// For testing, put these into two mats and display them


	// Create the final Mat. 
	Mat test(img1.cols*2, img1.rows*2, CV_8U, Scalar(0));

	// Add the first image
	for (unsigned int y = 0; y < img1.rows; ++y)
	{
		for (unsigned int x = 0; x < img1.cols; ++x)
		{
			test.at<uchar>(y + img1.rows/2, x + img1.cols/2) = img1.at<uchar>(y,x);
		}
	}

	for (unsigned int y = 0; y < img2.rows; ++y)
	{
		for (unsigned int x = 0; x < img2.cols; ++x)
		{
			Vector3f point(x, y, 1);
			auto Hx = H * point;
			test.at<uchar>(img2.rows / 2 + (int)Hx(1), img2.cols / 2 + (int)Hx(0)) = img2.at<uchar>(y, x);
		}
	}

	std::string debugWindowName = "test image";
	namedWindow(debugWindowName);
	imshow(debugWindowName, test);
	waitKey(0);
	return img1;
}
