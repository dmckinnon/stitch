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
	Get final image size, by predicting the overlap and getting 
	the largest rectangle that fits over this
*/
pair<int, int> GetFinalImageSize(const Mat& img1, const Mat& img2, const Matrix3f& H)
{
	Vector3f topRight(img2.cols, 0, 1);
	Vector3f bottomRight(img2.cols, img2.rows, 1);
	Vector3f bottomLeft(0, img2.rows, 1);
	Vector3f topLeft(0, 0, 1);

	auto tr = H * topRight;
	auto br = H * bottomRight;
	auto bl = H * bottomLeft;
	auto tl = H * topLeft;

	int matHeight = (int)abs(max((float)img1.rows, max(bl(1), br(1))) - min(0.f, min(tr(1), tl(1))));
	int matWidth = (int)abs(max((float)img1.cols, max(tr(0), br(0))) - min(0.f, min(tl(0), bl(0))));

	return make_pair(matWidth, matHeight);
}

/*
	Stitch two images together, given the homography from img2 to img1.
	We use img1 as the reference plane, and return a Mat that is bigger than
	both, containing the stitched images. 

	We do not assume that the images are the same size.

	Should this fail, we return a 1x1 Mat that contains the value 0. 
*/
// Support functions
uchar BilinearInterpolatePixel(const Mat& img, const float& x, const float& y)
{
	float x1 = floor(x);
	float x2 = ceil(x);
	float y1 = floor(y);
	float y2 = ceil(y);

	uchar y1Val = ((x2-x)/(x2-x1))*img.at<uchar>(y1, x1) + ((x - x1) / (x2 - x1))*img.at<uchar>(y1, x2);
	uchar y2Val = ((x2 - x) / (x2 - x1))*img.at<uchar>(y2, x1) + ((x - x1) / (x2 - x1))*img.at<uchar>(y2, x2);

	uchar val = ((y2-y)/(y2-y1))*y1Val + ((y-y1)/(y2-y1))*y2Val;
	return val;
}
// Actual function
void Stitch(const cv::Mat& img1, const cv::Mat& img2, const Eigen::Matrix3f& H, Mat& composite)
{
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

	auto tr = H * topRight;
	auto br = H * bottomRight;
	auto bl = H * bottomLeft;
	auto tl = H * topLeft;

	int img1YOffset = (int)abs(min(0.f, min(tr(1), tl(1))));
	int img1XOffset = (int)abs(min(0.f, min(tl(0), bl(0))));

	// Add the first image to the centre of the new big Mat
	for (unsigned int y = 0; y < img1.rows; ++y)
	{
		for (unsigned int x = 0; x < img1.cols; ++x)
		{
			composite.at<uchar>(y + img1YOffset, x + img1XOffset) = img1.at<uchar>(y,x);
		}
	}

	// For the second image, reproject every pixel in the first Mat back into image to be stitched in.
	// If it isn't there, move on.
	// If it is there, bilinearly interpolate the value of that sub-pixel location
	// In the original Mat, if this clashes with a point in the original image,
	// take the average and place that there
	// TODO: add blending at this stage
	for (unsigned int y = 0; y < composite.rows; ++y)
	{
		for (unsigned int x = 0; x < composite.cols; ++x)
		{
			Vector3f pixel(x-img1XOffset,y-img1YOffset,1);
			auto transformedPixel = H.inverse() * pixel;

			uchar pixelVal = 0;
			if (0 < transformedPixel(0) && transformedPixel(0) < img2.cols-1)
			{
				if (0 < transformedPixel(1) && transformedPixel(1) < img2.rows-1)
				{
					uchar pixelVal = BilinearInterpolatePixel(img2, transformedPixel(0), transformedPixel(1));
					if (composite.at<uchar>(y, x) == 0)
						composite.at<uchar>(y, x) = pixelVal;
					else
						composite.at<uchar>(y, x) = (pixelVal+ composite.at<uchar>(y, x))/2;
				}
			}
		}
	}
}
