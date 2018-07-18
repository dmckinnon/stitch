#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

struct point
{
	int x;
	int y;
};

/* Function prototypes */
bool FindFASTFeatures(Mat img, vector<point>& features);

/* Function Implementations */
int main(int argc, char** argv)
{
	/* Panorama stitching
	
	The aim of this exercise is to learn all the different parts to image stitching. 
	These are:
	- feature detection
	- feature description
	- feature matching 
	- optimisation via direct linear transform
	- applying transformations to images

	Also:
	- alpha blending
	- using more than two views

	The overarching aim is to learn first hand more computer vision techniques. This gives
	me a broad knowledge of a lot of skills, whilst also implementing a fun project. 
	An advantage is that we don't need camera calibration here. 

	Best if we pick two images with a relatively small baseline compared to the distance
	from the main object. 

	For testing I'm using Adobe's dataset: https://sourceforge.net/projects/adobedatasets.adobe/files/adobe_panoramas.tgz/download
	Reference Implementation of FAST: https://github.com/edrosten/fast-C-src

	Log:
	- Starting with the theory of FAST features
	- implementing fast features soon i hope, iteration 1
	*/

	// pull in both images
	// Starting with goldengate 0 and 1
	Mat leftImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-00.png");
	Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-01.png");
	
	// Find features in each image



	// Display for debug reasons

	// Score features with Shi-Tomasi score, or Harris score
	// Cluster features here too

	// Create descriptors for each feature in each image

	return 0;
}


/*
	Given an image, return a vector of all FAST features in the image.
	This uses N=12 or above, and an adaptive threshold.
*/
bool FindFASTFeatures(Mat img, vector<point>& features)
{


	return true;
}