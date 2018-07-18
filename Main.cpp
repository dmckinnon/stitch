#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

#define FAST_SPACING 3

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
	// TODO: make sure these are black and white
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
	Assumed: img is grayscale
	     16  1  2
	  15     +    3 
   14        +      4
   13  +  +  p  + + 5
   12        +      6
      11     +    4
	     10  9  8

   We start with a threshold of 10?
   Something to experiment with
*/
#define THRESH 10
bool FindFASTFeatures(Mat img, vector<point>& features)
{
	// For each pixel (three in from each side)
	int width = img.cols;
	int height = img.rows;
	for (int h = FAST_SPACING; h < height- FAST_SPACING; ++h)
	{
		for (int w = FAST_SPACING; w < width - FAST_SPACING; ++w)
		{
			// Adapt a brightness threshold?

			int p = img.at<int>(w, h);
			int pb = p + THRESH;
			int p_b = p - THRESH;

			// Just make some threshold and figure out an adaptive threshold

			// For a speed-up, check 1, 9, then 5, 13
			int i1 = img.at<int>(w,h-FAST_SPACING);
			int i5 = img.at<int>(w+FAST_SPACING, h);
			int i9 = img.at<int>(w, h + FAST_SPACING);
			int i13 = img.at<int>(w-FAST_SPACING, h);
			if (i1 > pb && i9 > pb)
			{

			}
			else if (i5 < p_b || i13 < p_b)
			{

			}

			// We didn't fail the check
		}
	}

	return true;
}