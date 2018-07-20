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

bool ThreeOfFourValuesBrighterOrDarker(int i1, int i5, int i9, int i13, int pb, int p_b);

bool CheckForSequential12(std::vector<int> points, int p_b, int pb);

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
      11     +    7
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
			// Any three of 1,5,9,13 can be all brighter or darker. If not ... not a corner
			int i1 = img.at<int>(w,h-FAST_SPACING);
			int i5 = img.at<int>(w+FAST_SPACING, h);
			int i9 = img.at<int>(w, h + FAST_SPACING);
			int i13 = img.at<int>(w-FAST_SPACING, h);
			if (!ThreeOfFourValuesBrighterOrDarker(i1, i5, i9, i13, pb, p_b))
			{
				continue;
			}
			else {
				// Now check the rest
				// need 12 or more sequential values above or below
				// First, get all the values
				int i2 = img.at<int>(w+1, h - FAST_SPACING);
				int i3 = img.at<int>(w+2, h - 2);
				int i4 = img.at<int>(w + FAST_SPACING, h-1);
				int i6 = img.at<int>(w + FAST_SPACING, h+1);
				int i7 = img.at<int>(w + 2, h + 2);
				int i8 = img.at<int>(w-1, h + FAST_SPACING);
				int i10 = img.at<int>(w+1, h + FAST_SPACING);
				int i11 = img.at<int>(w - 2, h + 2);
				int i12 = img.at<int>(w - FAST_SPACING, h+1);
				int i14 = img.at<int>(w - FAST_SPACING, h-1);
				int i15 = img.at<int>(w - 2, h - 2);
				int i16 = img.at<int>(w - 1, h - FAST_SPACING);
				std::vector<int> points{ i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16 };

				// Pass values into evaluation function
				if (!CheckForSequential12(points, p_b, pb))
				{
					continue;
				}

				// We have a feature
			}
			

			// We didn't fail the check
		}
	}

	return true;
}

/*
	If three of the four i values are all brighter than pb or darker than p_b, return true.
	Else, return false
*/
bool ThreeOfFourValuesBrighterOrDarker(int i1, int i5, int i9, int i13, int pb, int p_b)
{
	// Fast fail
	// If both i1 and i9 lie within [p_b, pb] then we do not have a corner
	if (p_b < i1 && i1 < pb && p_b < i9 && i9 < pb)
	{
		return false;
	}
	else
	{
		int above_pb = 0;
		int below_p_b = 0;

		above_pb += i1 > pb ? 1 : 0;
		above_pb += i5 > pb ? 1 : 0;
		above_pb += i9 > pb ? 1 : 0;
		above_pb += i13 > pb ? 1 : 0;

		if (above_pb >= 3)
		{
			return true;
		}
		else {
			below_p_b += i1 < p_b ? 1 : 0;
			below_p_b += i5 < p_b ? 1 : 0;
			below_p_b += i9 < p_b ? 1 : 0;
			below_p_b += i13 < p_b ? 1 : 0;

			if (below_p_b >= 3)
			{
				return true;
			}
		}
	}

	return false;
}

/*
	If there is a sequence of i values that are all above pb or below p_b, return true.
	Else, return false.
*/
bool CheckForSequential12(std::vector<int> points, int p_b, int pb)
{
	// Do we try to do this intelligently or just brute force? 
	// For each in the list
	// if it's above or below
	// Search front and back until we find a break
	// count the sequence length

	// Yes, there are smarter ways to do this. No, I don't care right now.
	int p = (pb + p_b) / 2;
	for (unsigned int i = 0; i < points.size(); ++i)
	{
		
		// Prolly easiest to just use a function pointer at this stage

	}

	return false;
}