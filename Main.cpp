#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

#define FAST_SPACING 3
#define THRESH 15
#define ST_WINDOW 5
#define ST_THRESHOLD 0.04f


struct Feature
{
	Point p;
	float score;
};

/* Function prototypes */
bool FindFASTFeatures(Mat img, vector<Point>& features);

bool ThreeOfFourValuesBrighterOrDarker(int i1, int i5, int i9, int i13, int pb, int p_b);

bool CheckForSequential12(std::vector<int> points, int p_b, int pb);

bool ScoreAndClusterFeatures(Mat img, vector<Point>& features);

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
	- FAST features basic version implemented with a threshold of 10. Do we use a dynamic
	  threshold? I'll experiment with different numbers. 15 is working for now.
	- Now score each feature with Shi-Tomasi score (then make descriptors)
	- TODO: put all feature stuff (detection, scoring, descripting, matching) into separate cpp
	*/

	// pull in both images
	// Starting with goldengate 0 and 1
	// TODO: make sure these are black and white
	Mat leftImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-00.png");
	Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-01.png");

	// Find features in each image
	vector<Feature> leftFeatures;
	if (!FindFASTFeatures(leftImage, leftFeatures))
	{
		cout << "Failed to find features in left image" << endl;
	}
	vector<Feature> rightFeatures;
	if (!FindFASTFeatures(rightImage, rightFeatures))
	{
		cout << "Failed to find features in right image" << endl;
	}

	// Score features with Shi-Tomasi score, or Harris score
	std::vector<Feature> goodLeftFeatures = ScoreAndClusterFeatures(leftImage, leftFeatures);
	if (goodLeftFeatures.empty())
	{
		cout << "Failed to score and cluster features in left image" << endl;
	}
	std::vector<Feature> goodRightFeatures = ScoreAndClusterFeatures(rightImage, rightFeatures);
	if (goodRightFeatures.empty())
	{
		cout << "Failed to score and cluster features in right image" << endl;
	}

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
bool FindFASTFeatures(Mat img, vector<Feature>& features)
{
	// For each pixel (three in from each side)
	int width = img.cols;
	int height = img.rows;
	for (int h = FAST_SPACING; h < height - FAST_SPACING; ++h)
	{
		for (int w = FAST_SPACING; w < width - FAST_SPACING; ++w)
		{
			// Adapt a brightness threshold?

			int p = img.at<uchar>(h, w); // wrong number of dims ... 
			int pb = p + THRESH;
			int p_b = p - THRESH;

			// Just make some threshold and figure out an adaptive threshold

			// For a speed-up, check 1, 9, then 5, 13
			// Any three of 1,5,9,13 can be all brighter or darker. If not ... not a corner
			int i1 = img.at<uchar>(h-FAST_SPACING, w);
			int i5 = img.at<uchar>(h, w + FAST_SPACING);
			int i9 = img.at<uchar>(h + FAST_SPACING, w);
			int i13 = img.at<uchar>(h, w - FAST_SPACING);
			if (!ThreeOfFourValuesBrighterOrDarker(i1, i5, i9, i13, pb, p_b))
			{
				continue;
			}
			else {
				// Now check the rest
				// need 12 or more sequential values above or below
				// First, get all the values
				int i2 = img.at<uchar>(h - FAST_SPACING, w + 1);
				int i3 = img.at<uchar>(h - 2, w + 2);
				int i4 = img.at<uchar>(h-1, w + FAST_SPACING);
				int i6 = img.at<uchar>(h+1, w + FAST_SPACING);
				int i7 = img.at<uchar>(h + 2, w + 2);
				int i8 = img.at<uchar>(h + FAST_SPACING, w - 1);
				int i10 = img.at<uchar>(h + FAST_SPACING, w + 1);
				int i11 = img.at<uchar>(h + 2, w - 2);
				int i12 = img.at<uchar>(h+1, w - FAST_SPACING);
				int i14 = img.at<uchar>(h-1, w - FAST_SPACING);
				int i15 = img.at<uchar>(h - 2, w - 2);
				int i16 = img.at<uchar>(h - FAST_SPACING, w - 1);
				std::vector<int> points{ i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16 };

				// Pass values into evaluation function
				if (!CheckForSequential12(points, p_b, pb))
				{
					continue;
				}

				// We have a feature. Mark this spot
				Feature feature;
				feature.p.x = w;
				feature.p.y = h;
				features.push_back(feature);
			}
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

// Helper functions for FAST features
bool greaterthan(int i, int pb, int p_b)
{
	return i > pb;
}

bool lessthan(int i, int pb, int p_b)
{
	return i < p_b;
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

	bool(*comp)(int, int, int);
	for (int i = 0; i < (int)points.size(); ++i)
	{
		if (points[i] > pb)
		{
			comp = &greaterthan;
		}
		else if (points[i] < p_b)
		{
			comp = &lessthan;
		}
		else {
			continue;
		}
		
		// Now loop over the rest of the sequence, forward and backward,
		// until both sides return false
		// Forward loop
		int fLen = 0;
		for (int j = i + 1; j!=i; ++j)
		{
			// quit when we get back to i
			if (j == 16)
				j = 0;
			if (j == i)
				break;
			if (comp(points[j], pb, p_b))
				fLen++;
			else
				break;
		}
		int bLen = 0;
		for (int j = i - 1; j != 1; --j)
		{
			// quit when we get back to i
			if (j == -1)
				j = 15;
			if (j == i)
				break;
			if (comp(points[j], pb, p_b))
				bLen++;
			else
				break;
		}
		int seqLen = fLen + bLen + 1;
		if (seqLen >= 12)
		{
			return true;
		}
	}

	return false;
}

/*
	Score features with Shi-Tomasi score.
	Any features below the cut-off are removed. 
	
	Then we do a second pass, and if any features are sufficiently close,
	we cluster them to the one with the highest score, and boost its score.

	Also should perform non-maximal suppression

	The Shi-Tomasi score uses the minimum eigenvalue of the matrix
	I_x^2     I_x I_y
	I_x I_y     I_x ^2
	where I_x is the derivative in X of the image i at x,y.
	TODO: which derivative to use?
	Sobel operator, with a default value of 3
	4x4 window size? 5x5?
	
	Parameters:
	- There is a cutoff value for the Shi-Tomasi corner detector
	- Window size for deformation matrix
	
*/
std::vector<Feature> ScoreAndClusterFeatures(Mat img, vector<Feature>& features)
{
	// let's cheat and use opencv to compute the sobel derivative, window size 3
	// over the whole image
	// lol this doesn't actually save us much time but whatevs
	Mat sobel;
	GaussianBlur(img, sobel, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat grad_x, grad_y;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_8U;
	Sobel(sobel, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(sobel, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	// We have our x and y gradients
	// Now with our window size, go over the image

	int width = img.cols;
	int height = img.rows;
	int numFeatures = features.size();
	std::vector<Feature> goodFeatures;
	for (int i = 0; i < numFeatures; ++i)
	{
		auto& f = features[i];
		int winSize = ST_WINDOW / 2;
		Mat M = Mat::zeros(2, 2, CV_32F);
		M.at<float>(0, 0) = grad_x.at<float>(f.p.y, f.p.x) * grad_x.at<float>(f.p.y, f.p.x);
		M.at<float>(0, 1) = grad_x.at<float>(f.p.y, f.p.x) * grad_y.at<float>(f.p.y, f.p.x);
		M.at<float>(1, 0) = grad_x.at<float>(f.p.y, f.p.x) * grad_y.at<float>(f.p.y, f.p.x);
		M.at<float>(1, 1) = grad_y.at<float>(f.p.y, f.p.x) * grad_y.at<float>(f.p.y, f.p.x);
		M *= img.at<int>(f.p.y, f.p.x);

		// Compute the eigenvalues of M
		// so the equation is
		// (Ix2 - E)(Iy2 - E) - Ixy2, solve for two solutions of e
		float a = 1.f; // yeah, just for show
		float b = -1 * (M.at<float>(0, 0) + M.at<float>(1, 1));
		float c = M.at<float>(0, 0)*M.at<float>(1, 1) - M.at<float>(1, 0)*M.at<float>(0, 1);
		float eigen1 = (-b + sqrt(b*b - 4 * a*c)) / 2 * a;
		float eigen2 = (-b - sqrt(b*b - 4 * a*c)) / 2 * a;

		float minEigenvalue = min(eigen1, eigen2);
		f.score = minEigenvalue;
		if (f.score <= ST_THRESHOLD)
		{
			goodFeatures.push_back(f);
		}
	}

	// For every good feature?
		// Perform clustering based on some minimum distance ...
		// What sort of clustering?
		// Can just ignore this for now
	
	// Return the highest scoring ones. 
	// Loop over these and if any are within min distance
	// Select the higher
	// Can cut down to a number or not?

	return goodFeatures;
}