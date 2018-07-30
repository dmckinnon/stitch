#include "Features.h"
#include <iostream>

using namespace cv;
using namespace std;

/*
	Feature function implementations

	functions:
	- Find all FAST features in an image
		- Supporting functions for this
	- Score features with Shi-Tomasi
*/

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
// Support function prototypes
bool ThreeOfFourValuesBrighterOrDarker(int i1, int i5, int i9, int i13, int pb, int p_b);
bool CheckForSequential12(std::vector<int> points, int p_b, int pb);
// Actual fast features function
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
			int i1 = img.at<uchar>(h - FAST_SPACING, w);
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
				int i4 = img.at<uchar>(h - 1, w + FAST_SPACING);
				int i6 = img.at<uchar>(h + 1, w + FAST_SPACING);
				int i7 = img.at<uchar>(h + 2, w + 2);
				int i8 = img.at<uchar>(h + FAST_SPACING, w - 1);
				int i10 = img.at<uchar>(h + FAST_SPACING, w + 1);
				int i11 = img.at<uchar>(h + 2, w - 2);
				int i12 = img.at<uchar>(h + 1, w - FAST_SPACING);
				int i14 = img.at<uchar>(h - 1, w - FAST_SPACING);
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
		for (int j = i + 1; j != i; ++j)
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
	Sobel(sobel, grad_x, ddepth, 1, 0, ST_WINDOW, scale, delta, BORDER_DEFAULT);
	Sobel(sobel, grad_y, ddepth, 0, 1, ST_WINDOW, scale, delta, BORDER_DEFAULT);

	// COMBINE THE GRADIENTS?

	// We have our x and y gradients
	// Now with our window size, go over the image

	// Get gaussian kernel for weighting the gradients within the window
	Mat gaussKernel = Mat(ST_WINDOW, ST_WINDOW, CV_32F, 1);
	for (int i = 0; i < ST_WINDOW; ++i) for (int j = 0; j < ST_WINDOW; ++j) gaussKernel.at<float>(i, j) = 1;
	//Mat gaussKernel = getGaussianKernel(ST_WINDOW, 1, CV_32F);
	GaussianBlur(gaussKernel, gaussKernel, Size(ST_WINDOW, ST_WINDOW), 1, 1, BORDER_DEFAULT);

	int width = img.cols;
	int height = img.rows;
	int numFeatures = features.size();
	std::vector<Feature> goodFeatures;
	float avgEigen = 0.f;
	for (int i = 0; i < numFeatures; ++i)
	{
		auto& f = features[i];
		int winSize = ST_WINDOW / 2;
		Mat M = Mat::zeros(2, 2, CV_32F);
		// Go through the window around the feature
		// Accumulate M weighted by the kernel
		for (int n = -(ST_WINDOW / 2); n <= ST_WINDOW / 2; ++n)
		{
			for (int m = -(ST_WINDOW / 2); m <= (ST_WINDOW / 2); ++m)
			{
				int i = n + f.p.y;
				int j = m + f.p.x;
				float w = gaussKernel.at<float>(n + (ST_WINDOW / 2), m + (ST_WINDOW / 2));
				M.at<float>(0, 0) += w * (float)(grad_x.at<uchar>(i, j) * grad_x.at<uchar>(i, j));
				M.at<float>(0, 1) += w * (float)(grad_x.at<uchar>(i, j) * grad_y.at<uchar>(i, j));
				M.at<float>(1, 0) += w * (float)(grad_x.at<uchar>(i, j) * grad_y.at<uchar>(i, j));
				M.at<float>(1, 1) += w * (float)(grad_y.at<uchar>(i, j) * grad_y.at<uchar>(i, j));
			}
		}

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
		avgEigen += f.score;
		//cout << "F score: " << f.score << " at " << f.p.x << "," << f.p.y << endl;
		if (f.score > ST_THRESHOLD)
		{
			goodFeatures.push_back(f);
		}
	}

	cout << "Average score: " << avgEigen / features.size() << endl;

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