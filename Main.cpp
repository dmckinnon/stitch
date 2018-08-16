#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "Features.h"
#include "Estimation.h"

using namespace cv;
using namespace std;

/* Function Implementations */
int main(int argc, char** argv)
{
	/* Panorama stitching
	
	The aim of this exercise is to learn all the different parts to image stitching. 
	These are:
	- feature detection
	- feature description
	- feature matching with NN
	- optimisation via direct linear transform
	- applying transformations to images RANSAC wooo

	Also:
	- bundle adjustment
	- alpha blending
	- using more than two views

	The overarching aim is to learn first hand more computer vision techniques. This gives
	me a broad knowledge of a lot of skills, whilst also implementing a fun project. 
	An advantage is that we don't need camera calibration here. 

	Best if we pick two images with a relatively small baseline compared to the distance
	from the main object. 

	At the end I should comment the crap out of this for anyone reading in future

	For testing I'm using Adobe's dataset: https://sourceforge.net/projects/adobedatasets.adobe/files/adobe_panoramas.tgz/download
	Reference Implementation of FAST: https://github.com/edrosten/fast-C-src

	TODO:
	- Adaptive threshold for FAST features?
	- Commentary throughout functions
	- Detect features at different scales
		- Feature detection
		- feature description
	    - matching? Only match at same scale
		- create a scale pyramid
		- We don't actually need this at all, we get enough features
		- and it's easy enough to implement
	- Need to tweak threshold for Shi Tomasi

	Issues:
	- Features detected in one are not detected in the other
	- Matches are wrong


	Log:
	- Starting with the theory of FAST features
	- implementing fast features soon i hope, iteration 1
	- FAST features basic version implemented with a threshold of 10. Do we use a dynamic
	  threshold? I'll experiment with different numbers. 15 is working for now.
	- Now score each feature with Shi-Tomasi score (then make descriptors)
	- Iteration one of scoring done, testing now. Need more tweaking
	- Got it at least working. Need a better threshold?
	- Now for feature description. HOG won't work - not orientationally invariant. SURF or SIFT ... 
	  Won't use ORB, want the more difficult route. 
	  Gonna use SIFT - http://aishack.in/tutorials/sift-scale-invariant-feature-transform-features/
	- Non-maximal suppression on features added.
	- Feature orientation added
	- SIFT descriptors added. 
	- SIFT descriptors possibly working?
	- Feature matching working? We get some matches at least
	- Going to test on other images
	- So, given that these images are probably poor for FAST corners, I'm going to
	  implement multi-scale feature detection, to see if that improves anything
	- Don't do multi-scale yet, just debug FAST features
	- Write some tests for FAST features, like check for sequential12
	- Made images grayscale, and that bloody fixed it. UGH. Was using different colour channels
	- THINGS WORK SO FAR UP TO MATCHING
	- Need to tweak Shi Tomasi threshold
	- Now to compute a Direct Linear Transform between the two images. I think
	  I can do the Szeliski algorithm, or RANSAC, or a combination? Then bundle adjust it all
	  Read Szeliski, figure out the transform
	- Homography estimation and RANSAC
	*/

	// pull in both images
	// Starting with goldengate 0 and 1
	// TODO: make sure these are black and white
	// TODO: Make these command line args
	//Mat leftImage = imread("C:\\Users\\d_mcc\\OneDrive\\Pictures\\test2.JPG", IMREAD_GRAYSCALE);
	//Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\lion\\left.jpg", IMREAD_GRAYSCALE);
	Mat leftImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\lion\\left.jpg", IMREAD_GRAYSCALE);
	Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\lion\\right.jpg", IMREAD_GRAYSCALE);
	//Mat leftImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-00.png");
	//Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-01.png");


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

	// Draw the features on the image
	Mat temp = leftImage.clone();
	// Debug display
	std::string debugWindowName = "debug image";
	namedWindow(debugWindowName);
	//std::string debugWindowName1 = "debug image1";
	//namedWindow(debugWindowName1);
	Mat matchImage;
	hconcat(leftImage, rightImage, matchImage);
	int offset = leftImage.cols;
	// Draw the features on the image
	for (unsigned int i = 0; i < leftFeatures.size(); ++i)
	{
		circle(matchImage, leftFeatures[i].p, 2, (255, 255, 0), -1);
	}
	for (unsigned int i = 0; i < rightFeatures.size(); ++i)
	{
		Point p(rightFeatures[i].p.x+offset, rightFeatures[i].p.y);
		circle(matchImage, p, 2, (0, 255, 0), -1);
	}
	// Debug display
	imshow(debugWindowName, matchImage);
	//imshow(debugWindowName1, rightImage);
	waitKey(0);
	

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

	// Cull each list to top 100 matches
	vector<Feature> bestLeftFeatures, bestRightFeatures;
	for (unsigned int i = 0; i < goodLeftFeatures.size() || i > 100; ++i)
		bestLeftFeatures.push_back(goodLeftFeatures[i]);
	for (unsigned int i = 0; i < goodRightFeatures.size() || i > 100; ++i)
		bestRightFeatures.push_back(goodRightFeatures[i]);

	Mat matchImageScored;
	hconcat(leftImage, rightImage, matchImageScored);
	// Draw the features on the image
	for (unsigned int i = 0; i < goodLeftFeatures.size(); ++i)
	{
		circle(matchImageScored, goodLeftFeatures[i].p, 2, (255, 255, 0), -1);
	}
	for (unsigned int i = 0; i < goodRightFeatures.size(); ++i)
	{
		Point p(goodRightFeatures[i].p.x + offset, goodRightFeatures[i].p.y);
		circle(matchImageScored, p, 2, (0, 255, 0), -1);
	}
	// Debug display
	imshow(debugWindowName, matchImageScored);
	waitKey(0);

	// Create descriptors for each feature in each image
	std::vector<FeatureDescriptor> descLeft;
	if (!CreateSIFTDescriptors(leftImage, goodLeftFeatures, descLeft))
	{
		cout << "Failed to create feature descriptors for left image" << endl;
	}
	std::vector<FeatureDescriptor> descRight;
	if (!CreateSIFTDescriptors(rightImage, goodRightFeatures, descRight))
	{
		cout << "Failed to create feature descriptors for right image" << endl;
	}

	// Nearest neighbour matching with Lowe ratio test
	std::vector<std::pair<Feature, Feature> > matches = MatchDescriptors(goodLeftFeatures, goodRightFeatures);
	cout << "Number of matches: " << matches.size() << std::endl;

	// Debug display
	Mat matchImageFinal;
	hconcat(leftImage, rightImage, matchImageFinal);
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		Feature f1 = matches[i].first;
		Feature f2 = matches[i].second;
		f2.p.x += offset;

		circle(matchImageFinal, f1.p, 2, (255, 255, 0), -1);
		circle(matchImageFinal, f2.p, 2, (255, 255, 0), -1);
		line(matchImageFinal, f1.p, f2.p, (0,255,255), 2, 8, 0);
	}
	imshow(debugWindowName, matchImageFinal);
	waitKey(0);

	// Homography estimation and RANSAC
	// First, pad the images to allow for warping

	return 0;
}

/*
	Find the homography between the two images. 

	Using a RANSAC approach, pick four random matches. Estimate the homography between
	the images using just these four. Measure the success of this homography by how well
	it predicts the rest of the matches. If it is below some epsilon, done!
	If not, repeat for another random four. 

	How do we estimate the homography?  
	First, normalise all points
	Scale so that average distance to origin is srt(2) - apparently this makes it behave nicely?

	Create the Matrix A, which is
	[ -u1  -v1  -1   0    0    0   u1u'1  v1u'1  u'1]
	[  0    0    0  -u1  -v1  -1   u1v'1  v1v'1  v'1] * h = 0
	................................................
	[  0    0    0  -u4  -v4  -1   u4v'4  v4v'4  v'4]
	where x' = Hx and h = [h1 ... h9] as a vector

	Use Singular Value Decomposition to compute A:
	UDV^T = A
	h = V_smallest (column of V corresponding to smallest singular value)
	Then form H out of that. 
	Then unnormalise H, using the inverse of the normalisation matrix for the points
*/
