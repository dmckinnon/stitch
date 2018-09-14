#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "Features.h"
#include "Estimation.h"
#include "Compositor.h"

using namespace cv;
using namespace std;
using namespace Eigen;

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
		- This may require Levenberg-Marquardt to tighten things up
	- applying transformations to images  - actual stitching

	Also:
	- 
	- bundle adjustment - yesss
	- alpha blending - poisson blending
	- using more than two views - yesss
	- parallelise it!

	The overarching aim is to learn first hand more computer vision techniques. This gives
	me a broad knowledge of a lot of skills, whilst also implementing a fun project. 
	An advantage is that we don't need camera calibration here. 

	Best if we pick two images with a relatively small baseline compared to the distance
	from the main object. 

	At the end I should comment the crap out of this for anyone reading in future

	For testing I'm using Adobe's dataset: https://sourceforge.net/projects/adobedatasets.adobe/files/adobe_panoramas.tgz/download
	Reference Implementation of FAST: https://github.com/edrosten/fast-C-src
	Panorama stitching: https://courses.engr.illinois.edu/cs543/sp2011/lectures/Lecture%2021%20-%20Photo%20Stitching%20-%20Vision_Spring2011.pdf
http://ppwwyyxx.com/2016/How-to-Write-a-Panorama-Stitcher/#Blending

	TODO:
	- Adaptive threshold for FAST features?2
	- Need to tweak threshold for Shi Tomasi
	- Tweak RANSAC threshold
	- images as command-line args
	- error handling? Eh
	- Cleaning and commentary
	- fix normalisation scaling
	- Update levenberg marquardt parameter


	Issues:
	- homography estimation is biased


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
	- Trying to get SVD for the homography estimation
	- Imported eigen
	- Implemented RANSAC to get best homography. Untested
	- Homography returns something, at least. RANSAC epsilon might need tuning
	- Starting compositing
	- Without scaling of homography coordinates, it all actually works!!
	- changed RANSAC criterion
	- Next on the list is Levenberg-Marquardt, and then blending,
	  although I should get normalisation working first
    - Bundle adjustment isn't working, though I think I have the wrong Jacobian ... ?
	- LM is minimising something, but I think the initial error is so wrong that we are stuck elsewhere
	  Try to get a better initial homography. So try noramlisation
	- normalisation works, bundle adjsutment is still bad
	*/

	// Point normalisation test
	// Create four pairs of matching points, each pair the corner of a rectangle
	// normalise them, and they should be a square
	/*Point2f p1(0, 0); // unit square should return unit square?
	Point2f p2(0, 2); // 0,2
	Point2f p3(3, 2); // 3,2
	Point2f p4(3, 0); // 3,0
	vector<pair<Feature, Feature> > testMatch;
	Feature a1, b1, a2, b2, a3, b3, a4, b4;
	a1.p = p1;
	a2.p = p2;
	a3.p = p3;
	a4.p = p4;
	b1.p = p1;
	b2.p = p2;
	b3.p = p3;
	b4.p = p4;
	testMatch.push_back(make_pair(a1, b1));
	testMatch.push_back(make_pair(a2, b2));
	testMatch.push_back(make_pair(a3, b3));
	testMatch.push_back(make_pair(a4, b4));
	auto matrices = ConvertPoints(testMatch);

	// points should be square
	Vector3f a(p1.x, p1.y, 1);
	Vector3f b(p2.x, p2.y, 1);
	Vector3f c(p3.x, p3.y, 1);
	Vector3f d(p4.x, p4.y, 1);
	cout << "Converted points: " << endl << matrices.first * a << endl;
	cout << endl << matrices.first*b << endl;
	cout << endl << matrices.first*c << endl;
	cout << endl << matrices.first*d << endl;

	return 0;*/

	/*
	 6.31809e-05  6.31809e-05  6.31809e-05            0            0            0      -57.5527     -57.5527     -57.5527
           0            0            0  1.54972e-05  1.54972e-05 -3.21865e-05     -12.9533     -12.9533     -12.9533
	*/

	// pull in both images
	// TODO: Make these command line args
	//Mat leftImage = imread("C:\\Users\\d_mcc\\OneDrive\\Pictures\\test2.JPG", IMREAD_GRAYSCALE);
	//Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\lion\\left.jpg", IMREAD_GRAYSCALE);
	Mat leftImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\lion\\left.jpg", IMREAD_GRAYSCALE);
	Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\lion\\right.jpg", IMREAD_GRAYSCALE);
	//Mat leftImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-00.png");
	//Mat rightImage = imread("C:\\Users\\d_mcc\\source\\adobe_panoramas\\data\\goldengate\\goldengate-01.png");

	// For debuggging
	std::string debugWindowName = "debug image";
	namedWindow(debugWindowName);

	// Find features in each image
	vector<Feature> leftFeatures;
	if (!FindFASTFeatures(leftImage, leftFeatures))
	{
		cout << "Failed to find features in left image" << endl;
		return 0;
	}
	vector<Feature> rightFeatures;
	if (!FindFASTFeatures(rightImage, rightFeatures))
	{
		cout << "Failed to find features in right image" << endl;
		return 0;
	}

#ifdef DEBUG
	// Draw the features on the image
	Mat temp = leftImage.clone();
	// Debug display
	
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
	waitKey(0);
#endif

	// Score features with Shi-Tomasi score, or Harris score
	std::vector<Feature> goodLeftFeatures = ScoreAndClusterFeatures(leftImage, leftFeatures);
	if (goodLeftFeatures.empty())
	{
		cout << "Failed to score and cluster features in left image" << endl;
		return 0;
	}
	std::vector<Feature> goodRightFeatures = ScoreAndClusterFeatures(rightImage, rightFeatures);
	if (goodRightFeatures.empty())
	{
		cout << "Failed to score and cluster features in right image" << endl;
		return 0;
	}

	// Sort features and cull each list to top MAC_NUM_FEATURES features
	if (goodLeftFeatures.size() > MAX_NUM_FEATURES)
	{
		sort(goodLeftFeatures.begin(), goodLeftFeatures.end(), FeatureCompare);
		goodLeftFeatures.resize(MAX_NUM_FEATURES);
	}
	if (goodRightFeatures.size() > MAX_NUM_FEATURES)
	{
		sort(goodRightFeatures.begin(), goodRightFeatures.end(), FeatureCompare);
		goodRightFeatures.resize(MAX_NUM_FEATURES);
	}

#ifdef DEBUG
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
#endif

	// Create descriptors for each feature in each image
	std::vector<FeatureDescriptor> descLeft;
	if (!CreateSIFTDescriptors(leftImage, goodLeftFeatures, descLeft))
	{
		cout << "Failed to create feature descriptors for left image" << endl;
		return 0;
	}
	std::vector<FeatureDescriptor> descRight;
	if (!CreateSIFTDescriptors(rightImage, goodRightFeatures, descRight))
	{
		cout << "Failed to create feature descriptors for right image" << endl;
		return 0;
	}

	// Nearest neighbour matching with Lowe ratio test
	// The first in each pair in matches is from the left; the second, from the right. 
	std::vector<std::pair<Feature, Feature> > matches = MatchDescriptors(goodLeftFeatures, goodRightFeatures);
	cout << "Number of matches: " << matches.size() << std::endl;

	// Debug display
#ifdef DEBUG
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
#endif

	Matrix3f H;
	if (!FindHomography(H, matches))
	{
		cout << "Failed to find sufficiently accurate homography for matches" << endl;
		return 0;
	}
	cout << "Homography: \n" << H << std::endl;

	// Refine the homography with bundle adjustment
	BundleAdjustment(matches, H);

	cout << "New homography: \n" << H << std::endl;

	// Stitch the images together
	pair<int, int> size = GetFinalImageSize(leftImage, rightImage, H);
	Mat composite(size.second, size.first, CV_8U, Scalar(0));
	Stitch(leftImage, rightImage, H, composite);

//#ifdef DEBUG
	imshow(debugWindowName, composite);
	waitKey(0);
//#endif
	// Alpha blending. Poisson blending looks good here
	H;
	// TODO: multiple images
	return 0;
}

