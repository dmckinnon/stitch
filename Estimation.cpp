#include "Features.h"
#include <iostream>
#include <algorithm>
#include <Eigen/SVD>

using namespace cv;
using namespace std;
using namespace Eigen;

/*
	Estimation function implementations
*/

bool FindHomography(Mat H, std::vector<std::pair<Feature,
	Feature> >& matches, std::vector<Feature>& list1,
	std::vector<Feature>& list2)
{
	return true;
}

/*
	resources on SVD:
	https://courses.engr.illinois.edu/cs543/sp2011/lectures/Lecture%2021%20-%20Photo%20Stitching%20-%20Vision_Spring2011.pdf
	https://www.cse.unr.edu/~bebis/CS791E/Notes/SVD.pdf
	https://hal.inria.fr/file/index/docid/174739/filename/RR-6303.pdf
	https://blog.statsbot.co/singular-value-decomposition-tutorial-52c695315254
	https://cims.nyu.edu/~donev/Teaching/NMI-Fall2010/Lecture5.handout.pdf
	https://eigen.tuxfamily.org/dox/group__SVD__Module.html

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

	To SVD A, we use Eigen
*/
bool EstimateHomography(vector<Point> points, Mat& H)
{
	// Construct A

	return false;
}
