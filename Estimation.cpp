#include "Features.h"
#include <iostream>
#include <algorithm>
#include <Eigen/SVD>
#include "Estimation.h"
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;
using namespace Eigen;

/*
	Estimation function implementations
*/

/*
	Find the best homography between the two images. This is returned in H.

	Using a RANSAC approach, pick four random matches. Estimate the homography between
	the images using just these four. Measure the success of this homography by how well
	it predicts the rest of the matches. If it is below some epsilon, done!
	If not, repeat for another random four.

	We'll do this a maximum number of times, and remember the best.
	If we never find a homography that produces matches below the epsilon, well,
	maybe this image pair just ain't good, yeah?

	NOTE: The points in matches must be in normalised image coordinates
*/
// Support functions
void GetRandomFourIndices(int& i1, int& i2, int& i3, int& i4, int max)
{
	i1 = rand() % max;
	do
	{
		i2 = rand() % max;
	} while (i2 == i1);

	do
	{
		i3 = rand() % max;
	} while (i3 == i1 || i3 == i2);

	do
	{
		i4 = rand() % max;
	} while (i4 == i1 || i4 == i2 || i4 == i3);
}

// Actual function
bool FindHomography(Matrix3f& homography, const std::vector<std::pair<Feature,Feature> >& matches)
{
	// Initialise RNG
	srand(time(NULL));

	// RANSAC
	int numMatches = matches.size();
	for (int k = 0; k < MAX_RANSAC_ITERATIONS; ++k)
	{
		// Pick four random matches by generating four random indices
		// and ensuring they are not equal
		int i1, i2, i3, i4;
		GetRandomFourIndices(i1, i2, i3, i4, numMatches);
		
		// Get the points for those features and generate the homography
		// Since we match from left to right, and the homography goes from right
		// to left, the first in the pair is the feature on the right, and the second on the left
		vector<pair<Point, Point>> points;
		Matrix3f H;
		points.push_back(make_pair(matches[i1].second.p, matches[i1].first.p));
		points.push_back(make_pair(matches[i2].second.p, matches[i2].first.p));
		points.push_back(make_pair(matches[i3].second.p, matches[i3].first.p));
		points.push_back(make_pair(matches[i4].second.p, matches[i4].first.p));
		if (!GetHomographyFromMatches(points, H))
			continue;
		
		// Test the homography with all points
		float score = EvaluateHomography(matches, H);
		std::cout << "Score: " << score << std::endl; // this is for knowing what a good score is?
		// at most 100 matches, ideally a max distance of 1 ... 100 then?
		if (score < H_QUALITY_SCORE)
		{
			// We have enough inliers. End here
			homography = H;
			return true;
		}

		// Not enough inliers. Loop again
	}

	// We failed to find anything good
	return false;
}

/*
	resources on SVD:
	https://courses.engr.illinois.edu/cs543/sp2011/lectures/Lecture%2021%20-%20Photo%20Stitching%20-%20Vision_Spring2011.pdf
	https://www.cse.unr.edu/~bebis/CS791E/Notes/SVD.pdf
	https://hal.inria.fr/file/index/docid/174739/filename/RR-6303.pdf
	https://blog.statsbot.co/singular-value-decomposition-tutorial-52c695315254
	https://cims.nyu.edu/~donev/Teaching/NMI-Fall2010/Lecture5.handout.pdf
	https://eigen.tuxfamily.org/dox/group__SVD__Module.html

	Find the homography for four sets of corresponding points

	How do we estimate the homography?
	TODO: First, normalise all points
	TODO: Scale so that average distance to origin is srt(2) - apparently this makes it behave nicely?

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

	NOTE:
	In the pairs, the first is x, the point from the image we come from
	the second is x prime, the point in the image we are transforming to

	To SVD A, we use Eigen. 
	Really, we could also just do eigenvalue decomposition on AT * A
	Since V's columns are eigenvectors of AT * A
	But whatever
*/
bool GetHomographyFromMatches(const vector<pair<Point, Point>> points, Matrix3f& H)
{
	// Construct A
	Matrix<float, 8, 9> A;
	A.setZero();
	for (unsigned int i = 0; i < points.size(); ++i)
	{
		auto& p = points[i];
		A(2*i,   0) = -1 * p.first.x;
		A(2 * i, 1) = -1 * p.first.y;
		A(2 * i, 2) = -1;
		A(2 * i, 6) = p.first.x * p.second.x;
		A(2 * i, 7) = p.first.y * p.second.x;
		A(2 * i, 8) = p.second.x;

		A(2 * i + 1, 3) = -1 * p.first.x;
		A(2 * i + 1, 4) = -1 * p.first.y;
		A(2 * i + 1, 5) = -1;
		A(2 * i + 1, 6) = p.first.x * p.second.y;
		A(2 * i + 1, 7) = p.first.y * p.second.y;
		A(2 * i + 1, 8) = p.second.y;
	}

	// Get the V matrix of the SVD decomposition
	BDCSVD<MatrixXf> svd(A, ComputeThinU | ComputeFullV);
	if (!svd.computeV())
		return false;
	auto& V = svd.matrixV();
	//std::cout << "V: " << svd.matrixV() << std::endl;

	// Set H to be the column of V corresponding to the smallest singular value
	// which is the last as singular values come well-ordered
	H << V(0, 8), V(1, 8), V(2, 8),
		 V(3, 8), V(4, 8), V(5, 8),
		 V(6, 8), V(7, 8), V(8, 8);

	return true;
}

/*
	Evaluate a potential Homography, given the two lists of points. 
	The homography transforms from the second in each pair to the first. 

	Compute the euclidean difference between Hx and x', for each pair. 
	From this, sort and get the median, then the average. Return the median, 
	as we want to be robust to outliers. 
*/
float EvaluateHomography(std::vector<std::pair<Feature,Feature> >& matches, Matrix3f& H)
{
	vector<float> diffs;
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		// Convert both points to Eigen points, in normalised homogeneous coords
		// TODO: should these be in pixel, or normalised?
		Vector3f x(matches[i].second.p.x, matches[i].second.p.y, 1);
		Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y, 1);

		auto Hx = H * x;
		auto diffVector = xprime - Hx;
		diffs.push_back(diffVector.norm());
	}

	sort(diffs.begin(), diffs.end());
	return diffs[diffs.size()/2];
}
