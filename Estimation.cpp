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

	NOTE: We scale and shift the points to have 0 mean and std dev of 1
*/
// Support functions
void GetRandomFourIndices(int& i1, int& i2, int& i3, int& i4, int max)
{
	// Initialise RNG
	//srand(time(NULL));

	// DEBUG
	srand(5);

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
//
pair<Matrix3f, Matrix3f> ConvertPoints(const vector<pair<Feature, Feature> >& matches)
{
	// For each point in first and second, collect the mean
	// and compute std deviation
	Point2f firstAvg(0.f, 0.f);
	Point2f secondAvg(0.f, 0.f);
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		firstAvg += matches[i].first.p;
		secondAvg += matches[i].second.p;
	}
	firstAvg /= (float)matches.size();
	secondAvg /= (float)matches.size();

	// Now compute std deviation
	Point2f firstStdDev(0.f, 0.f);
	Point2f secondStdDev(0.f, 0.f);
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		auto temp = matches[i].first.p - firstAvg;
		firstStdDev += Point2f(temp.x*temp.x, temp.y*temp.y);

		temp = matches[i].second.p - secondAvg;
		secondStdDev += Point2f(temp.x*temp.x, temp.y*temp.y);
	}
	firstStdDev /= (float)matches.size();
	secondStdDev /= (float)matches.size();
	firstStdDev.x = sqrt(firstStdDev.x);
	firstStdDev.y = sqrt(firstStdDev.y);
	secondStdDev.x = sqrt(secondStdDev.x);
	secondStdDev.y = sqrt(secondStdDev.y);

	// The first of the pair is the matrix for the second point;
	// The second of the pair is the matrix for the first point.
	Matrix3f conversionForSecondPoints;
	conversionForSecondPoints << 1 / secondStdDev.x,             0.f        , secondAvg.x / secondStdDev.x,
		                                  0.f      ,      1 / secondStdDev.y, secondAvg.y / secondStdDev.y,
		                                  0.f      ,             0.f        ,           1.f;
 	Matrix3f conversionForFirstPoints;
	conversionForFirstPoints << 1 / firstStdDev.x,             0.f        , firstAvg.x / firstStdDev.x,
		                                  0.f      ,      1 / firstStdDev.y, firstAvg.y / firstStdDev.y,
		                                  0.f      ,             0.f        ,           1.f;

	return make_pair(conversionForSecondPoints, conversionForFirstPoints);
}
// Actual function
bool FindHomography(Matrix3f& homography, const vector<pair<Feature,Feature> >& matches)
{
	// RANSAC
	int maxInliers = 0;
	Matrix3f bestH;
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
		// Normalise homography
		H /= H(2, 2);
		int inliers = EvaluateHomography(matches, H);
		if (inliers > maxInliers)
		{
			maxInliers = inliers;
			bestH = H;
		}

		// Not enough inliers. Loop again
	}

	cout << "max inliers: " << maxInliers << endl;

	if (maxInliers != 0)
	{
		homography = bestH;
		return true;
	}

	// TODO: Bundle Adjustment

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

		// normalise the points with the matrices provided
		auto secondPoint = Vector3f(p.second.x, p.second.y, 1.f); // left
		auto firstPoint = Vector3f(p.first.x, p.first.y, 1.f); // right

		// Continue building A
		A(2*i,   0) = -1 * firstPoint(0);
		A(2 * i, 1) = -1 * firstPoint(1);
		A(2 * i, 2) = -1;
		A(2 * i, 6) = firstPoint(0) * secondPoint(0);
		A(2 * i, 7) = firstPoint(1) * secondPoint(0);
		A(2 * i, 8) = secondPoint(0);

		A(2 * i + 1, 3) = -1 * firstPoint(0);
		A(2 * i + 1, 4) = -1 * firstPoint(1);
		A(2 * i + 1, 5) = -1;
		A(2 * i + 1, 6) = firstPoint(0) * secondPoint(1);
		A(2 * i + 1, 7) = firstPoint(1) * secondPoint(1);
		A(2 * i + 1, 8) = secondPoint(1);
	}

	// Get the V matrix of the SVD decomposition
	BDCSVD<MatrixXf> svd(A, ComputeThinU | ComputeFullV);
	if (!svd.computeV())
		return false;
	auto& V = svd.matrixV();

	// Set H to be the column of V corresponding to the smallest singular value
	// which is the last as singular values come well-ordered
	H << V(0, 8), V(1, 8), V(2, 8),
		 V(3, 8), V(4, 8), V(5, 8),
		 V(6, 8), V(7, 8), V(8, 8);

	// Normalise H
	H /= H(2, 2);

	return true;
}

/*
	Evaluate a potential Homography, given the two lists of points. 
	The homography transforms from the second in each pair to the first. 

	Compute the euclidean difference between Hx and x', for each pair. 
	From this, sort and get the median, then the average. Return the median, 
	as we want to be robust to outliers. 
*/
int EvaluateHomography(const vector<pair<Feature,Feature> >& matches, const Matrix3f& H)
{
	vector<float> diffs;
	int numInliers = 0;
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		// Convert both points to Eigen points, in normalised homogeneous coords
		// TODO: should these be in pixel, or normalised?
		Vector3f x(matches[i].second.p.x, matches[i].second.p.y, 1);
		Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y, 1);

		Vector3f Hx = H * x;

		// Normalise
		Hx /= Hx(2);

		// Use total reprojection error
		// This is L2(x' - Hx) + L2(x - Hinverse x')
		auto projectiveDiff = xprime - H * x;
		auto reprojectiveDiff = x - H.inverse() * xprime;
		float totalError = projectiveDiff.norm() + reprojectiveDiff.norm();
		if (totalError < POSITIONAL_UNCERTAINTY * RANSAC_INLIER_MULTIPLER)
		{
			numInliers++;
		}
	}

	return numInliers;
}


/*
	Bundle Adjustment
	http://www.cs.unc.edu/~marc/tutorial/node159.html
	https://engineering.purdue.edu/kak/computervision/ECE661.08/solution/hw5_s2.pdf

	The formula for J comes from Multiple View Geometry, page 146, equ. 5.11
	As of now, I don't understand it. 

	Here is an explanation and a derivation. This also explains how to get the covariance
	I think
	https://pdfs.semanticscholar.org/66e4/283c28a2a93c4d4674f4213e1e9f67cfc737.pdf


	We ignore covariance for now

*/
// Helper functions
float ErrorInHomography(const vector<pair<Feature, Feature> >& matches, const Matrix3f& H)
{
	float error = 0;
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		Vector3f x(matches[i].second.p.x, matches[i].second.p.y, 1);
		Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y, 1);

		// Get the error term
		auto projectiveDiff = xprime - H * x;
		auto reprojectiveDiff = x - H.inverse() * xprime;
		error += projectiveDiff.norm() + reprojectiveDiff.norm();
	}

	return error;
}
// Actual function
void BundleAdjustment(const vector<pair<Feature, Feature> >& matches, Matrix3f& H)
{

	// L-M update parameter
	float lambda = 0;// 0.001f;
	float prevError = 100000000;// ErrorInHomography(matches, H);
	for (int its = 0; its < MAX_BA_ITERATIONS; ++its)
	{
		VectorXf update(9);
		Vector2f error_accum;
		error_accum.setZero();
		MatrixXf JtJ(9, 9);
		JtJ.setZero();
		VectorXf Jte(9);
		Jte.setZero();

		

		// Over all feature points
		for (unsigned int i = 0; i < matches.size(); ++i)
		{
			// As above, first is x, the point on the right,
			// and second is x', the point on the left
			Vector3f x(matches[i].second.p.x, matches[i].second.p.y , 1);
			Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y , 1);

			// Get the error term
			Vector3f Hx = H * x;
			float w = Hx(2);
			Hx /= w;
			Vector3f e = xprime - Hx;
			const Vector2f e2(e(0), e(1));

			// Build the Jacobian
			MatrixXf J(2, 9);
			J.setZero();
			J << x(0), x(1), x(2), 0, 0, 0, -Hx(0)*x(0), -Hx(0)*x(1), -Hx(0),
				0, 0, 0, x(0), x(1), x(2), -Hx(1)*x(0), -Hx(1)*x(1), -Hx(1);
			J /= w;
			// ,
				//0, 0, 0, 0, 0, 0, 0, 0, 0, -xprime(2), -xprime(2), -xprime(2);
			//J << -x(0), -x(1), -x(2), 0, 0, 0, 0, 0, 0,
			//	0, 0, 0, -x(0), -x(1), -x(2), 0, 0, 0,
			//	0, 0, 0, 0, 0, 0, -x(0), -x(1), -x(2);

			// Accumulate
			JtJ += J.transpose() * J;
			Jte += J.transpose() * e2;

			error_accum += e2;
		}

		// Levenberg-Marquardt update
		// TODO: pick a lambda, update lambda
		for (int i = 0; i < JtJ.rows(); ++i)
		{
			JtJ(i, i) += lambda *JtJ(i, i);
		}

		// Compute the update
		update = JtJ.inverse() * Jte;
		Matrix3f updateToH;
		updateToH << update(0), update(1), update(2),
			         update(3), update(4), update(5),
			         update(6), update(7), update(8);

		// Test the update. If our error increased at all,
		// cut out and we'll stop optimising.
		// If the error decreases ... well, update the true H and keep going
		float currError = ErrorInHomography(matches, H + updateToH); // or multiplied the other way?
		if (currError < prevError)
		{
			// update and continue
			
			//lambda /= 10;
		}
		else
		{
			//lambda *= 10;
			//break;
		}
		prevError = currError;
		cout << update << endl;
		H += updateToH;
		H /= H(2, 2);
		cout << H << endl;
	}
}

void FiniteDiff(const Matrix3f& H)
{
	const Vector3f x(1.f, 2.f, 1.f);

	// Let f(h) = Hx
	// Compute f(h+epsilon) and f(h), then divide the difference by epsilon
	Vector3f Hx = H * x;
	float w = Hx(2);
	Hx /= w;
	float e = 0.01f;
	Matrix3f epsilon;
	epsilon.setConstant(e);
	Vector3f Hx_plus_e = (H + epsilon)*x;
	//Hx_plus_e /= Hx_plus_e(2);
	MatrixXf difference(2,9);
	difference.setZero();
	difference(0, 0) = (((H(0,0)+e)*x(0) + H(0,1)*x(1) + H(0,2)*x(2))/w - Hx(0))/e;
	difference(0, 1) = ((H(0, 0)*x(0) + (H(0, 1)+e)*x(1) + H(0, 2)*x(2)) / w - Hx(0)) / e;
	difference(0, 2) = ((H(0, 0)*x(0) + H(0, 1)*x(1) + (H(0, 2)+e)*x(2)) / w - Hx(0)) / e;

	difference(1, 3) = (((H(1, 0) + e)*x(0) + H(1, 1)*x(1) + H(1, 2)*x(2)) / w - Hx(1)) / e;
	difference(1, 4) = ((H(1, 0)*x(0) + (H(1, 1) + e)*x(1) + H(1, 2)*x(2)) / w - Hx(1)) / e;
	difference(1, 5) = ((H(1, 0)*x(0) + H(1, 1)*x(1) + (H(1, 2) + e)*x(2)) / w - Hx(1)) / e;

	float w_e7 = 1 / ((H(2,0)+e)*x(0) + H(2,1)*x(1) + H(2,2)*x(2));
	float w_e8 = 1 / (H(2, 0)*x(0) + (H(2, 1)+e)*x(1) + H(2, 2)*x(2));
	float w_e9 = 1 / (H(2, 0)*x(0) + H(2, 1)*x(1) + (H(2, 2)+e)*x(2));

	float x1 = H(0, 0)*x(0) + H(0, 1)*x(1) + H(0, 2)*x(2);
	float x2 = H(1, 0)*x(0) + H(1, 1)*x(1) + H(1, 2)*x(2);
	difference(0, 6) = (x1 / w_e7 - Hx(0)) / e;
	difference(0, 7) = (x1 / w_e8 - Hx(0)) / e;
	difference(0, 8) = (x1 / w_e9 - Hx(0)) / e;
	difference(1, 6) = (x2 / w_e7 - Hx(1)) / e;
	difference(1, 7) = (x2 / w_e8 - Hx(1)) / e;
	difference(1, 8) = (x2 / w_e9 - Hx(1)) / e;

	// Next, compute the Jacobian using Hartley and Zisserman's method,
	// at x. 
	MatrixXf J(2, 9);
	J.setZero();
	J << x(0), x(1), x(2), 0, 0, 0, -Hx(0)*x(0), -Hx(0)*x(1), -Hx(0)*x(2),
		0, 0, 0, x(0), x(1), x(2), -Hx(1)*x(0), -Hx(1)*x(1), -Hx(1)*x(2);
	J /= w;

	// finally, return the difference between these matrices. The difference should be vanishing
	cout << "J: " << endl << J << endl;
	cout << "Finite difference: " << endl << difference << endl;
	cout << J - difference << endl;
}