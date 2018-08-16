#include "Features.h"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

/*
	Estimation function implementations
*/

bool FindHomography(Mat H, std::vector<std::pair<Feature,
	Feature> >& matches, std::vector<Feature>& list1,
	std::vector<Feature>& list2)
{
	return true;
}

bool EstimateHomography(vector<Point> points, Mat& H)
{

}
