#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

#define FAST_SPACING 3
#define THRESH 15
#define ST_WINDOW 3
#define ST_THRESHOLD 10000.f

struct Feature
{
	cv::Point p;
	float score;
};

/*
	Feature Detection functions
*/
bool FindFASTFeatures(cv::Mat img, std::vector<Feature>& features);

std::vector<Feature> ScoreAndClusterFeatures(cv::Mat img, std::vector<Feature>& features);