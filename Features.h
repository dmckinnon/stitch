#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

#define DESC_LENGTH 128
struct FeatureDescriptor
{
	float vec[DESC_LENGTH];
};

struct Feature
{
	cv::Point p;
	float score;
	float angle;
	FeatureDescriptor desc;
};

/*
	Feature Detection functions
*/
bool FindFASTFeatures(cv::Mat img, std::vector<Feature>& features);

std::vector<Feature> ScoreAndClusterFeatures(cv::Mat img, std::vector<Feature>& features);

bool CreateSIFTDescriptors(cv::Mat img, std::vector<Feature> features, std::vector<FeatureDescriptor>& descriptors);
