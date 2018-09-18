#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>

#define MAX_NUM_FEATURES 400

#define FAST_SPACING 3
#define THRESH 30
#define ST_WINDOW 3
#define ST_THRESH 20000.f
#define NMS_WINDOW 2
#define ANGLE_WINDOW 9
#define ORIENTATION_HIST_BINS 36
#define DESC_BINS 8
#define DESC_BIN_SIZE 45
#define DESC_WINDOW 16
#define DESC_SUB_WINDOW 4
#define ILLUMINANCE_BOUND 0.2f

#define NN_RATIO 0.6


#define PI 3.14159f
#define RAD2DEG(A) (A*180.f/PI)
#define DEG2RAD(A) (A*PI/180.f)

#define DESC_LENGTH 128
struct FeatureDescriptor
{
	float vec[DESC_LENGTH];
};

struct Feature
{
	int scale;
	cv::Point2f p;
	float score;
	float angle;
	FeatureDescriptor desc;
};

// Feature comparator
bool FeatureCompare(Feature a, Feature b);

/*
	Feature Detection functions
*/
bool FindFASTFeatures(cv::Mat img, std::vector<Feature>& features);

std::vector<Feature> ScoreAndClusterFeatures(cv::Mat img, std::vector<Feature>& features);

bool CreateSIFTDescriptors(cv::Mat img, std::vector<Feature>& features, std::vector<FeatureDescriptor>& descriptors);

std::vector<std::pair<Feature, Feature> > MatchDescriptors(std::vector<Feature> list1, std::vector<Feature> list2);

/*
	Feature Detection Unit Test functions
*/
void TestSequential12(void);