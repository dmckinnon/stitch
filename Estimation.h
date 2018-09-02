#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include "Features.h"
#include <Eigen/Dense>

#define MAX_RANSAC_ITERATIONS 50
#define RANSAC_INLIER_MULTIPLER 2.447f
#define POSITIONAL_UNCERTAINTY 20.f

/* Estimation Functions */
bool FindHomography(Eigen::Matrix3f& homography, const std::vector<std::pair<Feature, Feature> >& matches);

// Estimate Homography
bool GetHomographyFromMatches(const std::vector<std::pair<cv::Point, cv::Point>> points, Eigen::Matrix3f& H);

// Evaluate Homography
int EvaluateHomography(const std::vector<std::pair<Feature, Feature> >& matches, const Eigen::Matrix3f& H);