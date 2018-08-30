#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include "Features.h"
#include <Eigen/Dense>

#define MAX_RANSAC_ITERATIONS 50
#define H_QUALITY_SCORE 1.6f

/* Estimation Functions */
bool FindHomography(Eigen::Matrix3f& homography, const std::vector<std::pair<Feature, Feature> >& matches);

// Estimate Homography
bool GetHomographyFromMatches(const std::vector<std::pair<cv::Point, cv::Point>> points, Eigen::Matrix3f& H, const std::pair<Eigen::Matrix3f, Eigen::Matrix3f>& normaliseMatrices);

// Evaluate Homography
float EvaluateHomography(const std::vector<std::pair<Feature, Feature> >& matches, const Eigen::Matrix3f& H);