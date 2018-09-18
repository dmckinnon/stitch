#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include "Features.h"
#include <Eigen/Dense>

#define MAX_RANSAC_ITERATIONS 2000
#define RANSAC_INLIER_MULTIPLER 2.447f
#define POSITIONAL_UNCERTAINTY 0.1f
#define MAX_BA_ITERATIONS 10

#define HUBER_K 1.345f
#define TUKEY_K 4.685

/* Estimation Functions */
bool FindHomography(Eigen::Matrix3f& homography, std::vector<std::pair<Feature, Feature> > matches);

// Normalise points
std::pair<Eigen::Matrix3f, Eigen::Matrix3f> ConvertPoints(const std::vector<std::pair<Feature, Feature> >& matches);

// Estimate Homography
bool GetHomographyFromMatches(const std::vector<std::pair<cv::Point, cv::Point>> points, Eigen::Matrix3f& H);

// Evaluate Homography
int EvaluateHomography(const std::vector<std::pair<Feature, Feature> >& matches, const Eigen::Matrix3f& H);

// Bundle Adjustment
void BundleAdjustment(const std::vector<std::pair<Feature, Feature> >& matches, Eigen::Matrix3f& H);

// Robust cost functions
void Huber(const float& e, const float& stddev, float& objectiveValue, float& weight);
void Tukey(const float& e, const float& stddev, float& objectiveValue, float& weight);

// Unit test for Jacobians
void FiniteDiff(const Eigen::Matrix3f& H);