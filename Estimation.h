#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include "Features.h"

/* Estimation Functions */
bool FindHomography(cv::Mat H, std::vector<std::pair<Feature, Feature> >& matches, std::vector<Feature>& list1, std::vector<Feature>& list2);

// RANSAC

// Evaluate Homography

// Estimate Homography
bool EstimateHomography(std::vector<std::pair<cv::Point, cv::Point>> points, cv::Mat& H);