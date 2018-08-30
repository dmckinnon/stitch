#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include "Features.h"
#include <Eigen/Dense>

/* Composition Functions */

// Get the composition of two images
cv::Mat Stitch(const cv::Mat& img1, const cv::Mat& img2, Eigen::Matrix3f H);

// Perform alpha blending 
