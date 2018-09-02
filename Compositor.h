#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>
#include "Features.h"
#include <Eigen/Dense>

/* Composition Functions */

// Final image size
std::pair<int, int> GetFinalImageSize(const cv::Mat& img1, const cv::Mat& img2, const Eigen::Matrix3f& H);

// Get the composition of two images
void Stitch(const cv::Mat& img1, const cv::Mat& img2, const Eigen::Matrix3f& H, cv::Mat& composite);

// Perform alpha blending 
