// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat computeGLCM(const cv::Mat& img, int distance, int angle) 
{
	if (img.channels() != 1)
		throw "Only single-channel images are supported";
	if (img.depth() != CV_8U)
		throw "Only 8-bits images are supported";

	int numLevels = 256;
	int rows = img.rows;
	int cols = img.cols;

	cv::Mat glcm = cv::Mat::zeros(numLevels, numLevels, CV_32FC1);

	int dr = ucas::round(distance * std::sin(angle * CV_PI / 180));
	int dc = ucas::round(distance * std::cos(angle * CV_PI / 180));

	int count = 0;
	for (int i = 0; i < rows; i++) 
	{
		for (int j = 0; j < cols; j++) 
		{
			int r2 = i + dr;
			int c2 = j + dc;

			if (r2 >= 0 && r2 < rows && c2 >= 0 && c2 < cols) 
			{
				glcm.at<float>(img.at<uchar>(i, j), img.at<uchar>(r2, c2)) ++;
				count++;
			}
		}
	}

	glcm /= count;

	return glcm;
}

float computeGLCMCorrelation(const cv::Mat& glcm) 
{
	int numLevels = glcm.rows;

	float mr=0, mc=0;
	float sr=0, sc=0;

	for (int i = 0; i < numLevels; i++) 
		for (int j = 0; j < numLevels; j++)
		{
			mr += i * glcm.at<float>(i, j);
			mc += j * glcm.at<float>(i, j);
		}

	for (int i = 0; i < numLevels; i++)
		for (int j = 0; j < numLevels; j++)
		{
			sr += (i - mr) * (i - mr) * glcm.at<float>(i, j);
			sc += (j - mc) * (j - mc) * glcm.at<float>(i, j);
		}
	sr = std::sqrt(sr);
	sc = std::sqrt(sc);

	float correlation = 0.0;
	for (int i = 0; i < numLevels; ++i) 
	{
		for (int j = 0; j < numLevels; ++j) 
		{
			float p = glcm.at<float>(i, j);
			correlation += ((i - mr) * (j - mc) * p) / (sr * sc);
		}
	}

	return correlation;
}

float computeGLCMContrast(const cv::Mat& glcm)
{
	float contrast = 0;

	int numLevels = glcm.rows;

	for (int i = 0; i < numLevels; i++)
		for (int j = 0; j < numLevels; j++)
			contrast += (i - j) * (i - j) * glcm.at<float>(i, j);

	return contrast;
}

int main(int argc, char** argv) 
{
	int distance = 1;
	int angle = 0; // 0, 45, 90, or 135

	// process texture files
	std::vector < std::string > files;
	cv::glob(std::string(EXAMPLE_IMAGES_PATH), files);
	for (auto& f : files)
	{
		// discard files that do not contain 'texture'
		if (f.find("texture") == std::string::npos)
			continue;

		cv::Mat img = cv::imread(f, cv::IMREAD_GRAYSCALE);

		cv::Mat glcm = computeGLCM(img, distance, angle);
		printf("correlation = %f\n", computeGLCMCorrelation(glcm));
		printf("contrast = %f\n", computeGLCMContrast(glcm));
		printf("\n\n");
		aia::imshow(f, img);
	}

	return EXIT_SUCCESS;
}
