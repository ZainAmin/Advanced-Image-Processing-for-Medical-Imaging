// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc/slic.hpp>


int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/parrot.png");

	// small gaussian blur
	cv::Mat img_blurred;
	cv::GaussianBlur(img, img_blurred, cv::Size(3, 3), 0, 0);

	// switch to Lab
	cv::cvtColor(img_blurred, img_blurred, cv::COLOR_BGR2Lab);

	// instance and run SLIC
	cv::Ptr<cv::ximgproc::SuperpixelSLIC> algo =
		cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::SLICO, 10);
	algo->iterate(10);

	// get and draw superpixels
	cv::Mat mask;
	algo->getLabelContourMask(mask);
	cv::Mat img_tesselated = img.clone();
	img_tesselated.setTo(cv::Scalar(0, 255, 255), mask);
	aia::imshow("Superpixels", img_tesselated, true, 2.f);

	// replace original image pixels with superpixels means
	cv::Mat labels;
	algo->getLabels(labels);
	cv::Mat img_clustered = img.clone();
	for (int k = 0; k < algo->getNumberOfSuperpixels(); k++)
	{
		cv::Mat class_mask;
		cv::inRange(labels, cv::Scalar(k), cv::Scalar(k), class_mask);
		img_clustered.setTo(cv::mean(img, class_mask), class_mask);
	}
	aia::imshow("Clustered image", img_clustered, true, 2.f);

	return EXIT_SUCCESS;
}