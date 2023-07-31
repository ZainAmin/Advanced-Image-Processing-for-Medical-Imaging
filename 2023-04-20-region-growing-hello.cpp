// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat localVariance(const cv::Mat img, int k)
{
	cv::Mat img_sq;
	img.convertTo(img_sq, CV_32F);
	cv::pow(img_sq, 2, img_sq);

	cv::Mat img_sq_avg;
	cv::boxFilter(img_sq, img_sq_avg, CV_32F, cv::Size(k, k));

	cv::Mat img_avg;
	cv::boxFilter(img, img_avg, CV_32F, cv::Size(k, k));
	cv::Mat img_avg_sq;
	cv::pow(img_avg, 2, img_avg_sq);

	cv::Mat img_variance = img_sq_avg - img_avg_sq;

	cv::normalize(img_variance, img_variance, 0, 255, cv::NORM_MINMAX);
	img_variance.convertTo(img_variance, CV_8U);

	return img_variance;
}

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning.jpg", 
		cv::IMREAD_GRAYSCALE);

	cv::Mat seed_img;
	cv::threshold(img, seed_img, 254, 255, cv::THRESH_BINARY);
	aia::imshow("Seed image", seed_img);

	cv::Mat predicate1_img;
	cv::threshold(img, predicate1_img, 100, 255, cv::THRESH_BINARY);
	aia::imshow("Predicate 1 image", predicate1_img);

	cv::Mat img_variance = localVariance(img, 5);
	aia::imshow("Variance image", img_variance);
	cv::Mat predicate2_img;
	cv::threshold(img_variance, predicate2_img, 5, 255, cv::THRESH_BINARY);
	aia::imshow("Predicate 2 image", predicate2_img);

	cv::Mat predicate_img = predicate1_img & predicate2_img;
	cv::Mat grow_img = seed_img;
	cv::Mat grow_prev;
	do 
	{
		grow_prev = grow_img.clone();

		// for 8-connectivity growing, we use a 3x3 SE
		// dilation is to get the slightly larger regions
		cv::Mat candidate_neighbors;
		cv::dilate(grow_img, candidate_neighbors,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));
		// by difference, we isolate only the neighbors
		candidate_neighbors -= grow_img;

		// the growing step
		grow_img += candidate_neighbors & predicate_img;

		cv::imshow("Growing in progress", grow_img);
		cv::waitKey(10);

	} while (cv::countNonZero(grow_img - grow_prev));

	aia::imshow("Final result", grow_img);

	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_region_growing.png", grow_img);
		
	return EXIT_SUCCESS;
}


