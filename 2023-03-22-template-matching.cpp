// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>


int main()
{
	cv::Mat plate_letters = cv::imread(
		std::string(EXAMPLE_IMAGES_PATH) + "/" + "plate_letters_IT.png", cv::IMREAD_GRAYSCALE);
	cv::Mat plate_digits = cv::imread(
		std::string(EXAMPLE_IMAGES_PATH) + "/" + "plate_numbers_IT.png", cv::IMREAD_GRAYSCALE);
	aia::imshow("Letters", plate_letters);
	aia::imshow("Digits", plate_digits);

	for (int l = 0; l < 8; l++)
	{
		cv::Mat img_template = cv::imread(
			ucas::strprintf("%s/plates/plate_letter_%02d.png", EXAMPLE_IMAGES_PATH, l), cv::IMREAD_GRAYSCALE);
		float rescale_ratio = 103.0f / img_template.rows;
		cv::resize(img_template, img_template, cv::Size(0, 0), rescale_ratio, rescale_ratio);
		aia::imshow("Check", img_template, true, 3.0f);

		cv::Mat corrmap;
		cv::matchTemplate(plate_letters, img_template, corrmap, cv::TM_CCOEFF_NORMED);
		aia::imshow("NCC map", ucas::heatMap(cv::abs(corrmap), false, false));

		double minV, maxV;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(corrmap, &minV, &maxV, &minLoc, &maxLoc);
		maxLoc.x += img_template.cols / 2;
		maxLoc.y += img_template.rows / 2;
		cv::Mat visual_check = plate_letters.clone();
		cv::circle(visual_check, maxLoc, 5, cv::Scalar(120), cv::FILLED);
		aia::imshow("Letters map", visual_check);

	}

	for (int d = 0; d < 6; d++)
	{
		cv::Mat img_template = cv::imread(
			ucas::strprintf("%s/plates/plate_digit_%02d.png", EXAMPLE_IMAGES_PATH, d), cv::IMREAD_GRAYSCALE);
		float rescale_ratio = 103.0f / img_template.rows;
		cv::resize(img_template, img_template, cv::Size(0, 0), rescale_ratio, rescale_ratio);
		aia::imshow("Check", img_template, true, 3.0f);
		
		cv::Mat corrmap;
		cv::matchTemplate(plate_digits, img_template, corrmap, cv::TM_CCOEFF_NORMED);
		aia::imshow("NCC map", ucas::heatMap(cv::abs(corrmap), false, false));
	
		double minV, maxV;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(corrmap, &minV, &maxV, &minLoc, &maxLoc);
		maxLoc.x += img_template.cols / 2;
		maxLoc.y += img_template.rows / 2;
		cv::Mat visual_check = plate_digits.clone();
		cv::circle(visual_check, maxLoc, 5, cv::Scalar(120), cv::FILLED);
		aia::imshow("Digits map", visual_check);
	}

	return EXIT_SUCCESS;
}