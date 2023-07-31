// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

cv::Mat frameProcessor(const cv::Mat & img)
{
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	cv::Mat dx, dy;
	cv::Sobel(img_gray, dx, CV_32F, 1, 0);
	cv::Sobel(img_gray, dy, CV_32F, 0, 1);

	cv::Mat mag;
	cv::magnitude(dx, dy, mag);

	cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
	mag.convertTo(mag, CV_8U);

	return 3 * mag;

	// optional: implement (basic) cartoonization processing

	/*cv::Mat median_filtered;
	cv::medianBlur(img, median_filtered, 7);

	cv::Mat mask;
	cv::threshold(mag, mask, 40, 255, cv::THRESH_BINARY);

	median_filtered.setTo(cv::Scalar(0, 0, 0), mask);

	return median_filtered;*/
}

int main()
{
		// EXAMPLE 5: video processing: Face Detection
		// - requires a (working) camera
		// - requires a Face Detector (pre-trained classifier): automatically loaded within faceRectangles()
		/* comment this if you do not want to run it / it does not work */
		aia::processVideoStream("", frameProcessor);
		//                      /\                       /\
		//                      || empty video input = use camera
		//                                               || function / processing to be applied to each frame of the video sequence

	return EXIT_SUCCESS;
}

