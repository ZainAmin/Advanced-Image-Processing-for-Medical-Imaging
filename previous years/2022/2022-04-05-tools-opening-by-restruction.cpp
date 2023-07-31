// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	// load the image
	cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.png", cv::IMREAD_GRAYSCALE);

	aia::imshow("Original image", input_img);
	aia::imshow("Histogram", ucas::imhist(input_img));

	// TRIANGLE: better choice since dark background is dominant
	cv::Mat binarized_img_triangle;
	int T = ucas::getTriangleAutoThreshold(ucas::histogram(input_img));
	printf("Triangle T = %d\n", T);
	cv::threshold(input_img, binarized_img_triangle, T, 255, cv::THRESH_BINARY);
	aia::imshow("Triangle-binarized image", binarized_img_triangle);

	cv::Mat SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));

	cv::Mat eroded_img;
	cv::morphologyEx(binarized_img_triangle, eroded_img, cv::MORPH_ERODE, SE);
	aia::imshow("Triangle after erosion", eroded_img);
	
	cv::Mat marker_cur = eroded_img;
	cv::Mat marker_prev;
	cv::Mat mask = binarized_img_triangle;
	do
	{
		marker_prev = marker_cur.clone();

		cv::dilate(marker_cur, marker_cur, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
		marker_cur = marker_cur & mask;

		cv::imshow("Reconstruction in progress", marker_cur);
		cv::waitKey(100);

	} while (cv::countNonZero(marker_cur - marker_prev));

	aia::imshow("Reconstruction result", marker_cur);

	return EXIT_SUCCESS;
}