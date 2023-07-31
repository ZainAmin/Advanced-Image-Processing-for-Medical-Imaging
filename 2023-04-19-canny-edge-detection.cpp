// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int sigma_x10 = 10;
int T_low = 0;
std::string winname = "Canny's method";
cv::Mat img;

void CannyEdgeDetection(int pos, void* userdata)
{
	cv::Mat img_smoothed;
	cv::GaussianBlur(img, img_smoothed, cv::Size(0, 0), sigma_x10 / 10.0f, sigma_x10 / 10.0f);

	cv::Mat img_edges;
	cv::Canny(img_smoothed, img_edges, T_low, 3 * T_low);

	cv::imshow(winname, img_edges);
}

int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/girl.png",
		cv::IMREAD_GRAYSCALE);
	//cv::resize(img, img, cv::Size(0, 0), 0.5, 0.5);

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
	cv::createTrackbar("sigma_x10", winname, &sigma_x10, 100, CannyEdgeDetection);
	cv::createTrackbar("T_low", winname, &T_low, 200, CannyEdgeDetection);

	CannyEdgeDetection(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}