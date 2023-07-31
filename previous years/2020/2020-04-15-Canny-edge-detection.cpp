// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace
{
	cv::Mat img;
	int sigmaX10 = 10;
	int T_L = 20;
	std::string win_name;

	void CannyEdgeDetection(int pos, void* userdata)
	{
		cv::Mat img_blurred;
		float sigma = sigmaX10/10.0f;
		int filter_size = 6*sigma;
		if(filter_size%2 == 0)
			filter_size += 1;

		cv::GaussianBlur(img, img_blurred, cv::Size(filter_size,filter_size), sigma, sigma );
		cv::Canny(img_blurred, img_blurred, T_L, T_L*3, 3, true);
		
		cv::imshow(win_name, img_blurred);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/child.png");
		aia::imshow("Image", img);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

		win_name = "Canny edge detection";
		cv::namedWindow(win_name);
		cv::createTrackbar("sigmaX10", win_name, &sigmaX10, 100, CannyEdgeDetection);
		cv::createTrackbar("T low", win_name, &T_L, 100, CannyEdgeDetection);

		CannyEdgeDetection(0, 0);
		cv::waitKey(0);

		return 1;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

