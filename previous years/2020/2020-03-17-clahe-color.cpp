// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat claheq(const cv::Mat & img_mono, int n_tiles = 8)
{
	cv::Ptr<cv::CLAHE> algo = cv::createCLAHE(40, cv::Size(n_tiles, n_tiles));
	cv::Mat img_eq;
	algo->apply(img_mono, img_eq);
	return img_eq;
}

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina_lowcontrast.bmp");
		aia::imshow("Image", img);

		// HE in Lab space --> correct!
		cv::Mat img_Lab;
		cv::cvtColor(img, img_Lab, cv::COLOR_BGR2Lab);
		std::vector <cv::Mat> channels;
		cv::split(img_Lab, channels);
		channels[0] = claheq(channels[0], 4);
		cv::Mat img_eq;
		cv::merge(channels, img_eq);
		cv::cvtColor(img_eq, img_eq, cv::COLOR_Lab2BGR);
		aia::imshow("Image post HE (with Lab)", img_eq);

		// HE in HSV space --> correct!
		cv::Mat img_hsv;
		cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
		cv::split(img_hsv, channels);
		channels[2] = claheq(channels[2], 4);
		cv::merge(channels, img_eq);
		cv::cvtColor(img_eq, img_eq, cv::COLOR_HSV2BGR);
		aia::imshow("Image post HE (with HSV)", img_eq);

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

