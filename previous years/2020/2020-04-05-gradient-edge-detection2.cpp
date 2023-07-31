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
	int sigma_x10 = 30;
	int threshold = 20;
	std::string win_name;

	void gradientEdgeDetection(int pos, void* userdata)
	{
		// denoising step
		cv::Mat img_denoised = img.clone();
		if(sigma_x10 != 0)
			cv::GaussianBlur(img, img_denoised, cv::Size(0,0), sigma_x10/10.0, sigma_x10/10.0);

		cv::Mat dx, dy;
		cv::Sobel(img_denoised, dx, CV_32F, 1, 0);
		cv::Sobel(img_denoised, dy, CV_32F, 0, 1);

		cv::Mat mag;
		cv::magnitude(dx, dy, mag);

		cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
		mag.convertTo(mag, CV_8U);

		double minV, maxV;
		cv::minMaxLoc(mag, &minV, &maxV);
		cv::threshold(mag, mag, (threshold/100.0)*maxV, 255, CV_THRESH_BINARY);

		cv::imshow(win_name, mag);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/brain_ct.jpeg");
		aia::imshow("Image", img);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

		win_name = "Gradient-based edge detection";
		cv::namedWindow(win_name);
		cv::createTrackbar("sigma", win_name, &sigma_x10, 100, gradientEdgeDetection);
		cv::createTrackbar("threshold", win_name, &threshold, 100, gradientEdgeDetection);

		gradientEdgeDetection(0, 0);
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

