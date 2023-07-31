// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat gradientImage(const cv::Mat & img)
{
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	// denoising step
	cv::GaussianBlur(img_gray, img_gray, cv::Size(3,3), 0, 0);

	cv::Mat dx, dy;
	cv::Sobel(img_gray, dx, CV_32F, 1, 0);
	cv::Sobel(img_gray, dy, CV_32F, 0, 1);

	cv::Mat mag;
	cv::magnitude(dx, dy, mag);

	cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
	mag.convertTo(mag, CV_8U);

	double minV, maxV;
	cv::minMaxLoc(mag, &minV, &maxV);
	cv::threshold(mag, mag, 0.20*maxV, 255, CV_THRESH_BINARY);

	return mag;
}

int main() 
{
	try
	{
		aia::processVideoStream("", gradientImage);

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

