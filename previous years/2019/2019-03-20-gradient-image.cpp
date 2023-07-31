// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// to use fast non local means
#include <opencv2/photo/photo.hpp>

cv::Mat gradientImage(const cv::Mat& frame)
{
	// convert color image to gray image
	cv::Mat img_gray;
	cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);

	// calculate first derivative with Sobel
	cv::Mat dx, dy;
	cv::Sobel(img_gray, dx, CV_32F, 1, 0);
	cv::Sobel(img_gray, dy, CV_32F, 0, 1);

	// calculate gradient magnitude
	cv::Mat mag;
	cv::magnitude(dx, dy, mag);

	// make gradient magnitude visualizable
	cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
	mag.convertTo(mag, CV_8U);

	return 3*mag;
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

