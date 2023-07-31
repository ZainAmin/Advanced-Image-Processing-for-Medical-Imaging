// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

cv::Mat gradientImage(const cv::Mat & frame) throw (aia::error)
{
	cv::Mat frame_grey;
	cv::cvtColor(frame, frame_grey, cv::COLOR_BGR2GRAY);
	cv::Mat dx, dy;
	cv::Sobel(frame_grey, dx, CV_32F, 1, 0);
	cv::Sobel(frame_grey, dy, CV_32F, 0, 1);

	cv::Mat mag;
	cv::magnitude(dx, dy, mag);

	cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
	mag.convertTo(mag, CV_8U);

	return 3*mag;
}

// GOAL: reduce salt-and-pepper noise
int main() 
{
	try
	{
		// load the image
		/*cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		cv::Mat dx, dy;
		cv::Sobel(input_img, dx, CV_32F, 1, 0);
		cv::Sobel(input_img, dy, CV_32F, 0, 1);

		cv::Mat mag;
		cv::magnitude(dx, dy, mag);

		cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
		mag.convertTo(mag, CV_8U);

		aia::imshow("Gradient image", mag);
		aia::imshow("Gradient image x2", 2*mag);
		aia::imshow("Gradient image x3", 3*mag);
		aia::imshow("Gradient image x4", 4*mag);*/

		aia::processVideoStream("", gradientImage);

		return EXIT_SUCCESS;
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

