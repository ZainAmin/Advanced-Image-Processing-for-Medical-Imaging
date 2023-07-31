// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img, true, 2.0);

		cv::Mat IF;
		cv::morphologyEx(img, IF, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40)));
		aia::imshow("IF", IF, true, 2.0);

		aia::imshow("Top-hat", img-IF, true, 2.0);

		cv::Mat img_corrected = (img-IF) + cv::mean(IF)[0];
		aia::imshow("Image corrected", img_corrected, true, 2.0);

		cv::threshold(img_corrected, img_corrected, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		aia::imshow("Image binarization", img_corrected, true, 2.0);


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

