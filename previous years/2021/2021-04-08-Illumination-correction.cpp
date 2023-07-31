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
		aia::imshow("Original image", img, true, 2.0);
		cv::Mat img_copy = img.clone();

		cv::Mat SE = cv::getStructuringElement(
				cv::MORPH_RECT, cv::Size(41, 41));

		cv::Mat IF;
		cv::morphologyEx(img, IF, cv::MORPH_OPEN, SE);

		aia::imshow("Illumination field", IF, true, 2.0);
		cv::Scalar avgIF = cv::mean(IF);

		cv::Mat corrected_image = (img-IF) + avgIF;
		aia::imshow("Corrected image", corrected_image, true, 2.0);

		cv::threshold(corrected_image, corrected_image, 0, 255, 
			cv::THRESH_BINARY | cv::THRESH_OTSU);
		aia::imshow("Binarized image", corrected_image, true, 2.0);


		cv::threshold(img_copy, img_copy, 0, 255, 
			cv::THRESH_BINARY | cv::THRESH_OTSU);
		aia::imshow("Binarized original image", img_copy, true, 2.0);
		
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


