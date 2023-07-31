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

		for(int i=1; i<10; i++)
		{
			cv::Mat SE = cv::getStructuringElement(
				cv::MORPH_ELLIPSE, cv::Size(i*2+1, i*2+1));

			cv::morphologyEx(img, img, cv::MORPH_OPEN, SE);
			cv::morphologyEx(img, img, cv::MORPH_CLOSE, SE);

			aia::imshow(ucas::strprintf("ASD iteration %d", i), img, true, 2.0);

		}
		
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


