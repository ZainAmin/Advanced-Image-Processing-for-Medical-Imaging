// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: remove background noise
int main() 
{
	try
	{
		// load image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");
		aia::imshow("Image", img);

		// apply opening and closing with increasingly larger SEs
		// NOTE: in this example (rice.png), one iteration is enough. Try with another image.
		int k = 3;
		while(1)
		{
			cv::morphologyEx(img, img, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k,k)));
			cv::morphologyEx(img, img, CV_MOP_CLOSE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(k,k)));
			printf("k = %d\n", k);
			aia::imshow("Morphological Smoothing", img);
			k += 2;
		}

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

