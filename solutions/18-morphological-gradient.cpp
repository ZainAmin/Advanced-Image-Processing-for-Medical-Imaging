// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: enhance edges
int main() 
{
	try
	{
		// load image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/brain_ct.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");
		aia::imshow("Image", img);

		// apply morphological gradient with a small SE
		cv::morphologyEx(img, img, CV_MOP_GRADIENT, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
		aia::imshow("Morphological gradient", img);
		aia::imshow("Morphological gradient enhanced (x2)", img*2);

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

