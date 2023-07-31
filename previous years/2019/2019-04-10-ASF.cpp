// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"


int main() 
{
	try
	{
		// load an image where there are lines that can be detected
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");

		aia::imshow("original image", img, true, 2.0);

		int k = 3;
		while(1)
		{
			cv::morphologyEx(
				img, 
				img, 
				cv::MORPH_OPEN, 
				cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
			
			cv::morphologyEx(img, 
				img, 
				cv::MORPH_CLOSE, 
				cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
			
			printf("k = %d\n", k);
			aia::imshow("ASF", img, true, 2.0);
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