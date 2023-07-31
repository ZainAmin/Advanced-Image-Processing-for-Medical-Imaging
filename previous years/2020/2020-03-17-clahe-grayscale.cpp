// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat claheq(const cv::Mat & img_mono, int n_tiles = 8)
{
	cv::Ptr<cv::CLAHE> algo = cv::createCLAHE(40, cv::Size(n_tiles, n_tiles));
	cv::Mat img_eq;
	algo->apply(img_mono, img_eq);
	return img_eq;
}

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img);

		aia::imshow("Image post CLAHE", claheq(img, 4));

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

