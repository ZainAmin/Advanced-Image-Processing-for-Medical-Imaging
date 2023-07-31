// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace
{
	cv::Mat img;
	cv::Mat img_sharpened;
	int k_x10 = 0;
	std::string win_name;

	void sharpenCallback(int pos, void* userdata)
	{
		cv::Mat img_blurred;
		cv::GaussianBlur(img, img_blurred, cv::Size(7,7), 0, 0);

		img_sharpened = img + (k_x10/10.0)*(img-img_blurred);
		
		cv::imshow(win_name, img_sharpened);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/moon.png", CV_LOAD_IMAGE_GRAYSCALE);
		//img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/eye.blurry.png");
		
		win_name = "Sharpening";
		cv::namedWindow(win_name);
		cv::createTrackbar("factor", win_name, &k_x10, 200, sharpenCallback);

		sharpenCallback(0, 0);
		cv::waitKey(0);

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

