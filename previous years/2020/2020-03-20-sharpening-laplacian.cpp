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
		float k = k_x10/10.0;
		cv::Mat sharp_kernel = 
			(cv::Mat_<float>(3,3) <<
			   -k,    -k,    -k,
			   -k,    1+8*k, -k,
			   -k,    -k,    -k
			);

		cv::filter2D(img, img_sharpened, CV_8U, sharp_kernel);

		// the final result will depend on the new min and max
		// that might be under 0 and above 255 due to the sharpening
		//cv::filter2D(img, img_sharpened, CV_32F, sharp_kernel);
		//cv::normalize(img_sharpened, img_sharpened, 0, 255, cv::NORM_MINMAX);
		//img_sharpened.convertTo(img_sharpened, CV_8U);
		
		cv::imshow(win_name, img_sharpened);
	}
}

int main() 
{
	try
	{
		//img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/moon.png", CV_LOAD_IMAGE_GRAYSCALE);
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/eye.blurry.png");
		
		win_name = "Sharpening";
		cv::namedWindow(win_name);
		cv::createTrackbar("factor", win_name, &k_x10, 100, sharpenCallback);

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

