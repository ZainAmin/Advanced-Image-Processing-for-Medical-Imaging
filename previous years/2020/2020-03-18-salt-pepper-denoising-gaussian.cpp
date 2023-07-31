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
	cv::Mat img_noised;
	cv::Mat img_denoised;
	int salt_pepper_perc = 10;
	int gaussian_filter_size = 3;
	std::string win_name;

	void saltPepperDenoising(int pos, void* userdata)
	{
		// add salt and pepper noise
		img.copyTo(img_noised);
		for(int y=0; y<img.rows; y++)
		{
			unsigned char* row = img_noised.ptr<unsigned char>(y);

			for(int x=0; x<img.cols; x++)
				if(rand()%100+1 <= salt_pepper_perc)
					row[x] = (rand()%2)*255;
		}

		if(gaussian_filter_size % 2)
			cv::GaussianBlur(img_noised, img_denoised, cv::Size(gaussian_filter_size, gaussian_filter_size), 0, 0);

		cv::imshow(win_name, img);
		cv::imshow("With noise", img_noised);
		cv::imshow("Denoised", img_denoised);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
		
		img_noised = img.clone();
		img_denoised = img.clone();

		win_name = "Salt-and-pepper denoising";
		cv::namedWindow(win_name);
		cv::createTrackbar("salt-pepper-perc", win_name, &salt_pepper_perc, 100, saltPepperDenoising);
		cv::createTrackbar("gaussian-filter", win_name, &gaussian_filter_size, 100, saltPepperDenoising);

		saltPepperDenoising(0, 0);
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

