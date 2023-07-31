// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	cv::Mat input_img;
	std::string win_name;
	int percSaltPepper = 10;
	int filter_size = 3;

	void denoiseSaltPepper(int pos, void* userdata)
	{
		cv::Mat img_noised = input_img.clone();

		for(int y=0; y<img_noised.rows; y++)
		{
			unsigned char* yRow = img_noised.ptr<unsigned char>(y);
			for(int x=0; x<img_noised.cols; x++)
				if(rand()%100 < percSaltPepper)
					yRow[x] = (rand()%2)*255;
		}

		cv::imshow(win_name, img_noised);

		if(filter_size % 2 == 0)
			return;

		cv::Mat img_denoised_gaussian;
		cv::GaussianBlur(img_noised, img_denoised_gaussian, cv::Size(filter_size, filter_size), 0);

		cv::Mat img_denoised_median;
		cv::medianBlur(img_noised, img_denoised_median, filter_size);

		cv::imshow("Gaussian-denoised", img_denoised_gaussian);
		cv::imshow("Median-denoised", img_denoised_median);
	}
}

// GOAL: reduce salt-and-pepper noise
int main() 
{
	try
	{
		// load the image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		// create the UI
		win_name = "Denoising salt pepper";
		cv::namedWindow(win_name);
		cv::createTrackbar("noise", win_name, &percSaltPepper, 100, denoiseSaltPepper);
		cv::createTrackbar("filter", win_name, &filter_size, 100, denoiseSaltPepper);

		// start the UI
		denoiseSaltPepper(0, 0);

		// waits for user to press a button and exit from the app
		cv::waitKey(0);

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
