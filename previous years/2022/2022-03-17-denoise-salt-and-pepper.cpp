// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
	cv::Mat img;
	cv::Mat img_corrupted;
	std::string win_name = "Salt-and-pepper demo";
	int salt_pepper_perc = 10;
	int median_filter_size = 1;

	void saltPepperDenoiseCallback(int pos, void* userdata)
	{
		for (int y = 0; y < img.rows; y++)
		{
			unsigned char* yRow		= img.ptr<unsigned char>(y);
			unsigned char* yRowCorr = img_corrupted.ptr<unsigned char>(y);

			for (int x = 0; x < img.cols; x++)
				if (rand() % 100 + 1 <= salt_pepper_perc)
					yRowCorr[x] = (rand() % 2) * 255; // <-- salt OR pepper noise with 50% prob
				else
					yRowCorr[x] = yRow[x];
		}

		cv::imshow(win_name, img_corrupted);

		if (median_filter_size % 2)
		{
			cv::medianBlur(img_corrupted, img_corrupted, median_filter_size);
			cv::imshow("Restored", img_corrupted);
		}
	}
};



int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", cv::IMREAD_GRAYSCALE);

	cv::namedWindow(win_name);
	cv::createTrackbar("noise level", win_name, &salt_pepper_perc, 100, saltPepperDenoiseCallback);
	cv::createTrackbar("median size", win_name, &median_filter_size, 50, saltPepperDenoiseCallback);
	img_corrupted = img.clone();
	saltPepperDenoiseCallback(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}