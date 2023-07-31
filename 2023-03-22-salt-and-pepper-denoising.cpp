// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int k = 1;
int salt_pepper_perc = 0;
std::string winname = "Denoising salt-and-pepper";
cv::Mat img;

void denoiseSaltPepper(int pos, void* userdata)
{
	cv::Mat img_corrupted = img.clone();

	for (int y = 0; y < img_corrupted.rows; y++)
	{
		unsigned char* yRow = img_corrupted.ptr(y);
		for (int x = 0; x < img_corrupted.cols; x++)
		{
			// corrupt pixel with salt_pepper_perc probability
			if((rand()%100+1) <= salt_pepper_perc)
				yRow[x] = rand()%2 ? 255 : 0;
		}
	}


	cv::imshow(winname, img_corrupted);

	if (k % 2 == 1)
	{
		cv::Mat img_denoised;
		cv::medianBlur(img_corrupted, img_denoised, k);
		cv::imshow("Denoised", img_denoised);
	}
}

int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/girl.png",
		cv::IMREAD_GRAYSCALE);

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("saltpepper perc", winname, &salt_pepper_perc, 100, denoiseSaltPepper);
	cv::createTrackbar("median size", winname, &k, 50, denoiseSaltPepper);

	denoiseSaltPepper(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}