// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int main()
{
	cv::Mat mammo = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram.tif",
		cv::IMREAD_UNCHANGED);

	printf("bitdepth = %d\n", aia::bitdepth(mammo.depth()));

	int L = std::pow(2, 14);
	float c = (L - 1) / std::log10(L);

	for (int i = 0; i < mammo.rows; i++)
	{
		unsigned short* iThRow = mammo.ptr<unsigned short>(i);
		for (int j = 0; j < mammo.cols; j++)
			iThRow[j] = c * std::log10(1 + iThRow[j]);
	}

	mammo = L - 1 - mammo;

	aia::imshow("Mammogram", mammo, true, 0.5);
	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram_processed.tif", mammo);

	return EXIT_SUCCESS;
}