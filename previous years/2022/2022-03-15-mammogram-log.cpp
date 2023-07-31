// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram.tif", cv::IMREAD_UNCHANGED);
	aia::imshow("Mammogram", img, true, 0.2);

	int bpp = ucas::imdepth_detect(img);
	int L = std::pow(2, bpp);
	printf("bpp = %d\n, L = %d\n", bpp, L);

	double c = (L - 1) / std::log(L);
	for (int y = 0; y < img.rows; y++)
	{
		unsigned short* yRow = img.ptr<unsigned short>(y);
		for (int x = 0; x < img.cols; x++)
			yRow[x] = c * std::log(1 + yRow[x]);
	}
	img = (L-1) - img;

	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram_processed.tif", img);



	return EXIT_SUCCESS;
}