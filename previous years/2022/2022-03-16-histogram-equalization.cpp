// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

// img should be 1-channel 8-bit
cv::Mat histEqualize(cv::Mat img)
{
	if (img.channels() != 1)
	{
		std::cerr << "only single channel images are supported\n";
		return img;
	}

	// histogram computation
	std::vector<float> hist(256);
	for (int y = 0; y < img.rows; y++)
	{
		unsigned char* yRow = img.ptr<unsigned char>(y);
		for (int x = 0; x < img.cols; x++)
			hist[yRow[x]]++;
	}

	// histogram normalization (= pdf)
	int N = img.rows * img.cols;
	for (int i = 0; i < hist.size(); i++)
		hist[i] /= N;

	// compute CDF
	std::vector<float> cdf(256);
	float acc = 0;
	for (int i = 0; i < hist.size(); i++)
	{
		acc += hist[i];
		cdf[i] = acc;
	}

	// transform CDF into the histogram equalization function
	for (int i = 0; i < cdf.size(); i++)
		cdf[i] *= 255;

	// apply histogram equalization function pointwise to input pixel values
	for (int y = 0; y < img.rows; y++)
	{
		unsigned char* yRow = img.ptr<unsigned char>(y);
		for (int x = 0; x < img.cols; x++)
			yRow[x] = cdf[yRow[x]];
	}

	return img;
}

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", cv::IMREAD_GRAYSCALE);

	aia::imshow("Original image", img);
	aia::imshow("Original histogram", ucas::imhist(img));

	cv::Mat dst;
	cv::normalize(img, dst, 0, 255, cv::NORM_MINMAX);
	aia::imshow("Stretched image", dst);
	aia::imshow("Stretched image (histogram)", ucas::imhist(dst));

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10);
	clahe->apply(img, dst);
	aia::imshow("CLAHE image", dst);
	aia::imshow("CLAHE image (histogram)", ucas::imhist(dst));

	img = histEqualize(img);
	aia::imshow("Histogram equalized image", img);
	aia::imshow("Histogram equalized image (histogram)", ucas::imhist(img));




	return EXIT_SUCCESS;
}