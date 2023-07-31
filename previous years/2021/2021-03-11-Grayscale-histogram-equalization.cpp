// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

cv::Mat histEq8bit(const cv::Mat & img)
{
	std::vector <int> cdf;
	std::vector <int> hist = ucas::histogram(img);
	int acc = 0;
	for(int j=0; j<256; j++)
	{
		acc += hist[j];
		cdf.push_back(acc);
	}

	cv::Mat img_eq = img.clone();
	float den = img.rows * img.cols;
	for(int y=0; y<img_eq.rows; y++)
	{
		unsigned char* yRow = img_eq.ptr<unsigned char>(y);
		for(int x=0; x<img_eq.cols; x++)
			yRow[x] = ucas::round<float>( (255.0f*cdf[yRow[x]]) / den );
	}

	return img_eq;
}

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		// load the image
		cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		aia::imshow("Before equalization", input_img);
		aia::imshow("Histogram (before)", ucas::imhist(input_img));

		cv::Mat img_eq = histEq8bit(input_img);
		aia::imshow("After HE", img_eq);
		aia::imshow("Histogram (after HE)", ucas::imhist(img_eq));

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4, cv::Size(8,8));

		clahe->apply(input_img, img_eq);
		aia::imshow("After CLAHE", img_eq);

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
