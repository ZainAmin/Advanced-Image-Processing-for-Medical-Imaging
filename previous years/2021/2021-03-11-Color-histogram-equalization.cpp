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
		cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/low_contrast_photo.png");
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		//cv::resize(input_img, input_img, cv::Size(0,0), 0.2, 0.2, CV_INTER_AREA);

		// not good solution: equalize all channels in BGR color space
		std::vector< cv::Mat > channels;
		cv::split(input_img, channels);

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4, cv::Size(8,8));
		std::vector< cv::Mat> channels_HE, channels_CLAHE;
		for(auto & channel : channels)
		{
			channels_HE.push_back(histEq8bit(channel));

			cv::Mat tmp;
			clahe->apply(channel, tmp);
			channels_CLAHE.push_back(tmp);
		}

		cv::Mat img_eq_HE, img_eq_CLAHE;
		cv::merge(channels_HE, img_eq_HE);
		cv::merge(channels_CLAHE, img_eq_CLAHE);

		aia::imshow("Before equalization", input_img);
		aia::imshow("Histogram (after HE)", img_eq_HE);
		aia::imshow("Histogram (after CLAHE)", img_eq_CLAHE);

		// better solution: equalize L channel in Lab color space
		cv::Mat img_Lab;
		cv::cvtColor(input_img, img_Lab, cv::COLOR_BGR2Lab);

		cv::split(img_Lab, channels);
		channels[0] = histEq8bit(channels[0]);
		cv::merge(channels, img_eq_HE);
		cv::cvtColor(img_eq_HE, img_eq_HE, cv::COLOR_Lab2BGR);

		cv::split(img_Lab, channels);
		clahe->apply(channels[0], channels[0]);
		cv::merge(channels, img_eq_CLAHE);
		cv::cvtColor(img_eq_CLAHE, img_eq_CLAHE, cv::COLOR_Lab2BGR);

		aia::imshow("Histogram (after HE, Lab)", img_eq_HE);
		aia::imshow("Histogram (after CLAHE, Lab)", img_eq_CLAHE);

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
