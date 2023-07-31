// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat histEq(const cv::Mat & img_mono)
{
	std::vector<int> hist = ucas::histogram(img_mono);

	int L = std::pow(2, ucas::imdepth_detect(img_mono));
	std::vector<int> transform(L);
	for(int k=0; k<L; k++)
	{
		transform[k] = 0;
		for(int j=0; j<=k; j++)
			transform[k] += hist[j];
		transform[k] = ucas::round(transform[k] * ((float(L)-1)/(img_mono.rows*img_mono.cols)));
	}

	cv::Mat img_mono_eq = img_mono.clone();
	for(int y=0; y<img_mono_eq.rows; y++)
	{
		unsigned char* row = img_mono_eq.ptr<unsigned char>(y);
		for(int x=0; x<img_mono_eq.cols; x++)
			row[x] = transform[row[x]];
	}

	return img_mono_eq;
}

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina_lowcontrast.bmp");
		aia::imshow("Image", img);
		
		// HE in RGB space --> wrong!
		/*std::vector <cv::Mat> channels;
		cv::split(img, channels);

		std::vector <cv::Mat> channels_eq(3);
		channels_eq[0] = histEq(channels[0]);
		channels_eq[1] = histEq(channels[1]);
		channels_eq[2] = histEq(channels[2]);

		cv::Mat img_eq;
		cv::merge(channels_eq, img_eq);
		aia::imshow("Image post HE", img_eq);*/

		// HE in Lab space --> correct!
		cv::Mat img_Lab;
		cv::cvtColor(img, img_Lab, cv::COLOR_BGR2Lab);
		std::vector <cv::Mat> channels;
		cv::split(img_Lab, channels);
		channels[0] = histEq(channels[0]);
		cv::Mat img_eq;
		cv::merge(channels, img_eq);
		cv::cvtColor(img_eq, img_eq, cv::COLOR_Lab2BGR);
		aia::imshow("Image post HE (with Lab)", img_eq);

		// HE in HSV space --> correct!
		cv::Mat img_hsv;
		cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
		cv::split(img_hsv, channels);
		channels[2] = histEq(channels[2]);
		cv::merge(channels, img_eq);
		cv::cvtColor(img_eq, img_eq, cv::COLOR_HSV2BGR);
		aia::imshow("Image post HE (with HSV)", img_eq);

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

