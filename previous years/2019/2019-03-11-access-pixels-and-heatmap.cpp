// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "ucasImageUtils.h"

// include my project functions
#include "functions.h"

cv::Mat toHeatMap(cv::Mat img)
{
	double minV, maxV;
	cv::minMaxLoc(img, &minV, &maxV);

	cv::Mat _hsv[3], hsv;
	_hsv[0] = cv::Mat(img.rows, img.cols, CV_8U);
	_hsv[1] = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(255));
	_hsv[2] = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(255));

	_hsv[0] = (maxV-img)*(120/float(maxV));


	cv::merge(_hsv, 3, hsv);
	cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);

	return hsv;
}

int main() 
{
	try
	{
		/*std::string imgPath = std::string(EXAMPLE_IMAGES_PATH) + "/lena.png";

		cv::Mat img = cv::imread(imgPath);
		if(!img.data)
			throw ucas::Error("Cannot read image");

		printf("rows = %d, cols = %d, channels = %d, depth = %d\n", 
			img.rows, img.cols, img.channels(), aia::bitdepth(img.depth()));

		img.at<cv::Vec3b>(250,250)[0] = 255;
		img.at<cv::Vec3b>(250,250)[1] = 255;
		img.at<cv::Vec3b>(250,250)[2] = 255;

		aia::imshow("image", img, true, 2.0);*/

		/*ucas::Timer timer;
		timer.start();
		cv::Mat grayshades(5000, 5000, CV_8U);
		for(int y=0; y<grayshades.rows; y++)
			for(int x=0; x<grayshades.cols; x++)
				grayshades.at<unsigned char>(y, x) = 150;
		printf("method 1: elapsed time = %.3f\n", timer.elapsed<float>());

		timer.restart();
		for(int y=0; y<grayshades.rows; y++)
		{
			unsigned char* ythRow = grayshades.ptr<unsigned char>(y);
			for(int x=0; x<grayshades.cols; x++)
				ythRow[x] = 150;
		}
		printf("method 2: elapsed time = %.3f\n", timer.elapsed<float>());

		aia::imshow("grayshades", grayshades);*/

		cv::Mat grayshades(500, 500, CV_8U);
		for(int y=0; y<grayshades.rows; y++)
		{
			unsigned char* ythRow = grayshades.ptr<unsigned char>(y);
			for(int x=0; x<grayshades.cols; x++)
				ythRow[x] = (y/float(grayshades.rows-1))*255;
		}
		//aia::imshow("grayshades", grayshades);

		aia::imshow("heatmap", toHeatMap(grayshades));

		aia::imshow("lightning", toHeatMap(cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", CV_LOAD_IMAGE_GRAYSCALE)));
                       
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

