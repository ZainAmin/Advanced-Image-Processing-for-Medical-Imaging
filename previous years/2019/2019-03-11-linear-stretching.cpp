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
		std::string imgPath = std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png";

		cv::Mat img = cv::imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);

		aia::imshow("histogram", ucas::imhist(img));

		cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);


		aia::imshow("histogram", ucas::imhist(img));


		aia::imshow("img", img);

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

