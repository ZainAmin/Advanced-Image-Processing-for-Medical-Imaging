// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() 
{
	try
	{
		//cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.bmp", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img, true, 1.0f);

		//cv::Mat img_bin = ucas::binarize(img.clone(), ucas::getOtsuAutoThreshold(ucas::histogram(img)));
		cv::Mat img_bin = ucas::binarize(img.clone(), ucas::getTriangleAutoThreshold(ucas::histogram(img)));

		std::vector <std::vector <cv::Point> > contours;
		cv::findContours(img_bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		int n_objects = 0;
		for(int k=0; k<contours.size(); k++)
			if(cv::contourArea(contours[k]) > 50)
			{
				n_objects++;
				cv::drawContours(img, contours, k, cv::Scalar(255), 2, CV_AA);
			}

		printf("Number of objects = %d\n", n_objects);

		aia::imshow("Image contours", img, true, 1.0f);


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

