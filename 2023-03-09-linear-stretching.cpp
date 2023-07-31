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
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png",
		cv::IMREAD_GRAYSCALE);

	aia::imshow("Original image", img);
	aia::imshow("Original histogram", ucas::imhist(img));

	double minV, maxV;
	cv::minMaxLoc(img, &minV, &maxV);
	
	// this does not work because of integer calculations
	// it would work if the image is converted to CV_32F (float)
	/*img - minV;
	img /= (maxV - minV);
	img *= 255;*/

	for (int i = 0; i < img.rows; i++)
	{
		unsigned char* iThRow = img.ptr(i);
		for (int j = 0; j < img.cols; j++)
			iThRow[j] = ((iThRow[j]-minV)/float(maxV-minV))*255;
	}

	// this is the buil-in OpenCV function that implements the opearation above
	//cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);

	aia::imshow("Linear stretched image", img);
	aia::imshow("Stretched histogram", ucas::imhist(img));

	return EXIT_SUCCESS;
}