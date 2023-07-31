// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// create 'n' rectangular Structuring Elements (SEs) at different orientations spanning the whole 360°
cv::vector<cv::Mat>						// vector of 'width' x 'width' uint8 binary images with non-black pixels being the SE
	createTiltedStructuringElements(
	int width,							// SE width (must be odd)
	int height,							// SE height (must be odd)
	int n)								// number of SEs
	throw (ucas::Error)
{
	// check preconditions
	if( width%2 == 0 )
		throw ucas::Error(ucas::strprintf("Structuring element width (%d) is not odd", width));
	if( height%2 == 0 )
		throw ucas::Error(ucas::strprintf("Structuring element height (%d) is not odd", height));

	// draw base SE along x-axis
	cv::Mat base(width, width, CV_8U, cv::Scalar(0));
	// workaround: cv::line does not work properly when thickness > 1. So we draw line by line.
	for(int k=width/2-height/2; k<=width/2+height/2; k++)
		cv::line(base, cv::Point(0,k), cv::Point(width, k), cv::Scalar(255));

	// compute rotated SEs
	cv::vector <cv::Mat> SEs;
	SEs.push_back(base);
	double angle_step = 180.0/n;
	for(int k=1; k<n; k++)
	{
		cv::Mat SE;
		cv::warpAffine(base, SE, cv::getRotationMatrix2D(cv::Point2f(base.cols/2.0f, base.rows/2.0f), k*angle_step, 1.0), cv::Size(width, width), CV_INTER_NN);
		SEs.push_back(SE);
	}

	return SEs;	 
}

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/angiogram.png", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img, true, 2.0);

		/*std::vector <cv::Mat> SEs = createTiltedStructuringElements(51, 1, 8);
		for(auto & se : SEs)
			aia::imshow("SE", se, true, 10);*/

		std::vector <cv::Mat> linearSEs = createTiltedStructuringElements(7, 1, 16);
		cv::Mat sumtophats(img.rows, img.cols, CV_16U, cv::Scalar(0));
		
		for(auto & se : linearSEs)
		{
			cv::Mat tophat;
			cv::morphologyEx(img, tophat, cv::MORPH_TOPHAT, se);
			aia::imshow("Directional Tophats", tophat, true, 2.0);
			tophat.convertTo(tophat, CV_16U);
			sumtophats += tophat;
		}
		cv::normalize(sumtophats, sumtophats, 0, 255, cv::NORM_MINMAX);
		sumtophats.convertTo(sumtophats, CV_8U);
		aia::imshow("Result", sumtophats, true, 2.0);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/angiogram-MAs.png", sumtophats);
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

