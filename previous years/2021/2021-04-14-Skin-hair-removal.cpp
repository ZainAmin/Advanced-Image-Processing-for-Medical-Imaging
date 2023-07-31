// inclusione delle librerie del corso e OpenCV necessarie
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

// create 'n' Structuring Elements (SEs) at different orientations spanning the whole 360°
cv::vector<cv::Mat>						// vector of 'width' x 'width' uint8 binary images with non-black pixels being the SE
	createTiltedStructuringElements(
	int width,							// SE width (must be odd)
	int height,							// SE height (must be odd)
	int n)								// number of SEs
	throw (ucas::Error);


/* 

GOAL:

Remove hairs from the dermatoscopic skin image (skin_hairs.jpg) without removing
the lesion.

*/

int main() 
{
	try
	{
		// load image and 50% downsampling to speedup processing
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/skin_hairs.jpg");
		cv::resize(img, img, cv::Size(0,0), 0.5, 0.5, cv::INTER_AREA);
		aia::imshow("Image", img);

		// switch to grayscale
		cv::Mat img_gray;
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

		// create 16 linear SEs so that they cannot fit the hairs
		// then perform sum of black hats
		cv::Mat sum_black_hats(img_gray.rows, img_gray.cols, CV_16U, cv::Scalar(0));
		for(auto & se : linearSEs)
		{
			cv::Mat blackhat;
			cv::morphologyEx(img_gray, blackhat, cv::MORPH_BLACKHAT, se);
			blackhat.convertTo(blackhat, CV_16U);
			sum_black_hats += blackhat;
		}

		// normalization
		cv::normalize(sum_black_hats, sum_black_hats, 0, 255, cv::NORM_MINMAX);
		sum_black_hats.convertTo(sum_black_hats, CV_8U);
		aia::imshow("Sum of black hats", sum_black_hats);

		// Otsu binarization
		cv::threshold(sum_black_hats, sum_black_hats, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		aia::imshow("Sum of black hats (binarized)", sum_black_hats);

		// prepare mask for next step (inpaint)
		cv::dilate(sum_black_hats, sum_black_hats, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));
		aia::imshow("Inpaint mask", sum_black_hats);

		// inpainting
		cv::inpaint(img, sum_black_hats, img, 7, cv::INPAINT_TELEA);
		aia::imshow("Final result", img);


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

