// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

bool areEqual(const cv::Mat & img1, const cv::Mat & img2) throw (ucas::Error)
{
	if(img1.channels() == 1 && img2.channels() == 1)
	{
		cv::Mat diff;
		cv::absdiff(img1, img2, diff);
		return !(cv::countNonZero(diff) != 0);
	}
	else if(img1.channels() == 3 && img2.channels() == 3)
	{
		cv::Mat chans1[3], chans2[3];
		cv::split(img1, chans1);
		cv::split(img2, chans2);
		return areEqual(chans1[0], chans2[0]) && 
			   areEqual(chans1[1], chans2[1]) && 
			   areEqual(chans1[2], chans2[2]);
	}
	else
		throw ucas::Error("Images have different channels, they cannot be compared");
}

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy.jpg");
		aia::imshow("Image", img);

		cv::Mat mask = img;
		cv::Mat marker;
		cv::morphologyEx(img, marker, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30,30)));
		aia::imshow("Marker", marker);

		cv::Mat marker_prev;
		int iters = 0;
		do 
		{
			printf("iteration # %d\n", ++iters);

			marker_prev = marker.clone();
			
			// grayscale morphological dilation on marker image
			cv::morphologyEx(marker, marker, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

			// point-wise minimum with mask
			marker = cv::min(marker, mask);

			//cv::imshow("Reconstruction in progress", marker, true);
			//cv::waitKey(50);
			
		} while ( !areEqual(marker_prev, marker) );

		// marker stores the reconstruction result
		aia::imshow("Reconstruction result", marker);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy-nostars.png", marker);
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

