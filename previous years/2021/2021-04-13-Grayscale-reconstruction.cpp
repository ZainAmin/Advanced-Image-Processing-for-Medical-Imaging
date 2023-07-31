// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy.jpg");

	
		cv::Mat img_opened;
		cv::erode(img, img_opened, 
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)));
		aia::imshow("After opening", img_opened);

		aia::imshow("Difference image", img-img_opened);

		cv::Mat marker = img_opened;
		cv::Mat mask = img;
		cv::Mat marker_prev;

		std::vector<cv::Mat> marker_prev_chans;
		std::vector<cv::Mat> marker_chans;

		do 
		{
			marker_prev = marker.clone();

			cv::split(marker_prev, marker_prev_chans);

			// geodesic dilation = dilation + minimum with mask
			cv::dilate(marker, marker, 
				cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
			marker = cv::min(marker, mask);

			cv::split(marker, marker_chans);

			//cv::waitKey(10);
			cv::imshow("Reconstruction in progress", marker);

		} while ( cv::countNonZero(marker_chans[0] - marker_prev_chans[0]) ||
			      cv::countNonZero(marker_chans[1] - marker_prev_chans[1]) ||
				  cv::countNonZero(marker_chans[2] - marker_prev_chans[2]));

		aia::imshow("After reconstruction", marker);

		aia::imshow("Difference image", mask-marker);

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