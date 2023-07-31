// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/text.png", CV_LOAD_IMAGE_GRAYSCALE);
		
		cv::threshold(img, img, 254, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
		aia::imshow("Binarized image", img);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/text_binarized.png", img);

		cv::Mat img_eroded;
		cv::erode(img, img_eroded, 
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 24)));
		aia::imshow("After erosion", img_eroded);

		cv::Mat marker = img_eroded;
		cv::Mat mask = img;
		cv::Mat marker_prev;
		do 
		{
			marker_prev = marker.clone();

			// geodesic dilation = dilation + intersection with mask
			// CROSS SE for 4-adjacency-based reconstruction
			cv::dilate(marker, marker, 
				cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
			marker = marker & mask;

			cv::waitKey(200);
			cv::imshow("Reconstruction in progress", marker);

		} while ( cv::countNonZero(marker - marker_prev));

		aia::imshow("After reconstruction", marker);

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