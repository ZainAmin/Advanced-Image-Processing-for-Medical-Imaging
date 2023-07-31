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
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/text.png", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img);

		img = 255 - img;
		cv::threshold(img, img, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/text_bin.png", img);
		aia::imshow("Image binarized", img);

		
		// detectiong of elongated characters
		cv::Mat marker;
		cv::erode(img, marker, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 25)));
		aia::imshow("Marker", marker);
		
		// reconstruction by dilation
		cv::Mat marker_prev = marker.clone();
		int iters = 0;
		do 
		{
			printf("reconstruction iter # %d\n", ++iters);
			marker.copyTo(marker_prev);
			cv::dilate(marker, marker, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3)));
			marker = marker & img;

			//cv::imshow("Reconstruction by dilation", marker);
			//cv::waitKey(200);

		} while (cv::countNonZero(marker-marker_prev));
		aia::imshow("Reconstruction result", marker);

		cv::Mat mask_filling = 255-marker.clone();
		cv::Mat marker_filling = 255-marker.clone();
		cv::rectangle(marker_filling, cv::Rect(1,1, marker.cols-2, marker.rows-2), cv::Scalar(0), CV_FILLED);
		aia::imshow("Marker for filling", marker_filling);

		marker_prev = marker_filling.clone();
		iters = 0;
		do 
		{
			printf("reconstruction iter # %d\n", ++iters);
			marker_filling.copyTo(marker_prev);
			cv::dilate(marker_filling, marker_filling, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
			marker_filling = marker_filling & mask_filling;

			//cv::imshow("Filling", marker_filling);
			//cv::waitKey(50);

		} while (cv::countNonZero(marker_filling-marker_prev));
		aia::imshow("Filling result", 255-marker_filling);

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

