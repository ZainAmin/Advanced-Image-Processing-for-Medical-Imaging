// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// suppose these are 3-channel images (BGR)
// suppose img2 >= img1
bool areImagesEqual(cv::Mat img1, cv::Mat img2)
{
	cv::Mat img1_chans[3];
	cv::split(img1, img1_chans);

	cv::Mat img2_chans[3];
	cv::split(img2, img2_chans);

	return cv::countNonZero(img2_chans[0]-img1_chans[0]) == 0 &&
		   cv::countNonZero(img2_chans[1]-img1_chans[1]) == 0 && 
		   cv::countNonZero(img2_chans[2]-img1_chans[2]) == 0;
}

int main() 
{
	try
	{
		// load an image where there are lines that can be detected
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy.jpg");
		if(!img.data)
			throw aia::error("Cannot open image");

		// marker generation
		cv::Mat marker;
		cv::morphologyEx(img, 
			marker, 
			cv::MORPH_OPEN,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31,31)));
		aia::imshow("marker", marker, true, 0.5);

		// reconstruction
		cv::Mat marker_prev;
		int iter = 0;
		do 
		{
			// make a backup
			marker_prev = marker.clone();

			// geodesic dilation
			cv::morphologyEx(marker, 
				marker, 
				cv::MORPH_DILATE,
				cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
			cv::min(marker, img, marker);

			//printf("iteration = %d\n", ++iter);
			cv::imshow("reconstruction", marker);
			if (cv::waitKey(200)>=0)
				cv::destroyWindow("reconstruction");

		} while ( !areImagesEqual(marker_prev, marker));

		aia::imshow("reconstructed", marker, true, 0.5);
		aia::imshow("stars", img-marker, true, 0.5);

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