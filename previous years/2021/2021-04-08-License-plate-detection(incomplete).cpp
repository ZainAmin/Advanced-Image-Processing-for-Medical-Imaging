// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

bool areaCriterion(std::vector <cv::Point> & obj)
{
	double area = cv::contourArea(obj);

	return area < 1000 || area > 10000;
}

cv::Mat licencePlateDetector(const cv::Mat & frame)
{
	cv::Mat result = frame.clone();

	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

	cv::Mat enhanced_plate_img;

	// approach 1: bottom-hat to selectively enhance characters 
	// in the license plate
	/*cv::morphologyEx(
		frame_gray, 
		enhanced_plate_img,
		cv::MORPH_BLACKHAT, 
		cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11)));*/

	// approach 2: top-hat to selectively enhance the white background
	// in the license plate
	cv::morphologyEx(
		frame_gray, 
		enhanced_plate_img,
		cv::MORPH_TOPHAT, 
		cv::getStructuringElement(cv::MORPH_RECT, cv::Size(33, 11)));

	// otsu binarization
	cv::threshold(enhanced_plate_img, enhanced_plate_img, 0, 255,
		cv::THRESH_BINARY | cv::THRESH_OTSU);

	// connected component extraction
	std::vector < std::vector <cv::Point> > candidate_objects;
	cv::findContours(enhanced_plate_img, candidate_objects,
		cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	
	// criterion 1: remove too small and too big objects
	candidate_objects.erase(
		std::remove_if(
			candidate_objects.begin(), 
			candidate_objects.end(), 
			areaCriterion), 
		candidate_objects.end());


	cv::drawContours(result, candidate_objects, -1, cv::Scalar(0, 255, 255), 2, CV_AA);

	return result;
}

int main() 
{
	try
	{
		aia::processVideoStream(
			std::string(EXAMPLE_IMAGES_PATH) + "/traffic1.avi", 
			licencePlateDetector);
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}

	return EXIT_SUCCESS;
}


