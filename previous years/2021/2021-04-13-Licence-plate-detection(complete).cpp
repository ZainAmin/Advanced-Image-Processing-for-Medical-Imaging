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

bool almostHorizontalCriterion(std::vector <cv::Point> & obj)
{
	// geometrical moments-based orientation estimation
	cv::Moments moments = cv::moments(obj, true);
	double angle = 0.5 * std::atan2(2*moments.mu11, moments.mu20-moments.mu02);

	// minimum area-based orientation estimation: does not work well
	//double angle = cv::minAreaRect(obj).angle;

	return (abs(angle)/ucas::PI)*180 > 10;
}

bool aspectRatioCriterion(std::vector <cv::Point> & first, std::vector <cv::Point> & second)
{
	cv::RotatedRect first_rect = cv::minAreaRect(first);
	cv::RotatedRect second_rect = cv::minAreaRect(second);
	float first_AR  = first_rect.size.width / first_rect.size.height;
	float second_AR = second_rect.size.width / second_rect.size.height;
	return std::abs(first_AR-4.7) < std::abs(second_AR-4.7);
}

cv::Mat licencePlateDetector(const cv::Mat & frame)
{
	cv::Mat result = frame.clone();

	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	frame_gray(cv::Rect(frame.cols-200, frame.rows-300, 200, 300)).setTo(cv::Scalar(255));

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

	// criterion 2: remove nonhorizontal objects
	candidate_objects.erase(
		std::remove_if(
		candidate_objects.begin(), 
		candidate_objects.end(), 
		almostHorizontalCriterion), 
		candidate_objects.end());

	// criterion 3: sort objects based on the difference w.r.t the ideal aspect ratio
	std::sort(candidate_objects.begin(), candidate_objects.end(), aspectRatioCriterion);

	cv::drawContours(result, candidate_objects, 0, cv::Scalar(0, 255, 255), 2, CV_AA);

	return result;
}

int main() 
{
	try
	{
		aia::processVideoStream(
			std::string(EXAMPLE_IMAGES_PATH) + "/traffic1.avi", 
			licencePlateDetector, "", true, 200);
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


