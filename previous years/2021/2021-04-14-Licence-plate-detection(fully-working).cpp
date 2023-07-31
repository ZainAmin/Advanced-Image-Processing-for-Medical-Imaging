// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef std::vector <cv::Point>  object;
typedef std::vector < object >   objects;

cv::RotatedRect correctedMinAreaRect(const object & obj)
{
	cv::RotatedRect rect = cv::minAreaRect(obj);
	if(rect.size.height > rect.size.width)
	{
		std::swap(rect.size.width, rect.size.height);
		rect.angle += 90.f;
	}
	return rect;
}

objects & filterObjects(objects & objs, bool (*pred)(object & obj))
{
	objs.erase( std::remove_if(objs.begin(), objs.end(), pred), objs.end());
	return objs;
}

bool areaCriterion(object & obj)
{
	double area = cv::contourArea(obj);
	return area < 1000 || area > 10000;
}

bool orientationCriterion(object & obj)
{
	//return correctedMinAreaRect(obj).angle > 10;

	// geometrical moments-based orientation estimation
	cv::Moments moments = cv::moments(obj, true);
	double angle = 0.5 * std::atan2(2*moments.mu11, moments.mu20-moments.mu02);
	return (abs(angle)/ucas::PI)*180 > 10;
}

bool aspectRatioCriterion(object & obj)
{
	cv::RotatedRect rect = correctedMinAreaRect(obj);
	float AR  = rect.size.width / rect.size.height;
	return std::abs(AR-4.7) > 2;
}

bool rectangularityCriterion(object & obj)
{
	cv::RotatedRect rect = correctedMinAreaRect(obj);
	return cv::contourArea(obj)/(rect.size.width*rect.size.height) < 0.7;
}

bool circularityCriterion(object & obj)
{
	float A = cv::contourArea(obj);
	float p = cv::arcLength(obj, true);
	float C = (4*ucas::PI*A)/(p*p);
	return C < 0.1;
}

bool sortByDescendingArea(object & first, object & second)
{
	return cv::contourArea(first) > contourArea(second);
}

cv::Mat licencePlateDetector(const cv::Mat & frame)
{
	cv::Mat result = frame.clone();

	// switch to grayscale and crop
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	cv::Rect results_box = cv::Rect(frame.cols-300, frame.rows-300, 300, 300);
	frame_gray(results_box).setTo(cv::Scalar(255));
	result(results_box).setTo(cv::Scalar(255, 255, 255));


	// top-hat to selectively enhance the white background in the license plate
	cv::Mat enhanced_plate_img;
	cv::morphologyEx(
		frame_gray, 
		enhanced_plate_img,
		cv::MORPH_TOPHAT, 
		cv::getStructuringElement(cv::MORPH_RECT, cv::Size(33, 11)));

	// otsu binarization
	cv::threshold(enhanced_plate_img, enhanced_plate_img, 0, 255,
		cv::THRESH_BINARY | cv::THRESH_OTSU);

	// connected component extraction
	objects candidate_objects;
	cv::findContours(enhanced_plate_img, candidate_objects,
		cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	
	// criterion 1: remove too small and too big objects
	filterObjects(candidate_objects, areaCriterion);

	// criterion 2: remove nonhorizontal objects
	filterObjects(candidate_objects, orientationCriterion);

	// criterion 3: remove objects with no 'licence-plate'-like aspect ratio
	filterObjects(candidate_objects, aspectRatioCriterion);

	// criterion 4: remove nonrectangular objects
	filterObjects(candidate_objects, rectangularityCriterion);

	// draw licence plate
	cv::drawContours(result, candidate_objects, -1, cv::Scalar(0, 255, 255), 2, CV_AA);

	// update trajectory
	static std::vector<cv::Point> trajectory;
	if(candidate_objects.size() == 1)
	{
		cv::Moments moments = cv::moments(candidate_objects[0], true);
		cv::Point center(moments.m10/moments.m00, moments.m01/moments.m00);
		
		// reset trajectory if current center is below last center on y-axis
		if(trajectory.size() && center.y > trajectory.back().y)
			trajectory.clear();

		trajectory.push_back(center);
	}

	// draw trajectory
	if(trajectory.size() > 1)
		for(int k=0; k<trajectory.size()-1; k++)
			cv::line(result, trajectory[k], trajectory[k+1], cv::Scalar(255, 0, 0), 2, CV_AA);


	// PHASE 2: License Plate Recognition
	if(candidate_objects.size() == 1)
	{
		// segment license plate characters
		cv::Mat plate_img;
		cv::morphologyEx(
			frame_gray(cv::boundingRect(candidate_objects[0])), 
			plate_img,
			cv::MORPH_BLACKHAT, 
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11)));

		// otsu binarization
		cv::Mat plate_img_bin;
		cv::threshold(plate_img, plate_img_bin, 0, 255,
			cv::THRESH_BINARY | cv::THRESH_OTSU);


		// connected component extraction
		objects candidate_chars;
		cv::findContours(plate_img_bin, candidate_chars,
			cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// remove elongated components
		filterObjects(candidate_chars, circularityCriterion);

		// remove components less than a certain percentage of plate height
		for(objects::iterator it = candidate_chars.begin(); it != candidate_chars.end(); )
		{
			if(cv::boundingRect(*it).height < 0.5*plate_img.rows)
				it = candidate_chars.erase(it);
			else
				it++;
		}

		cv::cvtColor(plate_img, plate_img, cv::COLOR_GRAY2BGR);
		
		std::sort(candidate_chars.begin(), candidate_chars.end(), sortByDescendingArea);

		if(candidate_chars.size() >= 7)
			for(int i=0; i<7; i++)
				cv::drawContours(plate_img, candidate_chars, i, cv::Scalar(0, 255, 255), 1, CV_AA);

		cv::imshow("Licence Plate", plate_img);
	}

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


