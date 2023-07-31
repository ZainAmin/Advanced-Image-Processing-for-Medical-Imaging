// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace aia;

cv::Mat licencePlateDetector(const cv::Mat& frame)
{
	// static variables for history data and processing
	static std::vector<cv::Point> trajectory;
	static std::vector<std::string> cur_plate_history;
	static std::vector<std::string> prev_plates;
	if (prev_plates.empty())
		prev_plates.push_back("*******");

	cv::Mat result = frame.clone();

	// switch to grayscale and crop
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	cv::Rect results_box = cv::Rect(frame.cols - 300, frame.rows - 300, 300, 300);
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
	contourSet candidate_objects;
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

	// PLATE DETECTION ENDED - draw license plate
	cv::drawContours(result, candidate_objects, -1, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

	// update trajectory
	if (candidate_objects.size() == 1)
	{
		cv::Moments moments = cv::moments(candidate_objects[0], true);
		cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);

		// reset trajectory and increase car count if current center is below last center
		if (trajectory.size() && center.y > trajectory.back().y)
		{
			trajectory.clear();
			cur_plate_history.clear();
			prev_plates.push_back("*******");
		}

		trajectory.push_back(center);
	}

	// draw trajectory
	if (trajectory.size() > 1)
		for (int k = 0; k < trajectory.size() - 1; k++)
			cv::line(result, trajectory[k], trajectory[k + 1], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);


	// PLATE RECOGNITION STARTS
	if (candidate_objects.size() == 1)
	{
		// segment license plate characters
		cv::Mat plate_img;
		cv::morphologyEx(
			frame_gray(cv::boundingRect(candidate_objects[0])),
			plate_img,
			cv::MORPH_BLACKHAT,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));

		// otsu binarization
		cv::Mat plate_img_bin;
		cv::threshold(plate_img, plate_img_bin, 0, 255,
			cv::THRESH_BINARY | cv::THRESH_OTSU);

		// connected component extraction
		contourSet candidate_chars;
		cv::findContours(plate_img_bin.clone(), candidate_chars,
			cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// remove elongated components
		filterObjects(candidate_chars, circularityCriterion);

		// remove components less than a certain percentage of plate height
		for (contourSet::iterator it = candidate_chars.begin(); it != candidate_chars.end(); )
		{
			if (cv::boundingRect(*it).height < 0.5 * plate_img.rows)
				it = candidate_chars.erase(it);
			else
				it++;
		}

		// sort by size
		std::sort(candidate_chars.begin(), candidate_chars.end(), sortByDescendingArea);

		// character detection = first 7 largest objects
		aia::contourSet plate_chars_objs;
		if (candidate_chars.size() >= 7)
			for (int i = 0; i < 7; i++)
				plate_chars_objs.push_back(candidate_chars[i]);

		// CHARACTER DETECTION ENDED
		if (plate_chars_objs.size() == 7)
		{
			// sort characters from the leftmost to the rightmost
			std::sort(plate_chars_objs.begin(), plate_chars_objs.end(), sortByAscendingX);

			// position-dependent character recognition
			std::string plate_chars;
			for (int k = 0; k < 7; k++)
			{
				plate_chars += aia::OCR::instance().intensity_matching(
					plate_img_bin(cv::boundingRect(plate_chars_objs[k])), k < 2 || k>4);

				//plate_chars += aia::OCR::instance().shape_matching(plate_chars_objs[k], k<2 || k>4);
			}

			cur_plate_history.push_back(plate_chars);
			//printf("%s\n", plate_chars.c_str());

			// update plates history
			std::string cur_plate = computeModeCharwise(cur_plate_history);
			prev_plates.pop_back();
			prev_plates.push_back(cur_plate);
		}
	}

	// display result
	for (int i = 0; i < prev_plates.size(); i++)
		cv::putText(result, ucas::strprintf("%02d) ", i + 1) + prev_plates[i], results_box.tl() + cv::Point(10, 30 + i * 25), 2, 0.7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

	return result;
}

int main()
{
	try
	{
		aia::processVideoStream(
			std::string(EXAMPLE_IMAGES_PATH) + "/traffic1.avi",
			licencePlateDetector, "", true, 0);
	}
	catch (aia::error& ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}

	return EXIT_SUCCESS;
}