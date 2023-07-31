// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// plate detection parameters
cv::Size tophat_kernel = cv::Size(33, 11);
int min_area = 1000;
int max_area = 10000; // warning: this should scale with image resolution
int max_angle = 10;
float min_rectangularity = 0.7f;
float ideal_aspect_ratio = 4.5f;
float aspect_ratio_tolerance = 1.5f;
bool visual_debug = false;

// characters recognition parameters
cv::Size bottomhat_kernel = cv::Size(11, 11);
float min_relative_height = 0.5;	// minimum relative height w.r.t. plate height

cv::Mat frameProcessor(const cv::Mat& frame)
{
	static std::vector< std::string > cur_plate_recognitions;
	static std::vector<std::string> all_plates_history;
	static int prev_plate_y = -1;
	cv::Mat output_frame = frame.clone();

	// *** PLATE DETECTION STARTED ***
	// switch to grayscale (alternatively you can try to process the (HS)V or L(ab) channel)
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

	// cropping bottom-right rectangle (only for traffic1 video)
	cv::Rect crop_rect = cv::Rect(frame_gray.cols - 300, frame_gray.rows - 300, 300, 300);
	frame_gray(crop_rect).setTo(cv::Scalar(255));

	// plate enhancement with grayscale morphological tophat
	cv::Mat tophat;
	cv::morphologyEx(frame_gray, tophat, cv::MORPH_TOPHAT,
		cv::getStructuringElement(cv::MORPH_RECT, tophat_kernel));
	if (visual_debug)
		aia::imshow("Image enhancement with tophat", tophat, false);

	// binarization (any method would work thanks to tophat enhancement)
	cv::Mat tophat_bin;
	cv::threshold(tophat, tophat_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	if (visual_debug)
		aia::imshow("Image binarization", tophat, false);

	// extract candidate regions
	std::vector< std::vector<cv::Point> > candidate_plates;
	cv::findContours(tophat_bin, candidate_plates, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	// filter by touching image borders
	candidate_plates.erase(std::remove_if(candidate_plates.begin(), candidate_plates.end(),
		[frame](const std::vector<cv::Point>& object)
	{
		cv::Rect bounding_rect = cv::boundingRect(object);

		return bounding_rect.x < 5 ||
			   bounding_rect.x + bounding_rect.width >= frame.cols - 6 ||
			   bounding_rect.y < 5 ||
			   bounding_rect.y + bounding_rect.height >= frame.rows - 6;
	}
	), candidate_plates.end());

	// filter by area
	candidate_plates.erase(std::remove_if(candidate_plates.begin(), candidate_plates.end(),
		[](const std::vector<cv::Point>& object)
	{
		double area = cv::contourArea(object);
		return area < min_area || area > max_area;
	}
	), candidate_plates.end());

	// filter by orientation
	candidate_plates.erase(std::remove_if(candidate_plates.begin(), candidate_plates.end(),
		[](const std::vector<cv::Point>& object)
	{
		return int(aia::correctedMinAreaRect(object).angle)%180 > max_angle;
	}
	), candidate_plates.end());

	// filter by rectangularity
	candidate_plates.erase(std::remove_if(candidate_plates.begin(), candidate_plates.end(),
		[](const std::vector<cv::Point>& object)
	{
		double area = cv::contourArea(object);
		cv::Rect bounding_rect = cv::boundingRect(object);
		return area / bounding_rect.area() < min_rectangularity;
	}
	), candidate_plates.end());

	// filter by aspect ratio
	candidate_plates.erase(std::remove_if(candidate_plates.begin(), candidate_plates.end(),
		[](const std::vector<cv::Point>& object)
	{
		cv::RotatedRect rot_rect = aia::correctedMinAreaRect(object);
		return std::abs(rot_rect.size.aspectRatio() - ideal_aspect_ratio) > aspect_ratio_tolerance;
	}
	), candidate_plates.end());
	// *** PLATE DETECTION ENDED ***
	

	// *** PLATE RECOGNITION STARTED ***
	// ASSUMPTION: 1 plate in the image
	if (candidate_plates.size() == 1)
	{
		cv::drawContours(output_frame, candidate_plates, 0, cv::Scalar(0, 0, 255), 2);
		aia::contour detected_plate = candidate_plates[0];

		// reset plate recognition history if new plate enters the image
		cv::Rect plate_rect = cv::boundingRect(detected_plate);
		if (plate_rect.y > prev_plate_y)
		{
			cur_plate_recognitions.clear();
			all_plates_history.push_back("******");
		}
		prev_plate_y = plate_rect.y;

		// *** PLATE RECOGNITION STARTED ***
		cv::Mat plate_img = frame_gray(plate_rect);

		// correct the orientation using warpAffine opencv's method
		// to make character recognition (template-matching-based) more robust
		cv::RotatedRect minarea_rect = aia::correctedMinAreaRect(detected_plate);
		int angle = int(minarea_rect.angle) % 180;
		cv::Mat corrected_plate_img;
		cv::warpAffine(plate_img, corrected_plate_img,
			cv::getRotationMatrix2D(cv::Point2f(plate_img.cols/2.f, plate_img.rows/2.f), angle, 1), cv::Size(-1, -1));
		if (visual_debug)
			aia::imshow("Orientation-corrected plate image", corrected_plate_img, false, 3.0f);

		// character enhancement with morphological bottom hat
		// WARNING: not necessary if the plate is "clean", meaning that there are
		// no shadows (this requires a perfect [for us] cloudy day)
		cv::Mat bottom_hat;
		cv::morphologyEx(corrected_plate_img, bottom_hat, cv::MORPH_BLACKHAT,
			cv::getStructuringElement(cv::MORPH_RECT, bottomhat_kernel));
		if (visual_debug)
			aia::imshow("Enhanced characters", bottom_hat, false, 3.0f);

		// binarization (any method would work thanks to tophat enhancement)
		cv::threshold(bottom_hat, bottom_hat, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		if (visual_debug)
			aia::imshow("Binarized characters", bottom_hat, false, 3.0f);

		// extract candidate characters
		cv::Mat plate_bin = bottom_hat;
		std::vector< std::vector<cv::Point> > candidate_characters;
		cv::findContours(plate_bin, candidate_characters, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		// filter by relative height
		candidate_characters.erase(std::remove_if(candidate_characters.begin(), candidate_characters.end(),
			[plate_bin](const std::vector<cv::Point>& object)
		{
			cv::Rect bounding_rect = cv::boundingRect(object);
			return bounding_rect.height < min_relative_height* plate_bin.rows;
		}
		), candidate_characters.end());

		// filter by vertical orientation
		candidate_characters.erase(std::remove_if(candidate_characters.begin(), candidate_characters.end(),
			[plate_bin](const std::vector<cv::Point>& object)
		{
			cv::Rect bounding_rect = cv::boundingRect(object);
			return bounding_rect.height < bounding_rect.width;
		}
		), candidate_characters.end());

		// filter by touching image borders
		candidate_characters.erase(std::remove_if(candidate_characters.begin(), candidate_characters.end(),
			[plate_img](const std::vector<cv::Point>& object)
		{
			cv::Rect bounding_rect = cv::boundingRect(object);

			int margin = 1;
			return bounding_rect.x < margin ||
				bounding_rect.x + bounding_rect.width >= plate_img.cols - margin - 1 ||
				bounding_rect.y < margin ||
				bounding_rect.y + bounding_rect.height >= plate_img.rows - margin - 1;
		}
		), candidate_characters.end());

		// re-extract contours, only externals
		plate_bin.setTo(cv::Scalar(0));
		cv::drawContours(plate_bin, candidate_characters, -1, cv::Scalar(255), cv::FILLED);
		candidate_characters.clear();
		cv::findContours(plate_bin, candidate_characters, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		cv::drawContours(plate_img, candidate_characters, -1, cv::Scalar(255), 1);
		if (visual_debug)
			aia::imshow("Filtered characters", plate_img, false, 3.0f);

		// early exit if no. of characters is not 7
		// we are unable to select the 'good' 7 if there are more
		// this can be improved of course
		if (candidate_characters.size() == 7)
		{
			// since italian plates have the format LL DDD LL (L = letter, D = digit)
			// we sort by ascending x so to implement position-dependent recognition
			std::sort(candidate_characters.begin(), candidate_characters.end(),
				[](aia::contour& left, aia::contour& right)
			{
				cv::Rect left_br = cv::boundingRect(left);
				cv::Rect right_br = cv::boundingRect(right);
				return left_br.x < right_br.x;
			});

			// current plate instance recognition
			std::string plate_str = "";
			for (int k = 0; k < candidate_characters.size(); k++)
			{
				cv::Mat cropped_char = corrected_plate_img(cv::boundingRect(candidate_characters[k]));
				char recognized_char = aia::OCR::instance().intensity_matching(cropped_char, k < 2 || k > 4);
				plate_str += recognized_char;
			}
			cur_plate_recognitions.push_back(plate_str);
			std::string cur_plate = aia::majorityVoting(cur_plate_recognitions);
			if (cur_plate_recognitions.size() > 2)
			{
				all_plates_history.pop_back();
				all_plates_history.push_back(cur_plate);
			}
		}
	}
	// *** PLATE RECOGNITION ENDED ***
	

	// display result
	cv::Rect result_box = crop_rect;
	for (int i = 0; i < all_plates_history.size(); i++)
		cv::putText(output_frame, ucas::strprintf("%02d] ", i + 1) + all_plates_history[i], cv::Point(frame.cols - 230, 30 + i * 40), 2, 0.9, cv::Scalar(0, 0, 0), i == all_plates_history.size() - 1 ? 2 : 1, cv::LINE_AA);

	return output_frame;
}

int main()
{
	aia::processVideoStream(
		std::string(EXAMPLE_IMAGES_PATH) + "/traffic1.avi",
		frameProcessor, "", true, 0, 0.8f);

	return EXIT_SUCCESS;
}

