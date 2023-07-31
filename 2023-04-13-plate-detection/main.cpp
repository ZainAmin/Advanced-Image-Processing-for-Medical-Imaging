// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// parameters
int min_area = 1000;
int max_area = 10000; // warning: this should scale with image resolution
int max_angle = 10;
float min_rectangularity = 0.7;
float ideal_aspect_ratio = 4.5;
float aspect_ratio_tolerance = 1.5;

cv::Mat frameProcessor(const cv::Mat& frame)
{
	// switch to grayscale (alternatively you can try to process the (HS)V or L(ab) channel)
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

	// cropping bottom-right rectangle that (just for this video)
	cv::Rect crop_rect = cv::Rect(frame_gray.cols - 300, frame_gray.rows - 300, 300, 300);
	frame_gray(crop_rect).setTo(cv::Scalar(255));

	// plate enhancement with grayscale morphological tophat
	cv::Mat tophat;
	cv::morphologyEx(frame_gray, tophat, cv::MORPH_TOPHAT,
		cv::getStructuringElement(cv::MORPH_RECT, cv::Size(33, 11)));

	// binarization (any method would work thanks to tophat enhancement)
	cv::Mat tophat_bin;
	cv::threshold(tophat, tophat_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

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

	cv::Mat output_frame = frame.clone();
	cv::drawContours(output_frame, candidate_plates, -1, cv::Scalar(0, 0, 255), 2);

	for (auto& candidate : candidate_plates)
	{
		cv::Rect bounding_rect = cv::boundingRect(candidate);
		cv::Mat plate_img = frame_gray(bounding_rect);

		// to make recognition more robust, it would be convenient to
		// correct the orientation using warpAffine opencv's method
		// below is a first attempt, to be fixed/improved
		/*cv::RotatedRect minarea_rect = aia::correctedMinAreaRect(candidate);
		int angle = int(minarea_rect.angle) % 180;
		cv::Mat corrected_plate_img;
		cv::warpAffine(plate_img, corrected_plate_img, 
			cv::getRotationMatrix2D(minarea_rect.center, angle, 1), cv::Size(-1, -1))
		aia::imshow("corrected plate image", corrected_plate_img, false, 3.0f);*/

		cv::Mat bottom_hat;
		cv::morphologyEx(plate_img, bottom_hat, cv::MORPH_BLACKHAT,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
		cv::threshold(bottom_hat, bottom_hat, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


		aia::imshow("binarized characters", bottom_hat, false, 3.0f);

	}
	/*for (auto& candidate : candidate_plates)
	{
		float angle = aia::correctedMinAreaRect(candidate).angle;

		cv::putText(output_frame, ucas::strprintf("%.0f", angle), candidate[0], 2, 1, cv::Scalar(0, 255, 255));
	}*/

	return output_frame;
}

int main()
{
	aia::processVideoStream(
		std::string(EXAMPLE_IMAGES_PATH) + "/traffic1.avi",
		frameProcessor, "", true, 100, 1.0f);

	return EXIT_SUCCESS;
}

