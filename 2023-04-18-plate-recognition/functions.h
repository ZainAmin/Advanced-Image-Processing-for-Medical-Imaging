#pragma once

#include "aiaConfig.h"
#include <opencv2/core/core.hpp>

// open namespace "aia"
namespace aia
{
	typedef std::vector< cv::Point > contour;
	typedef std::vector< std::vector <cv::Point> > contourSet;

	cv::RotatedRect correctedMinAreaRect(const contour& obj);

	std::string majorityVoting(std::vector <std::string>& plate_recognitions);

	class OCR
	{
		private:

			cv::Mat letters_image;
			cv::Mat digits_image;

			OCR();	// constructor inaccessible

		public:

			static OCR& instance();	// singleton design pattern

			// intensity-based matching
			char intensity_matching(cv::Mat& image, bool isLetter) throw (aia::error);
	};
}
