#include <iostream>
#include "functions.h"

using namespace aia;

cv::RotatedRect aia::correctedMinAreaRect(const contour& obj)
{
	cv::RotatedRect rect = cv::minAreaRect(obj);
	if (rect.size.height > rect.size.width)
	{
		std::swap(rect.size.width, rect.size.height);
		rect.angle += 90.f;
	}
	return rect;
}

OCR::OCR()
{
	letters_image = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/plate_letters_IT.png", cv::IMREAD_GRAYSCALE);
	digits_image = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/plate_numbers_IT.png", cv::IMREAD_GRAYSCALE);
	cv::threshold(letters_image, letters_image, 200, 255, cv::THRESH_BINARY_INV);
	cv::threshold(digits_image, digits_image, 200, 255, cv::THRESH_BINARY_INV);
}

OCR& OCR::instance()
{
	static OCR uniqueInstance;
	return uniqueInstance;
}

// intensity-based matching
char OCR::intensity_matching(cv::Mat& query_img, bool isLetter) throw (aia::error)
{
	// will be either (resized) letters or digits image			
	cv::Mat result;

	// we will slide the template (letters or digits) image over the query image
	// so that the best match will yield a minimum = location of recognized letter/digit
	float rescale_ratio = query_img.rows / 103.0f; // 103 = letter/digit height in the template image
	cv::Mat template_img;
	// letters/digits of query and template images must have similar dims
	cv::resize(isLetter ? letters_image : digits_image, template_img, cv::Size(0, 0), rescale_ratio, rescale_ratio);

	// squared intensity difference matching, best match will occur at minimum
	cv::matchTemplate(query_img, template_img, result, cv::TM_CCOEFF_NORMED);
	cv::Point minLoc;
	cv::Point matchLoc;
	cv::minMaxLoc(result, 0, 0, &minLoc, 0);
	matchLoc = minLoc;

	// map matching location to the corresponding letter/digit
	if (isLetter)
	{
		const char letter[] = { 'A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','S','T','V','W','X','Y','Z' };

		// map minimum location to letter
		bool flag = false;
		int k = 0;
		int x_min = template_img.cols / 70;
		int sum = (template_img.cols / 12) - 1;
		int y = template_img.rows / 2;
		while (!flag && k < 22)
		{
			if (matchLoc.y < y)
			{
				if ((matchLoc.x <= x_min + template_img.cols / 13) && (x_min <= matchLoc.x))
					flag = true;
				else
				{
					k++;
					x_min = x_min + sum;
				}
			}
			else
			{
				k = 12;
				y = template_img.rows;
			}
		}
		return letter[k];
	}
	else
	{
		if (matchLoc.x <= 70 * rescale_ratio)
			return '1';
		else if (matchLoc.x <= 150 * rescale_ratio)
			return '2';
		else if (matchLoc.x <= 235 * rescale_ratio)
			return '3';
		else if (matchLoc.x <= 315 * rescale_ratio)
			return '4';
		else if (matchLoc.x <= 395 * rescale_ratio)
			return '5';
		else if (matchLoc.x <= 476 * rescale_ratio)
			return '6';
		else if (matchLoc.x <= 555 * rescale_ratio)
			return '7';
		else if (matchLoc.x <= 640 * rescale_ratio)
			return '8';
		else if (matchLoc.x <= 720 * rescale_ratio)
			return '9';
		else
			return '0';
	}
}
