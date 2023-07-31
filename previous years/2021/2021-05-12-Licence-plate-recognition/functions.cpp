#include "functions.h"
#include "aiaConfig.h"
#include "ucasConfig.h"

using namespace aia;

bool aia::sortByAscendingX(contour& first, contour& second)
{
	cv::Rect bbox1 = cv::boundingRect(first);
	cv::Rect bbox2 = cv::boundingRect(second);
	return bbox1.x < bbox2.x;
}

contourSet& aia::filterObjects(contourSet& objs, bool (*pred)(contour& obj))
{
	objs.erase(std::remove_if(objs.begin(), objs.end(), pred), objs.end());
	return objs;
}

bool aia::areaCriterion(contour& obj)
{
	double area = cv::contourArea(obj);
	return area < 1000 || area > 10000;
}

bool aia::orientationCriterion(contour& obj)
{
	cv::Moments moments = cv::moments(obj, true);
	double angle = 0.5 * std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02);
	return (abs(angle) / ucas::PI) * 180 > 10;
}

bool aia::aspectRatioCriterion(contour& obj)
{
	cv::RotatedRect rect = correctedMinAreaRect(obj);
	float AR = rect.size.width / rect.size.height;
	return std::abs(AR - 4.7) > 2;
}

bool aia::rectangularityCriterion(contour& obj)
{
	cv::RotatedRect rect = correctedMinAreaRect(obj);
	return cv::contourArea(obj) / (rect.size.width * rect.size.height) < 0.7;
}

bool aia::circularityCriterion(contour& obj)
{
	double A = cv::contourArea(obj);
	double p = cv::arcLength(obj, true);
	double C = (4 * ucas::PI * A) / (p * p);
	return C < 0.1;
}

bool aia::sortByDescendingArea(contour& first, contour& second)
{
	return cv::contourArea(first) > contourArea(second);
}

cv::RotatedRect aia::correctedMinAreaRect(contour& obj)
{
	cv::RotatedRect rect = cv::minAreaRect(obj);
	if (rect.size.height > rect.size.width)
	{
		std::swap(rect.size.width, rect.size.height);
		rect.angle += 90.f;
	}
	return rect;
}

std::string aia::computeModeCharwise(std::vector <std::string>& plate_recognitions)
{
	std::string result = "*******";

	for (int k = 0; k < 7; k++)
	{
		std::map<char, int> char_occurences;
		for (auto& p : plate_recognitions)
			char_occurences[p[k]]++;

		char mode = '*';
		int max_occurrence = 0;
		for (auto& co : char_occurences)
			if (co.second > max_occurrence)
			{
				max_occurrence = co.second;
				mode = co.first;
			}

		result[k] = mode;
	}

	return result;
}

OCR::OCR()
{
	letters_image = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/plate_letters_IT.png", cv::IMREAD_GRAYSCALE);
	digits_image = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/plate_numbers_IT.png", cv::IMREAD_GRAYSCALE);
	cv::threshold(letters_image, letters_image, 200, 255, cv::THRESH_BINARY_INV);
	cv::threshold(digits_image, digits_image, 200, 255, cv::THRESH_BINARY_INV);

	cv::findContours(letters_image.clone(), letters_shapes, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	cv::findContours(digits_image.clone(), digits_shapes, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::sort(digits_shapes.begin(), digits_shapes.end(), aia::sortByAscendingX);
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
	cv::matchTemplate(query_img, template_img, result, cv::TM_SQDIFF);
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

// shape-based matching
char OCR::shape_matching(contour& shape, bool isLetter) throw (aia::error)
{
	if (isLetter)
	{
		// to be implemented
		return '*';
	}
	else
	{
		// 1-nearest-neighbor classification
		double minDist = std::numeric_limits<double>::infinity();
		char digit;
		int best_matching_index = -1;
		for (int k = 0; k < digits_shapes.size(); k++)
		{
			double dist = cv::matchShapes(digits_shapes[k], shape, cv::CONTOURS_MATCH_I1, 0);
			if (dist < minDist)
			{
				minDist = dist;
				digit = 48 + (k == 9 ? 0 : k + 1);
				best_matching_index = k;
			}
		}

		cv::Mat digits_image_copy = digits_image.clone();
		cv::drawContours(digits_image_copy, digits_shapes, best_matching_index, cv::Scalar(128), cv::FILLED, cv::LINE_AA);
		aia::imshow("Best matched shape", digits_image_copy);

		return digit;
	}
}