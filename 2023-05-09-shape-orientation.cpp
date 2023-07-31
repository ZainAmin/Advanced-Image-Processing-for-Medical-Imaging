// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: show the difference in shape orientation estimation between 
//       minimum area bounding rect and geometrical moments

namespace aia
{
	double degrees2rad(double degrees) {
		return (degrees / 180) * aia::PI;
	}
}

cv::RotatedRect correctedMinAreaRect(const std::vector < cv::Point >& obj)
{
	cv::RotatedRect rect = cv::minAreaRect(obj);
	if (rect.size.height > rect.size.width)
	{
		std::swap(rect.size.width, rect.size.height);
		rect.angle += 90.f;
	}
	return rect;
}

int main()
{

	// read the image
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.png");
	if (!img.data)
		throw aia::error("cannot read image");

	// convert to gray
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	// binarize with Triangle method
	std::vector<int> histo = ucas::histogram(img_gray);
	cv::threshold(img_gray, img_gray, ucas::getTriangleAutoThreshold(histo), 255, cv::THRESH_BINARY);

	// remove small structures
	cv::morphologyEx(img_gray, img_gray, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));


	// compute shape orientation with minimum area rectangles
	// cv::minAreaRect
	std::vector < std::vector <cv::Point> > contours;
	cv::findContours(img_gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	for (int k = 0; k < contours.size(); k++)
	{
		cv::Point2f points[4];
		cv::RotatedRect rect = correctedMinAreaRect(contours[k]);
		rect.points(points);

		for (int p = 0; p < 4; p++)
			cv::line(img, points[p], points[(p + 1) % 4], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

		cv::circle(img, rect.center, 5, cv::Scalar(255, 0, 0), cv::FILLED);

		cv::line(img,
			rect.center - cv::Point2f(100 * cos(aia::degrees2rad(rect.angle)), 100 * sin(aia::degrees2rad(rect.angle))),
			rect.center + cv::Point2f(100 * cos(aia::degrees2rad(rect.angle)), 100 * sin(aia::degrees2rad(rect.angle))), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
	}


	// compute shape orientations with geometrical moments
	for (int k = 0; k < contours.size(); k++)
	{
		cv::Moments moments = cv::moments(contours[k], true);
		cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);
		cv::circle(img, center, 5, cv::Scalar(0, 255, 255), cv::FILLED);

		double angle = 0.5 * std::atan2(2 * moments.mu11, moments.mu20 - moments.mu02);
		cv::line(img,
			center - cv::Point2f(100 * cos(angle), 100 * sin(angle)),
			center + cv::Point2f(100 * cos(angle), 100 * sin(angle)), cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
	}

	aia::imshow("Comparison", img, true);

	return EXIT_SUCCESS;
}