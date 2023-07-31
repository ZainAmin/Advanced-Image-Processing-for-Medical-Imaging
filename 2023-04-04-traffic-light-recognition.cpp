// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <functional>

// <>

// parameters
int min_area = 30;
int max_area = 1000; // warning: this should scale with image resolution
double min_circularity = 0.7;
int min_saturation = 128;
int hue_red = 0;
int hue_orange = 30;
int hue_green = 120;

double avgIntensity(
	const cv::Mat& single_chan_img, 
	std::vector<cv::Point> contour,
	std::function <unsigned char(unsigned char)> pixel_transform = [](unsigned char x) {return x; })
{
	cv::Rect bbox = cv::boundingRect(contour);
	double avg = 0;
	int count = 0;
	for (int y = bbox.y; y < bbox.y + bbox.height; y++)
	{
		const unsigned char* yRow = single_chan_img.ptr(y);
		for (int x = bbox.x; x < bbox.x + bbox.width; x++)
			if (cv::pointPolygonTest(contour, cv::Point(x, y), false) > 0)
			{
				count++;
				avg += pixel_transform( yRow[x] );
			}
	}
	return avg / count;
}

cv::Mat frameProcessor(const cv::Mat& img)
{
	cv::Mat img_out = img.clone();

	cv::Mat img_HSV;
	cv::cvtColor(img, img_HSV, cv::COLOR_BGR2HSV);
	std::vector <cv::Mat> hsv_chans(3);
	cv::split(img_HSV, hsv_chans);

	std::vector<int> V_hist = ucas::histogram(hsv_chans[2]);
	int T_otsu = ucas::getOtsuAutoThreshold(V_hist);
	cv::Mat V_binarized;
	cv::threshold(hsv_chans[2], V_binarized, T_otsu, 255, cv::THRESH_BINARY);

	std::vector< std::vector<cv::Point> > candidate_lights;
	cv::findContours(V_binarized, candidate_lights, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	// filter by area
	candidate_lights.erase(std::remove_if(candidate_lights.begin(), candidate_lights.end(),
		[](const std::vector<cv::Point>& object)
	{
		double area = cv::contourArea(object);
		return area < min_area || area > max_area;
	}
	), candidate_lights.end());

	// filter by circularity
	candidate_lights.erase(std::remove_if(candidate_lights.begin(), candidate_lights.end(),
		[](const std::vector<cv::Point>& object)
	{
		double area = cv::contourArea(object);
		double p = cv::arcLength(object, true);
		double C = 4 * ucas::PI * area / (p * p);
		return C < min_circularity;
	}
	), candidate_lights.end());

	// filter by average saturation or lightness
	candidate_lights.erase(std::remove_if(candidate_lights.begin(), candidate_lights.end(),
		[hsv_chans](const std::vector<cv::Point>& object)
	{
		return avgIntensity(hsv_chans[1], object) < min_saturation;
	}
	), candidate_lights.end());


	// traffic light recognition
	for (int k = 0; k < candidate_lights.size(); k++)
	{
		double avgHue = avgIntensity(
			hsv_chans[0], 
			candidate_lights[k],
			[](unsigned char val) 
		{
			return ucas::rad2deg(std::acos(std::cos(ucas::deg2rad(double(val*2))))); 
		});

		float dist_to_red = avgHue - hue_red;
		float dist_to_orange = std::abs(avgHue - hue_orange);
		float dist_to_green = std::abs(avgHue - hue_green);

		if (dist_to_green < dist_to_orange && dist_to_green < dist_to_red)
			cv::drawContours(img_out, candidate_lights, k, cv::Scalar(0, 255, 0), 2);
		else if(dist_to_orange < dist_to_red && dist_to_orange < dist_to_green)
			cv::drawContours(img_out, candidate_lights, k, cv::Scalar(0, 255, 255), 2);
		else
			cv::drawContours(img_out, candidate_lights, k, cv::Scalar(0, 0, 255), 2);
	}

	return img_out;
}

int main()
{
	aia::processVideoStream(
		std::string(EXAMPLE_IMAGES_PATH) + "/traffic_light_6.mp4", 
		frameProcessor, "", true);

	return EXIT_SUCCESS;
}