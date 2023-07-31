// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <map>
#include <list>

namespace TLR
{
	enum class Color { UNKNOWN, GREEN, ORANGE, RED };

	// parameters
	double A_min = 50;				// lights are assumed to be larger than 8 pixels in diameter
	double width_max_perc = 0.05;	// lights are assumed to be smaller than width_max_perc*image.width
	double C_min = 0.65;			// circularity threshold
	double S_min = 100;				// saturation threshold
	double V_min = 150;				// value (in hsV color space) threshold
	double upper_perc = 0.9;		// we will consider only objects lying in the upper upper_perc of the image
	int green_hue = 60;				// reference green hue
	int orange_hue = 14;			// reference orange hue
	int red_hue = 3;				// reference red hue
	double g_min = 0.8;				// goodness threshold (applied on spatial majority voting)
	int history_size = 20;			// for time-voting (see below)
	std::list<Color> color_history;	// contains last history_size detected frame colors

	// utility color 2 cv::Scalar conversion function
	cv::Scalar color2scalar(Color color)
	{
		if (color == Color::GREEN)
			return cv::Scalar(0, 255, 0);
		else if (color == Color::ORANGE)
			return cv::Scalar(0, 215, 255);
		else if (color == Color::RED)
			return cv::Scalar(0, 0, 255);
		else
			return cv::Scalar(128, 128, 128);
	}
	std::string color2text(Color color)
	{
		if (color == Color::GREEN)
			return "GO!";
		else if (color == Color::ORANGE)
			return "SLOW DOWN!";
		else if (color == Color::RED)
			return "STOP";
		else
			return "...";
	}

	// utility function: returns average intensity within the given contour
	double avgValInContour(
		const cv::Mat& img, 
		std::vector<cv::Point>& object, 
		bool cosine_correction = false)	// mirror hue values if img = hue channel
	{
		double sum = 0;
		int count = 0;
		cv::Rect bbox = cv::boundingRect(object);
		for (int y = bbox.y; y < bbox.y + bbox.height; y++)
		{
			const unsigned char* yRow = img.ptr<unsigned char>(y);
			for (int x = bbox.x; x < bbox.x + bbox.width; x++)
				if (cv::pointPolygonTest(object, cv::Point(x, y), false) > 0)
				{
					if (cosine_correction)
						sum += ucas::rad2deg(std::acos(std::cos(ucas::deg2rad(yRow[x] * 2.0)))) / 2.0;
					else
						sum += yRow[x];
					count++;
				}
		}

		return sum / count;
	}

	// utility function: returns detection 'goodness' score in [0,1]
	double goodness(double distH, double avgS, double avgV, double C)
	{
		return (C + avgS / 255.0 + avgV / 255.0) / 3;
	}


	// frame-by-frame processing function
	cv::Mat frameProcessor(const cv::Mat& img)
	{
		cv::Mat out_img = img.clone();
		std::map<Color, double> spatial_voting;

		// STEP 1: detect bright objects
		cv::Mat img_hsv;
		cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
		std::vector <cv::Mat> hsv_channels(3);
		cv::split(img_hsv, hsv_channels);
		cv::Mat binarized_v;
		cv::threshold(hsv_channels[2], binarized_v, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		// STEP 2: get connected components
		std::vector < std::vector <cv::Point> > all_objects;
		cv::findContours(binarized_v, all_objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		// STEP 3: traffic light detection decision tree + color recognition
		double A_max = std::pow(0.5 * (width_max_perc * img.cols), 2) * ucas::PI;
		for (int i = 0; i < all_objects.size(); i++)
		{
			// NODE 1: object location along Y axis
			cv::Rect bbox = cv::boundingRect(all_objects[i]);
			if (bbox.y + bbox.height / 2.0 > upper_perc * img.rows)
				continue;

			// NODE 2: area size
			double A = cv::contourArea(all_objects[i]);
			if (A < A_min || A > A_max)
				continue;

			// NODE 3: circularity
			double p = cv::arcLength(all_objects[i], true);
			double C = 4 * ucas::PI * A / (p * p);
			if (C < C_min)
				continue;

			// NODE 4: color saturation
			double avg_S = avgValInContour(hsv_channels[1], all_objects[i]);
			if (avg_S < S_min)
				continue;

			// NODE 5: luminosity
			double avg_V = avgValInContour(hsv_channels[2], all_objects[i]);
			if (avg_V < V_min)
				continue;
			
			// color recognition by minimizing distance from reference hues
			double avg_H = avgValInContour(hsv_channels[0], all_objects[i], true);
			double dist_green	= std::abs(avg_H - green_hue);
			double dist_orange	= std::abs(avg_H - orange_hue);
			double dist_red		= std::abs(avg_H - red_hue);
			double dist_H = 0;
			Color light_color = Color::UNKNOWN;
			if (dist_green < dist_orange && dist_green < dist_red)
			{
				light_color = Color::GREEN;
				dist_H = dist_green;
			}
			else if (dist_orange < dist_green && dist_orange < dist_red)
			{
				light_color = Color::ORANGE;
				dist_H = dist_orange;
			}
			else
			{
				light_color = Color::RED;
				dist_H = dist_red;
			}

			// add to spatial voting map
			double g = goodness(dist_H, avg_S, avg_V, C);
			spatial_voting[light_color] += g;
			
			// show result
			cv::Scalar cv_color = color2scalar(light_color);
			cv::drawContours(out_img, all_objects, i, cv_color, 3, cv::LINE_AA);
			cv::putText(out_img, ucas::strprintf("H = %.0f", avg_H), bbox.br() - cv::Point(-5, bbox.height), 2, 0.4, cv_color);
			cv::putText(out_img, ucas::strprintf("S = %.0f", avg_S), bbox.br() - cv::Point(-5, bbox.height - 13), 2, 0.4, cv_color);
			cv::putText(out_img, ucas::strprintf("V = %.0f", avg_V), bbox.br() - cv::Point(-5, bbox.height - 26), 2, 0.4, cv_color);
			cv::putText(out_img, ucas::strprintf("C = %.2f", C), bbox.br() - cv::Point(-5, bbox.height - 39), 2, 0.4, cv_color);
			cv::putText(out_img, ucas::strprintf("g = %.2f", g), bbox.br() - cv::Point(-5, bbox.height - 52), 2, 0.4, cv_color);

		}

		// current frame color = color with maximum accumulated goodness (majority voting in space domain)
		double max_g = g_min;
		Color frame_color = Color::UNKNOWN;
		for (auto it : spatial_voting)
			if (it.second >= max_g)
			{
				max_g = it.second;
				frame_color = it.first;
			}

		// update color history
		color_history.push_back(frame_color);
		if (color_history.size() > history_size)
			color_history.pop_front();

		// DECISION = most frequent color in color history (majority voting along in time domain)
		std::map<Color, int> time_voting;
		for (auto& col : color_history)
			time_voting[col]++;
		int max_freq = 0;
		Color decision = Color::UNKNOWN;
		for (auto it : time_voting)
			if (it.second >= max_freq)
			{
				max_freq = it.second;
				decision = it.first;
			}

		// display decision
		cv::putText(out_img, color2text(decision), cv::Point(100, 100), 2, 3, color2scalar(decision), 3);

		return out_img;
	}
}

int main()
{
	for(int i=6; i<=6; i++)
		aia::processVideoStream(
			ucas::strprintf("%s/traffic_light_%d.mp4", EXAMPLE_IMAGES_PATH, i), 
			TLR::frameProcessor, "", true, 0, 1);

	return EXIT_SUCCESS;
}