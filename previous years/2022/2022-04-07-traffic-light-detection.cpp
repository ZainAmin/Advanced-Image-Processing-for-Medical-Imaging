// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace
{
	// parameters
	double A_min = 50;				// we assume that we cannot recognize lights smaller than 8 pixels in diameter
	double width_max_perc = 0.05;	// percentage of image width
	double C_min = 0.65;
	double S_min = 160;

	double avgValInContour(const cv::Mat& img, std::vector<cv::Point>& object)
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
					sum += yRow[x];
					count++;
				}
		}

		return sum / count;
	}

	cv::Mat frameProcessor(const cv::Mat& img)
	{
		cv::Mat out_img = img.clone();

		// STEP 1: enhance bright objects
		cv::Mat img_hsv;
		cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
		std::vector <cv::Mat> hsv_channels(3);
		cv::split(img_hsv, hsv_channels);
		cv::Mat binarized_v;
		cv::threshold(hsv_channels[2], binarized_v, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		// STEP 2: get connected components
		std::vector < std::vector <cv::Point> > all_objects;
		cv::findContours(binarized_v, all_objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		// STEP 3: get circle objects only = filter objects by shape
		double A_max = std::pow(0.5 * (width_max_perc * img.cols), 2) * ucas::PI;
		for (int i = 0; i < all_objects.size(); i++)
		{
			// first criterion (decision tree layer): area size
			double A = cv::contourArea(all_objects[i]);
			if (A >= A_min && A <= A_max)
			{
				// second criterion: circularity
				double p = cv::arcLength(all_objects[i], true);
				double C = 4 * ucas::PI * A / (p * p);
				if (C >= C_min)
				{
					// third criterion: color saturation
					double avg_S = avgValInContour(hsv_channels[1], all_objects[i]);
					if(avg_S >= S_min)
						cv::drawContours(out_img, all_objects, i, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
				}
			}
			
		}

		
		


		return out_img;
	}
}

int main()
{
	aia::processVideoStream(
		std::string(EXAMPLE_IMAGES_PATH) + "/traffic_light_5.mp4", 
		frameProcessor, "", true, 0, 1);

	return EXIT_SUCCESS;
}

