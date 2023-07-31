// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace TLR
{
	// parameters
	float minArea = float(ucas::PI * 5  * 5);
	float maxArea = float(ucas::PI * 50 * 50);
	float minCircularity = 0.65f;
	float minSaturation = 160.0f;
	float minValue = 180.0f;
	int opening = 7;
	int orangeHue = 15;
	int greenHue = 60;

	// debug and testing
	bool visual_debug = false;
	int groundtruth = 0;		// true number of traffic lights in the video

	// utility function
	float averageIntensity(const cv::Mat & img, const std::vector <cv::Point> & contour)
	{
		float sum = 0;
		int count = 0;
		cv::Rect bbox = cv::boundingRect(contour);
		for(int y=bbox.y; y < bbox.y + bbox.height; y++)
		{
			const unsigned char* yRow = img.ptr<unsigned char>(y);
			for(int x=bbox.x; x < bbox.x + bbox.width; x++)
				if(cv::pointPolygonTest(contour, cv::Point(x, y), false) > 0)
				{
					sum += yRow[x];
					count++;
				}
		}
		return sum/count;
	}


	cv::Mat trafficLightRecognizer(const cv::Mat & frame)
	{
		cv::Mat result = frame.clone();

		// 1st step: traffic light detection
		// criteria: 
		// - find round objects (OK)
		// - find high-saturated objects (OK)
		// - find 3 circles aligned vertically (OK! But requires advanced tools)
		// - find bright objects in HSV (with high V) (OK)
		// - find not-to-small and not-too-large objects (range on Area) (OK...first criterion to apply)

		cv::Mat hsv;
		cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
		std::vector <cv::Mat> hsv_channels;
		cv::split(hsv, hsv_channels);

		// first attempt: binarize on H(SV) to find immediately red, green, and yellow objects
		/*
		cv::Mat red_objects;
		cv::inRange(hsv_channels[0], 0, 10, red_objects);
		cv::Mat tmp;
		cv::inRange(hsv_channels[0], 170, 180, tmp);
		red_objects += tmp;
		// ... to be continued
		*/

		// second attempt: binarize on (HS)V to find bright objects
		//cv::imshow("histogram", ucas::imhist(hsv_channels[2]));
		cv::Mat bright_objects_img;
		cv::threshold(hsv_channels[2], bright_objects_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		// opening to remove protrusions that can affect the rounded shape of traffic lights
		cv::morphologyEx(bright_objects_img, bright_objects_img, cv::MORPH_OPEN,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(opening, opening)));
	
		// object extraction
		std::vector < std::vector <cv::Point> > candidate_objects;
		cv::findContours(bright_objects_img.clone(), candidate_objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
		//cv::drawContours(frame_copy, candidate_objects, -1, cv::Scalar(0, 255, 255), 2, CV_AA);

		// apply abovementioned traffic light detection criteria
		std::vector <int> detected_objects;
		for(int i=0; i<candidate_objects.size(); i++)
		{
			float A = cv::contourArea(candidate_objects[i]);
			if(A >= minArea && A <= maxArea)
			{
				double p = cv::arcLength(candidate_objects[i], true);
				double circularity = (4*ucas::PI*A)/(p*p);
				
				if(circularity >= minCircularity)
				{
					float avgS = averageIntensity(hsv_channels[1], candidate_objects[i]);
					if(avgS >= minSaturation)
					{
						float avgV = averageIntensity(hsv_channels[2], candidate_objects[i]);
						if(avgV >= minValue)
						{
							detected_objects.push_back(i);
							//cv::drawContours(result, candidate_objects, i, cv::Scalar(0, 255, 255), 2, CV_AA);
				
							/*cv::putText(frame_copy, ucas::strprintf("C = %.2f", circularity), candidate_objects[i][0], 
							1, 1, cv::Scalar(0, 255, 255), 1, CV_AA);
							cv::putText(frame_copy, ucas::strprintf("S = %.0f", avgS), candidate_objects[i][0] + cv::Point(0, 15), 
								1, 1, cv::Scalar(0, 255, 255), 1, CV_AA);*/
						}
					}
				}
			}
		}

		// visual debugging of traffic light detection
		if(visual_debug && detected_objects.size() != candidate_objects.size())
		{

			for(int i=0; i<candidate_objects.size(); i++)
			{
				float A = cv::contourArea(candidate_objects[i]);
				if(A >= minArea && A <= maxArea)
				{
					double p = cv::arcLength(candidate_objects[i], true);
					double circularity = (4*ucas::PI*A)/(p*p);
					float avgS = averageIntensity(hsv_channels[1], candidate_objects[i]);
					float avgV = averageIntensity(hsv_channels[2], candidate_objects[i]);

					
					cv::Scalar color = cv::Scalar(0, 255, 255);
					if(circularity >= minCircularity && avgS >= minSaturation)
						color = cv::Scalar(255, 0, 0);

					cv::drawContours(result, candidate_objects, i, color, 2, CV_AA);

					cv::putText(result, ucas::strprintf("C = %.2f", circularity), candidate_objects[i][0] + cv::Point(-10, 0), 
						1, 1, color, 1, CV_AA);
					cv::putText(result, ucas::strprintf("S = %.0f", avgS), candidate_objects[i][0] + cv::Point(-10, 15), 
							1, 1, color, 1, CV_AA);
					cv::putText(result, ucas::strprintf("V = %.0f", avgV), candidate_objects[i][0] + cv::Point(-10, 30), 
						1, 1, color, 1, CV_AA);
					//cv::putText(result, ucas::strprintf("A = %.0f", A), candidate_objects[i][0] + cv::Point(-10, 45), 
					//	1, 1, color, 1, CV_AA);
				}
			}


			aia::imshow("Debugging image", result);
			aia::imshow("Binary image", bright_objects_img);

			//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/trafficlight_debug_visual.png", result);
			//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/trafficlight_debug_frame.png", frame);
			
		}

		// 2st step: traffic light recognition
		// criteria:
		// - hue ranges: 6 parameters, colors may fall outside all ranges --> higher chance of failure
		// - distances from 'pure' hues: 3 parameters, mutually exclusive --> lower  chance of failure
		for(auto & obj : detected_objects)
		{
			float avgHue = averageIntensity(hsv_channels[0], candidate_objects[obj]);

			float distance_from_red = std::min(avgHue-0, 180-avgHue);
			float distance_from_green = std::abs(avgHue-greenHue);
			float distance_from_orange = std::abs(avgHue-orangeHue);

			cv::Scalar color(0, 255, 255);
			std::string text = "SLOW";

			if(     distance_from_orange < distance_from_green && 
				    distance_from_orange < distance_from_red)
				;
			else if(distance_from_green < distance_from_orange && 
				    distance_from_green < distance_from_red)
			{
				color = cv::Scalar(0, 255, 0);
				text = "GO";
			}
			else
			{
				color = cv::Scalar(0, 0, 255);
				text = "STOP";
			}

			cv::drawContours(result, candidate_objects, obj, color, 3, CV_AA);
			cv::putText(result, text, candidate_objects[obj][0] + cv::Point(20, 20), 
				2, 1, color, 1.5, CV_AA);

		}

		return result;
	}
}

int main() 
{
	try
	{
		TLR::groundtruth = 1;
		aia::processVideoStream(std::string(EXAMPLE_IMAGES_PATH) + "/traffic_light.mp4",
			TLR::trafficLightRecognizer);

		return EXIT_SUCCESS;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}