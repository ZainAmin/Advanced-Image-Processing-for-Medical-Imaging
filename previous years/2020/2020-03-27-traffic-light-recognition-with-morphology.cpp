// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace tlr
{
	// parameters
	float minArea = 100;
	float maxArea = 3000;
	float minCircularity = 0.65;
	float minContourDist = 2;
	int minSaturation = 120;
	int minValue = 200;
	int expected_lights = 1;
	bool debug = false;

	// utility functions
	/*float circularity(const std::vector <cv::Point> & contour)
	{
		float p = cv::arcLength(contour, true);
		return 4*ucas::PI*cv::contourArea(contour)/(p*p);
	}*/
	float averageWithinContour(const cv::Mat & img, std::vector<cv::Point> & contour, float dist)
	{
		cv::Rect bbox = cv::boundingRect(contour);
		float sum = 0;
		int count = 0;
		for(int y=bbox.y; y<bbox.y+bbox.height; y++)
		{
			const unsigned char* row = img.ptr<unsigned char>(y);
			for(int x=bbox.x; x<bbox.x+bbox.width; x++)
				if(cv::pointPolygonTest(contour, cv::Point2f(x,y), true) >= dist)
				{
					sum += row[x];
					count++;
				}
		}
		return sum/count;
	}

	// the tlr function (frame-by-frame processing)
	cv::Mat iTrafficLight(const cv::Mat & frame)
	{
		// convert to HSV space for binarization in V and light color detection in HS
		cv::Mat frame_hsv;
		cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
		cv::Mat hsv[3];
		cv::split(frame_hsv, hsv);

		// binarization in V
		cv::Mat v_bin;
		cv::threshold(hsv[2], v_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		// eliminate protrusions
		cv::morphologyEx(v_bin, v_bin, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));

		// extract all bright objects in the image
		std::vector < std::vector <cv::Point> > objects;
		cv::findContours(v_bin, objects, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		std::vector<float> As(objects.size()), Cs(objects.size()), Ss(objects.size()), Vs(objects.size());

		// apply decision rules tree for each contour
		cv::Mat output = frame.clone();
		int n_detected_objects = 0;
		for(int k=0; k<objects.size(); k++)
		{
			// first level of decision tree 
			float A = cv::contourArea(objects[k]);
			As[k] = A;
			cv::Rect bbox = cv::boundingRect(objects[k]);
			if(A >= minArea && A <= maxArea)
			{
				// second level
				float p = cv::arcLength(objects[k], true);
				float C = 4*ucas::PI*A/(p*p);
				Cs[k] = C;
				if(C >= minCircularity)
				{
					// third level
					float avg_S = averageWithinContour(hsv[1], objects[k], minContourDist);
					Ss[k] = avg_S;
					if(avg_S >= minSaturation)
					{
						// fourth level
						float avg_V = averageWithinContour(hsv[2], objects[k], minContourDist);
						Vs[k] = avg_V;
						if(avg_V >= minValue)
						{
							float avg_H = averageWithinContour(hsv[0], objects[k], 2);
							cv::Scalar light_color;

							float dist_to_green  = std::abs(avg_H-60);
							float dist_to_yellow = std::abs(avg_H-15);
							float dist_to_red    = std::min(avg_H, 180-avg_H);

							if(dist_to_green < dist_to_yellow &&
							   dist_to_green < dist_to_red)
							   light_color = cv::Scalar(0, 255, 0);
							else if(dist_to_yellow < dist_to_red &&
								    dist_to_yellow < dist_to_green)
								light_color = cv::Scalar(0, 255, 255);
							else
								light_color = cv::Scalar(0, 0, 255);

							n_detected_objects++;
							cv::drawContours(output, objects, k, light_color, 2, CV_AA);
							cv::putText(output, ucas::strprintf("A = %.0f", A), bbox.tl() + cv::Point(10, 0), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
							cv::putText(output, ucas::strprintf("C = %.2f", C), bbox.tl() + cv::Point(10, 10), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
							cv::putText(output, ucas::strprintf("S = %.0f", avg_S), bbox.tl() + cv::Point(10, 20), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
							cv::putText(output, ucas::strprintf("V = %.0f", avg_V), bbox.tl() + cv::Point(10, 30), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
							cv::putText(output, ucas::strprintf("H = %.0f", avg_H), bbox.tl() + cv::Point(10, 40), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
						}
					}
				}
			}
		}

		if(debug && n_detected_objects != expected_lights)
		{
			aia::imshow("Error", output);
			for(int k=0; k<objects.size(); k++)
			{
				if(As[k] >= minArea && As[k] <= maxArea)
				{
					cv::Rect bbox = cv::boundingRect(objects[k]);
					cv::drawContours(output, objects, k, cv::Scalar(0, 255, 255), 2, CV_AA);
					cv::putText(output, ucas::strprintf("A = %.0f", As[k]), bbox.tl() + cv::Point(10, 0), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
					cv::putText(output, ucas::strprintf("C = %.2f", Cs[k]), bbox.tl() + cv::Point(10, 10), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
					cv::putText(output, ucas::strprintf("S = %.0f", Ss[k]), bbox.tl() + cv::Point(10, 20), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
					cv::putText(output, ucas::strprintf("V = %.0f", Vs[k]), bbox.tl() + cv::Point(10, 30), 2, 0.3, cv::Scalar(0,255,255), 1, CV_AA);
				}
			}
			aia::imshow("Debug", output);
			cv::destroyWindow("Debug");
			cv::destroyWindow("Error");
		}

		

		return output;
	}
}


int main() 
{
	try
	{
		tlr::debug = 1;

		tlr::expected_lights = 1;

		aia::processVideoStream(std::string(EXAMPLE_IMAGES_PATH) + "/traffic_light.mp4", tlr::iTrafficLight);
		//aia::processVideoStream(std::string(EXAMPLE_IMAGES_PATH) + "/traffic_light_2.mp4", tlr::iTrafficLight);


		return 1;
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

