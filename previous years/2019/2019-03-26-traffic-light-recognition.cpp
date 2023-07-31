// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace tlr
{
	// parameters
	float Amin = 100;		// minimum area threshold
	float Amax = 2000;		// maximum area threshold
	float Cmin = 0.65;		// circularity threshold
	float Smin = 120;		// average saturation threshold
	float Vmin = 200;		// average value threshold
	ucas::Timer timer;

	// for debugging purposes
	bool debug = true;
	int expected_lights = 1;

	// compute circularity of the given object
	float circularity(std::vector <cv::Point> & object)
	{
		float A = cv::contourArea(object);
		float p = cv::arcLength(object, true);
		return (4*ucas::PI*A) / (p*p);
	}

	// return the average of the given 8-bit intensity channel within the given object
	float averageWithinContour(cv::Mat & img_channel, std::vector <cv::Point> & object, double dist = 0 )
	{
		cv::Rect bbox = cv::boundingRect(object);
		float sum = 0;
		int num_pixels = 0;
		for(int y=bbox.y; y<bbox.y+bbox.height; y++)
		{
			unsigned char *yThRow = img_channel.ptr<unsigned char>(y);
			for(int x=bbox.x; x<bbox.x+bbox.width; x++)
				if(cv::pointPolygonTest(object, cv::Point2f(x,y), true) >= dist)
				{
					sum += yThRow[x];
					num_pixels++;
				}
		}
		return sum / num_pixels;
	}

	// return processed frame with highlighted traffic light
	cv::Mat trafficLightRecognizer(const cv::Mat& frame)
	{
		timer.restart();

		// convert color image to HSV color space
		cv::Mat frame_HSV;
		cv::cvtColor(frame, frame_HSV, cv::COLOR_BGR2HSV);
		std::vector <cv::Mat> hsv;
		cv::split(frame_HSV, hsv);

		// binarize in V
		cv::Mat frame_bin;
		cv::threshold(hsv[2], frame_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		cv::morphologyEx(frame_bin, frame_bin, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));

		// extract connected components (=objects)
		std::vector < std::vector <cv::Point> > objects;
		cv::findContours(frame_bin, objects, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// clone the original frame, so we can draw on top of the image
		cv::Mat processed_frame = frame.clone();

		// find the traffic light
		int detected_lights = 0;
		for(int k=0; k<objects.size(); k++)
		{
			float A = cv::contourArea(objects[k]);
			float C = circularity(objects[k]);

			// first decision criterion
			// based on minimum circularity and area
			if(C >= Cmin && A >= Amin && A <= Amax)
			{
				float avgS = averageWithinContour(hsv[1], objects[k]);
				float avgV = averageWithinContour(hsv[2], objects[k]);

				// second decision criterion
				// based on average saturation and value within contour
				if(avgS > Smin && avgV > Vmin)
				{
					// if we arrive here, we have detected a traffic light
					detected_lights++;

					// detect light color
					float avgH = averageWithinContour(hsv[0], objects[k], 3);

					float distance_to_yellow = std::abs(15-avgH);
					float distance_to_green = std::abs(65-avgH);
					float distance_to_red = std::min(std::abs(avgH), std::abs(120-avgH));

					cv::Scalar light_color;
					if(distance_to_yellow < distance_to_green &&
					   distance_to_yellow < distance_to_red)
						light_color = cv::Scalar(0, 255, 255);
					else if(distance_to_green < distance_to_yellow &&
						    distance_to_green < distance_to_red)
						light_color = cv::Scalar(0, 255, 0);
					else
						light_color = cv::Scalar(0, 0, 255);

					cv::drawContours(processed_frame, objects, k, light_color, 4, CV_AA);
				

					cv::Rect bbox = cv::boundingRect(objects[k]);
					cv::putText(processed_frame, 
						ucas::strprintf("H = %.2f", avgH),
						cv::Point(bbox.x, bbox.y), 1, 0.6, cv::Scalar(0, 255, 255), 1, CV_AA);

				}
			}
		}

		// debug when detected lights do not match with expected lights
		if(debug && detected_lights != expected_lights)
		{
			cv::Mat img_debug = frame.clone();

			// calculate and store all object features
			std::vector<bool> isDetectedLight;
			std::vector<int> object_indices;
			std::vector<float> As;
			std::vector<float> Cs;
			std::vector<float> Ss;
			std::vector<float> Vs;
			for(int k=0; k<objects.size(); k++)
			{
				// even if we are in debug mode, we still apply
				// the first decision criterion based on minimum area
				// so as to avoid too many objects to be processed
				float A = cv::contourArea(objects[k]);
				if(A >= Amin && A <= Amax)
				{
					// calculate and store object features
					float C = circularity(objects[k]);
					float avgS = averageWithinContour(hsv[1], objects[k]);
					float avgV = averageWithinContour(hsv[2], objects[k]);
					As.push_back(A);
					Cs.push_back(C);
					Ss.push_back(avgS);
					Vs.push_back(avgV);
					isDetectedLight.push_back(C >= Cmin && avgS >= Smin && avgV >= Vmin);
					object_indices.push_back(k);
				}
			}

			// draw contours first
			for(int i=0; i<object_indices.size(); i++)
				cv::drawContours(img_debug, objects, object_indices[i], isDetectedLight[i] ? cv::Scalar(255, 0, 0) : cv::Scalar(120, 120, 120), 2, CV_AA);

			// then draw text containing feature values on top of contours
			for(int i=0; i<object_indices.size(); i++)
			{
				cv::Rect bbox = cv::boundingRect(objects[object_indices[i]]);
				cv::putText(img_debug, 
					ucas::strprintf("A = %.0f, C = %.2f", As[i], Cs[i]),
					cv::Point(bbox.x, bbox.y), 1, 0.6, isDetectedLight[i] ? cv::Scalar(0, 255, 255) : cv::Scalar(120, 120, 120), 1, CV_AA);

				cv::putText(img_debug, 
					ucas::strprintf("S = %.0f, V = %.0f", Ss[i], Vs[i]),
					cv::Point(bbox.x, bbox.y+10), 1, 0.6, isDetectedLight[i] ? cv::Scalar(0, 255, 255) : cv::Scalar(120, 120, 120), 1, CV_AA);
			}

			ucas::imshow("Debug", img_debug, true);
		}

		printf("elapsed time = %.3f\n", timer.elapsed<float>());
		return processed_frame;
	}
}


int main() 
{
	try
	{	
		// configure debug
		tlr::debug = true;
		tlr::expected_lights = 1;

		// configure parameters
		tlr::Amin = 100;		// area threshold
		tlr::Amax = 2000;		// maximum area threshold
		tlr::Cmin = 0.65;		// circularity threshold
		tlr::Smin = 120;		// average saturation threshold
		tlr::Vmin = 180;		// average value threshold

		std::string video_name = "traffic_light.mp4";
		aia::processVideoStream(
			std::string(EXAMPLE_IMAGES_PATH) + "/" + video_name, 
			tlr::trafficLightRecognizer);

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