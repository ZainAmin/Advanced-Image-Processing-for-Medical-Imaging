// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace eiid
{
	// for debugging purposes
	int expected_number_of_objects = 1;
	cv::Scalar contourColor = cv::Scalar(0,255,255);
	cv::Scalar fontColor = cv::Scalar(255,0,255);
	double fontScale = 0.8;
	int fontFace = 1;

	cv::Mat trafficLightRecognizer(const cv::Mat & frame)
	{
		cv::Mat processed_frame = frame.clone();

		// use a luminance channel to enhance the traffic lights
		// (for binarization purposes)
		cv::Mat frame_transformed;
		cv::cvtColor(frame, frame_transformed, cv::COLOR_BGR2HSV);
		std::vector <cv::Mat> frame_channels;
		cv::split(frame_transformed, frame_channels);

		// binarization with otsu
		cv::Mat binarized;
		cv::threshold(frame_channels[2], binarized, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		// connected component extraction
		std::vector < std::vector <cv::Point> > contours;
		cv::findContours(binarized, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		// for debug purposes only (=to be removed/commented at deploy time)
		std::vector <double> As (contours.size());	// area of each object
		std::vector <double> Cs (contours.size());	// circularity of each object
		std::vector <double> mSs (contours.size());	// mean saturation of each object
		std::vector <int> objects_found;			// objects that met the detection criteria

		// loop on all objects extracted (candidate objects)
		for(int i=0; i<contours.size(); i++)
		{
			// criteria (human-designed decision rules) to detect a traffic light
			double A = cv::contourArea(contours[i]);
			double p = cv::arcLength(contours[i], true);
			double C = 4*ucas::PI*A / (p*p);				// circularity

			// for debug purposes only (=to be removed/commented at deploy time)
			As[i] = A;
			Cs[i] = C;
			mSs[i] = -1;

			// FIRST (fast) decision rule based on
			// - circularity: we search round objects
			// - area: we ignore very small objects
			if(C > 0.7 && A > 100)
			{
				// calculate average saturation within the contour
				double saturation_sum = 0;	// to calculate the average saturation
				double hue_sum = 0;			// to calculate the average hue
				int count = 0;				// count pixels within the contour
				// we restrict the calculation to the pixels within the bounding box
				// --> faster than checking all pixels in the image!
				cv::Rect brect = cv::boundingRect(contours[i]);
				for(int x=brect.x; x<(brect.x+brect.width); x++)
					for(int y=brect.y; y<(brect.y+brect.height); y++)
						// this allows to detect if we are within the contour
						if( cv::pointPolygonTest(contours[i], cv::Point2f(x,y), false) > 0 )
						{
							saturation_sum += frame_channels[1].at<unsigned char>(y,x);
							hue_sum += frame_channels[0].at<unsigned char>(y,x);
							count++;
						}
				double mean_saturation = saturation_sum / count;
				double mean_hue = hue_sum / count;

				// for debug purposes only (=to be removed/commented at deploy time)
				mSs[i] = mean_saturation;

				// SECOND decision rule based on
				// - average saturation: traffic lights have (almost) pure colors
				if(mean_saturation > 120)
				{
					// for debug purposes only (=to be removed/commented at deploy time)
					objects_found.push_back(i);

					double distance_from_green  = std::abs(mean_hue - 60);
					double distance_from_orange = std::abs(mean_hue - 15);
					double distance_from_red    = std::min(std::abs(mean_hue - 0), std::abs(mean_hue-180));

					cv::Scalar light_color;
					if(distance_from_green < distance_from_orange && distance_from_green < distance_from_red)
						light_color = cv::Scalar(0, 255, 0);
					else if(distance_from_orange < distance_from_green && distance_from_orange < distance_from_red)
						light_color = cv::Scalar(0, 255, 255);
					else
						light_color = cv::Scalar(0, 0, 255);

					cv::drawContours(processed_frame, contours, i, light_color, 3, CV_AA);
					cv::putText(processed_frame, ucas::strprintf("A = %.0f, C = %.2f, S = %.0f, H = %.0f", A, C, mean_saturation, mean_hue), contours[i][0]-cv::Point(100,0), fontFace, fontScale, fontColor, 1, CV_AA);
				}
			}
		}

		// for debug purposes only (=to be removed/commented at deploy time)
		if(objects_found.size() != expected_number_of_objects)
		{
			// if we detected more objects than expected, we just draw all detected objects
			// with their features, so we can see what happened
			if(objects_found.size() > expected_number_of_objects)
			{
				for(auto & i : objects_found)
				{
					cv::drawContours(processed_frame, contours, i, contourColor, 1, CV_AA);
					cv::putText(processed_frame, ucas::strprintf("A = %.0f, C = %.2f, S = %.0f", As[i], Cs[i], mSs[i]), contours[i][0]-cv::Point(100,0), fontFace, fontScale, fontColor, 1, CV_AA);
				}
			}
			// if we detected less objects than expected, we need to draw all candidate objects
			// with their features, so we can see what happened
			else
			{
				for(int i=0; i<contours.size(); i++)
				{
					// perhaps we can at least filter candidate objects with a low area threshold
					// so we can avoid that really *all* candidate objects are drawn (messy image)
					if(As[i] > 50)
					{
						cv::drawContours(processed_frame, contours, i, contourColor, 1, CV_AA);
						cv::putText(processed_frame, ucas::strprintf("A = %.0f, C = %.2f, S = %.0f", As[i], Cs[i], mSs[i]), contours[i][0]-cv::Point(100,0), fontFace, fontScale, fontColor, 1, CV_AA);
					}
				}
			}
			ucas::imshow("bug!", processed_frame);
			cv::destroyWindow("bug!");
		}
		return processed_frame;
	}
}


int main() 
{
	try
	{	
		std::string video_name = "traffic_light_1.mp4";
		eiid::expected_number_of_objects = 1;

		// the first input is the video source (empty = webcam)
		// the second input is the frame-by-frame processing function
		aia::processVideoStream(std::string(EXAMPLE_IMAGES_PATH) + "/" + video_name, eiid::trafficLightRecognizer);

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

