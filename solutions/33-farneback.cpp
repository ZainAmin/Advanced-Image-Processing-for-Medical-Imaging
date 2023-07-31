// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv image processing module
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace aia
{
	// global variables
	cv::Mat prev_img;							// previous image

	// single-frame processor
	cv::Mat calc_dense_flow(const cv::Mat & frame);
}


int main() 
{
	try
	{	
		// launch the system
		std::string path = std::string(EXAMPLE_IMAGES_PATH) + "/traffic1.avi";
		//std::string path = "";
		aia::processVideoStream(path, aia::calc_dense_flow, "", false);

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

// single-frame processor
cv::Mat aia::calc_dense_flow(const cv::Mat & frame)
{
	// prepare output image (we will draw the result on top if it)
	cv::Mat output = frame.clone();

	// convert frame to grayscale
	cv::Mat curr_img;
	cv::cvtColor(frame, curr_img, cv::COLOR_BGR2GRAY);

	// skip first frame (i.e. when no 'prev_img' has been set yet)
	if(prev_img.data)
	{
		// hide box in the bottom-right region (for traffic1.avi only)
		cv::rectangle(curr_img, cv::Rect(frame.cols-127, frame.rows-218, 127, 218), cv::Scalar(0), -1);

		// calculate Farneback flow
		cv::Mat flow_field(frame.rows, frame.cols, CV_32FC2);
		cv::calcOpticalFlowFarneback(prev_img, curr_img, flow_field, 0.5, 5, 31, 3, 5, 1.2, cv::OPTFLOW_FARNEBACK_GAUSSIAN);

		// smooth flow
		cv::GaussianBlur(flow_field, flow_field, cv::Size(11,11), 0, 0);

		// split velocity vector field into x and y components
		cv::Mat xy[2];
		cv::split(flow_field, xy);
		xy[1] = -xy[1];	// inverting y-axis because Farneback uses y pointing towards the top

		// calculate magnitude and angle of velocity vectors
		cv::Mat mag, angle;
		cv::cartToPolar(xy[0], xy[1], mag, angle, true);

		// normalize magnitude into range [0, 1]
		float minMag = 0;
		float maxMag = 100;
		mag = mag - minMag;
		mag = mag/(maxMag-minMag);

		// build hsv image
		bool by_value = false;
		cv::Mat _hsv[3], hsv;
		_hsv[0] = angle;
		_hsv[1] = by_value ? cv::Mat::ones(angle.size(), CV_32F) : mag;
		_hsv[2] = by_value ? mag : cv::Mat::ones(angle.size(), CV_32F);
		cv::merge(_hsv, 3, hsv);

		// convert to BGR
		cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
		hsv.convertTo(hsv, CV_8UC3, 255);

		output = hsv;
	}

	// update previous image with current image data
	prev_img = curr_img;

	return output;
}
