// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv image processing module
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

// include file I/O library
#include <fstream>

namespace
{
	cv::Mat cornersDetection(const cv::Mat & frame)
	{
		cv::Mat gray_frame;
		cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

		std::vector <cv::Point> corners;
		cv::goodFeaturesToTrack(gray_frame, corners, 50, 0.01, 20, cv::noArray(), 3, true, 0.04 );
		
		cv::Mat frame_cpy = frame.clone();
		for(auto & c : corners)
			cv::circle(frame_cpy, c, 5, cv::Scalar(0,255,255), CV_FILLED, CV_AA);
			
		return frame_cpy;
		

		/*cv::Mat raw_harris;

		cv::cornerHarris(gray_frame, raw_harris, 17, 3, 0.04);
		cv::normalize(raw_harris, raw_harris, 0, 255, cv::NORM_MINMAX);
		raw_harris.convertTo(raw_harris, CV_8U);

		return raw_harris;*/

	}
}

int main() 
{
	try
	{	
		
		//aia::processVideoStream(std::string(EXAMPLE_IMAGES_PATH) + "/traffic.avi", cornersDetection, "", true, 500);
		aia::processVideoStream("", cornersDetection, "", true);


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
