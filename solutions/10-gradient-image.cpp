// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace eiid
{
	cv::Mat gradientImage(const cv::Mat & frame)
	{
		cv::Mat processed_frame;

		// convert to gray: derivatives are not defined for color images
		cv::cvtColor(frame, processed_frame, cv::COLOR_BGR2GRAY);

		// compute x- and y-derivatives with Sobel
		// result *must* be stored in float to preserve negative
		// and out-of-range values
		cv::Mat dx,dy;
		cv::Sobel(processed_frame, dx, CV_32F, 1, 0);
		cv::Sobel(processed_frame, dy, CV_32F, 0, 1);

		// we need a float image to calculate the magnitude
		// (large values could be present)
		processed_frame.convertTo(processed_frame, CV_32F);
		cv::magnitude(dx, dy, processed_frame);

		// now we can normalize in [0,255] and convert back to 8U
		cv::normalize(processed_frame, processed_frame, 0, 255, cv::NORM_MINMAX);
		processed_frame.convertTo(processed_frame, CV_8U);

		// optionally, we might want to enhance the result
		// by multiplying the image with a given factor
		processed_frame *= 3;

		return processed_frame;
	}
}


int main() 
{
	try
	{	
		// the first input is the video source (empty = webcam)
		// the second input is the frame-by-frame processing function
		aia::processVideoStream("", eiid::gradientImage);

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

