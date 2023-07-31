// include aia and ucas utilities
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Paint
{
	// since we work with a GUI, one possible solution is to store parameters 
	// (including images) in global variables
	cv::Mat board;							// original image
	bool isLeftButtonPressed = false;		// flag = true when the left mouse button is kept pressed (pencil mode)
	bool isRightButtonPressed = false;		// flag = true when the right mouse button is kept pressed (eraser mode)

	// NOTE: this is a callback function we will link to the mouse event in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	//       see https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar#setmousecallback
	void updatePaint(int event, int x, int y, int, void* userdata)
	{
		// set flags according to event
		if(event == cv::EVENT_LBUTTONDOWN)
			isLeftButtonPressed = true;
		else if(event == cv::EVENT_LBUTTONUP)
			isLeftButtonPressed = false;
		else if(event == cv::EVENT_RBUTTONDOWN)
			isRightButtonPressed = true;
		else if(event == cv::EVENT_RBUTTONUP)
			isRightButtonPressed = false;

		// pencil mode
		if(isLeftButtonPressed)
			cv::circle(board, cv::Point(x,y), 20, cv::Scalar(255,0,0), CV_FILLED);
		// eraser mode
		else if(isRightButtonPressed)
			cv::circle(board, cv::Point(x,y), 20, cv::Scalar(255,255,255), CV_FILLED);

		// show the result
		cv::imshow("Paint", Paint::board);
	}
}


int main() 
{
	try
	{
		// initialize board with white color
		Paint::board = cv::Mat(500,500, CV_8UC(3), cv::Scalar(255, 255, 255));

		// create window and set mouse callback
		cv::namedWindow("Paint");
		cv::setMouseCallback("Paint", Paint::updatePaint);

		// show board
		cv::imshow("Paint", Paint::board);

		// wait for key press = windows stay opened until the user presses any key
		cv::waitKey(0);

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

