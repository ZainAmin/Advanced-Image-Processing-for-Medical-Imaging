// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace
{
	cv::Mat img;
	std::string win_name = "Magic Wand";

	void magicWand(int event, int x, int y, int, void* userdata)
	{
		static cv::Point prevPos(0,0);
		static bool drawing = false;
		static cv::Mat markers(img.rows, img.cols, CV_32S, cv::Scalar(0));
		static cv::Mat img_copy = img.clone();
		static int n_strokes = 0;

		if(event == cv::EVENT_LBUTTONDOWN)
		{
			prevPos.x = x;
			prevPos.y = y;
			drawing = true;
			n_strokes++;
		}
		else if (event == cv::EVENT_LBUTTONUP)
		{
			drawing = false;

			cv::Mat watershed_result = markers.clone();
			cv::watershed(img, watershed_result);

			// -1*255 = -255 (dams)
			// (>0)*255 = 255 (regions)
			// + 255 --> 0 (dams) and 255 (regions)
			// invert --> 255 (dams) and 0 (regions)
			// extract contours --> dams
			cv::Mat dams;
			watershed_result.convertTo(dams, CV_8U, 255, 255);
			dams = 255 - dams;
			std::vector < std::vector <cv::Point> > contours;
			cv::findContours(dams, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
			cv::drawContours(img_copy, contours, -1, cv::Scalar(0, 255, 255), 2, CV_AA);

			watershed_result = cv::abs(watershed_result);
			cv::normalize(watershed_result, watershed_result, 0, 255, cv::NORM_MINMAX);
			watershed_result.convertTo(watershed_result, CV_8U);
			cv::imshow("Segmentation", watershed_result);
		}
		else if (event == cv::EVENT_MOUSEMOVE && drawing)
		{
			// display strokes on top of the image
			cv::line(img_copy, prevPos, cv::Point(x,y), cv::Scalar(255,0,0), 2, CV_AA);
			
			// one stroke = one marker
			cv::line(markers, prevPos, cv::Point(x,y), cv::Scalar(n_strokes));
			
			// previous pos update
			prevPos.x = x;
			prevPos.y = y;
		}

		cv::imshow(win_name, img_copy);
	}

}

// GOAL: Pectoral Muscle Segmentation with Mean-Shift
int main() 
{
	try
	{
		// load image
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg");
		if(!img.data)
			throw aia::error("Cannot open image");

		// launch GUI application
		cv::namedWindow(win_name);
		cv::setMouseCallback(win_name, magicWand);
		magicWand(0, 0, 0, 0, 0);
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
