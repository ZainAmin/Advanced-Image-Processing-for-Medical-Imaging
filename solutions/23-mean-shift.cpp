// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace aia
{
	// frame-by-frame cartoonification function declaration ( see definition after the main() )
	cv::Mat cartoonify(const cv::Mat & frame) throw (aia::error);
}

// GOAL: Mean-Shift filtering
int main() 
{
	try
	{
		// load Lena image and apply Mean-Shift filtering with (hs, hr) = (10, 30)
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png");
		if(!img.data)
			throw aia::error("Cannot open image");
		aia::imshow("Lena image", img, true, 1.0f);
		cv::Mat img_ms;
		cv::pyrMeanShiftFiltering(img, img_ms, 10, 30, 0);
		aia::imshow("Lena image + Mean-Shift", img_ms, true, 1.0f);


		// load pills image and apply Mean-Shift filtering with (hs, hr) = (10, 30)
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pills2.png");
		if(!img.data)
			throw aia::error("Cannot open image");
		aia::imshow("Pills image", img, true, 3.0f);
		cv::pyrMeanShiftFiltering(img, img_ms, 10, 30, 0);
		aia::imshow("Pills image + Mean-Shift", img_ms, true, 3.0f);

		// video cartoonification
		aia::processVideoStream("", aia::cartoonify);

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


cv::Mat aia::cartoonify(const cv::Mat & frame) throw (aia::error)
{
	// first we need to convert the color image (Blue, Green, Red) to grayscale
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

	// calculate first-order derivatives through convolution with Sobel filters
	cv::Mat dx,dy;
	cv::Sobel(frame_gray, dx, CV_32F, 1, 0);	// Sobel filter that yields x-derivative
	cv::Sobel(frame_gray, dy, CV_32F, 0, 1);	// Sobel filter that yields y-derivative
	//                          /\
	//                          || derivatives can yield negative values, better to store in float

	// calculate magnitude of gradient G = (dx dy)'
	cv::Mat mag;
	cv::magnitude(dx, dy, mag);

	// 'mag' is not meant for direct visualization
	// we need to normalize its histogram so that it can fit within the visualization range [0, 255]
	double minV, maxV;
	cv::minMaxLoc(mag, &minV, &maxV);	// calculate minimum and maximum intensities
	mag = mag - minV;					// shift whole image histogram to the right, by minV
	mag = mag * (255.0)/(maxV-minV);	// normalize whole histogram in the [0,255] range
	mag.convertTo(mag, CV_8U);			// convert from 32F ( = float) to 8U type ( = unsigned char )
	mag = mag*3;						// edge amplification / strengthening by a factor of '3'

	// we now want to display strong edges (the ones with higher values) as dark contours 
	// displayed on top of the color image after rasterization with Mean-Shift
	cv::Mat img = frame.clone();
	cv::pyrMeanShiftFiltering(img, img, 10, 30, 1);
	for(int y=0; y<img.rows; y++)
	{
		unsigned char* imgRow = img.ptr<unsigned char>(y);
		unsigned char* magRow = mag.ptr<unsigned char>(y);

		for(int x=0; x<img.cols; x++)
		{
			// if pixel(y,x) has a high gradient magnitude
			if(magRow[x] > 80)
			{
				// set color image pixel to (0,0,0) = black
				imgRow[3*x + 0] = 0;  // B
				imgRow[3*x + 1] = 0;  // G
				imgRow[3*x + 2] = 0;  // R
			}
		}
	}

	return img;
}