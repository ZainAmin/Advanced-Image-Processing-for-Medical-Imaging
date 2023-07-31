// include aia and ucas utilities
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main() 
{
	try
	{
		// create a 1000x1000 sized matrix, 1 channel (grayscale), 16-bits, initialized to black (0)
		cv::Mat gray_img(1000, 1000, CV_16U, cv::Scalar(0));
		printf("Image created: dims = %d x %d, channels = %d, bitdepth = %d\n",
			gray_img.rows, gray_img.cols, gray_img.channels(), aia::bitdepth(gray_img.depth()));
		aia::imshow("A 16-bit grayscale image", gray_img, true, 0.5f);
		//                                       	 	/\    /\
		//                                        		||    || scale factor (useful for images too big to fit in the screen!)
		//                                       	 	|| wait for the user to press a key before closing the window

		// easy (slower) way to access image pixels
		ucas::Timer timer;
		// an image is a matrix of pixels: we need a double loop
		for(int y=0; y<gray_img.rows; y++)	// loop on rows
			for(int x=0; x<gray_img.cols; x++)	// loop on columns
				gray_img.at<unsigned short>(y,x) = unsigned short ( (y / float(gray_img.rows-1) ) * 65535);
		//                /\            /\                              /\
		//                ||            ||                              || 
		//                || direct access (read/write) pixel at (y,x)  ||
		//                || y = row index, x = column index            ||
		//                              ||                              || shades of gray along the vertical axis (yshade):
		//                              ||                              || y is in [0, n_rows - 1]
		//                              ||                              || ynorm = y / (n_rows-1) is in [0,1]
		//                              ||                              || yshade = ynorm * 65535 is in [0,65535]
		//                              || we need to specify the native type of the pixel
		//                              || 8-bit  grayscale images --> unsigned char         or aia::uint8
		//                              || 16-bit grayscale images --> unsigned short        or aia::uint16
		//                              || real-valued (single-precision) images --> float   or aia::real32
		//                              || real-valued (double-precision) images --> double  or aia::real64
		//                              || for multi-channel images, see ocv-access-image-pixels-color.cpp example
		printf("time elapsed = %f\n", timer.elapsed<float>()*1000);
		aia::imshow("Gray shades", gray_img, true, 0.3f);

		// hard (faster) way to access image pixels
		timer.restart();
		for(int y=0; y<gray_img.rows; y++)
		{
			unsigned short* ythRow = gray_img.ptr<unsigned short>(y);
			//         /\                     /\ 
			//         ||                     || access to y-th row (which is an array), returns a memory address
			//         ||                     || access is 'typed' (we need to know the image type, like for at() )
			//         || since we are accessing dynamically-allocated memory, we need a pointer to store the address of the y-th row

			for(int x=0; x<gray_img.cols; x++)
				ythRow[x] = unsigned short ( (y / float(gray_img.rows-1) ) * 65535);
			//  access the x-th element of the y-th row, like we do for a traditional array
		}
		printf("time elapsed = %f\n", timer.elapsed<float>()*1000);
		aia::imshow("Gray shades", gray_img, true, 0.5f);

		

		// in the following, it is reported the code actually implemented in the class
		//int rows = 1000;
		//int cols = 5000;
		//cv::Mat grayImg(rows, cols, CV_16U, cv::Scalar(0));
		//
		//printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",
		//	grayImg.rows, grayImg.cols, grayImg.channels(), aia::bitdepth(grayImg.depth()));
		//aia::imshow("gray image (initialized)", grayImg);

		//// easy / slow access
		//ucas::Timer timer;
		//for(int y=0; y<grayImg.rows; y++)
		//{
		//	for(int x=0; x<grayImg.cols; x++)
		//		grayImg.at<unsigned short>(y,x) = ucas::round(65535 * (y/float(grayImg.rows-1)));
		//}

		//printf("\ntime elapsed (easy/slow access) = %f seconds\n", 
		//	timer.elapsed<float>());

		//// hard / fast access
		//timer.restart();
		//for(int y=0; y<grayImg.rows; y++)
		//{
		//	unsigned short* yThRow = grayImg.ptr<unsigned short>(y);
		//	for(int x=0; x<grayImg.cols; x++)
		//		yThRow[x] = ucas::round(65535 * (y/float(grayImg.rows-1)));
		//}

		//printf("\ntime elapsed (hard/fast access) = %f seconds\n", 
		//	timer.elapsed<float>());

		//aia::imshow("gray image (with shades)", grayImg);
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

