// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// works in general for L >> 2
cv::Mat reduceGrayLevels(cv::Mat & img, int L)
{
	// number of original gray levels mapped for
	// each output gray level
	int step = 256/L;

	for(int y=0; y<img.rows; y++)
	{
		unsigned char* yRow = img.ptr<unsigned char>(y);
		for(int x=0; x<img.cols; x++)
			// integer division /step performs the many-to-one mapping
			// multiplication with step attempts to restore the original image dynamic
			yRow[x] = (yRow[x]/step)*step;
	}

	return img;
}

// advanced gray level reduction
// corrects pure black (0) and pure white (255) mapping
cv::Mat reduceGrayLevelsAdvanced(cv::Mat & img, int L)
{
	int step = 256/L;
	int max_quantized_value = 255/step;
	for(int y=0; y<img.rows; y++)
	{
		unsigned char* yRow = img.ptr<unsigned char>(y);
		for(int x=0; x<img.cols; x++)
			yRow[x] = ((yRow[x] / step) * 255) / max_quantized_value;
	}

	return img;
}

// GOAL 1: create an image with shades of gray along vertical (y) axis
// GOAL 2: reduce original gray levels down to 50
int main() 
{
	try
	{
		// create 8-bit grayscale image of size 500 x 500 pixels
		cv::Mat img(500, 500, CV_8U);

		// access pixels, method 1 (easy, slow)
		ucas::Timer timer;
		for(int y=0; y<img.rows; y++)
			for(int x=0; x<img.cols; x++)
				img.at<unsigned char>(y, x) = ucas::round<float>( (y/float(img.rows-1))*255 );
		printf("EASY(slow) METHOD processing time = %.3f\n", timer.elapsed<float>());
		

		// access pixels, method 2 (hard, fast)
		timer.restart();
		for(int y=0; y<img.rows; y++)
		{
			unsigned char* yRow = img.ptr<unsigned char>(y);
			for(int x=0; x<img.cols; x++)
				yRow[x] = ucas::round<float>( (y/float(img.rows-1))*255 );
		}
		printf("HARD(fast) METHOD processing time = %.3f\n", timer.elapsed<float>());

		// all shades
		//cv::imwrite("Z:/Users/Administrator/Desktop/all_gray.png", img);
		aia::imshow("All shades of gray", img);

		// reduced shades
		reduceGrayLevels(img, 50);
		//cv::imwrite("Z:/Users/Administrator/Desktop/reduced_gray.png", img);
		aia::imshow("50 shades of gray", img);

		return EXIT_SUCCESS;
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
