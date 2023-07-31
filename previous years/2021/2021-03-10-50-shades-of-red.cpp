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

// GOAL: create 50 shades of red image
int main() 
{
	try
	{
		// create 8-bit grayscale image of size 500 x 500 pixels
		cv::Mat shades_img(500, 500, CV_8U);

		// create shades of gray
		for(int y=0; y<shades_img.rows; y++)
		{
			unsigned char* yRow = shades_img.ptr<unsigned char>(y);
			for(int x=0; x<shades_img.cols; x++)
				yRow[x] = ucas::round<float>( (y/float(shades_img.rows-1))*255 );
		}

		// all shades
		aia::imshow("All shades of gray", shades_img);

		// reduced shades
		reduceGrayLevels(shades_img, 50);
		aia::imshow("50 shades of gray", shades_img);

		// first attempt: use RGB (BGR in OpenCV)
		std::vector< cv::Mat > channels;
		cv::Mat empty_channel = shades_img.clone();
		empty_channel.setTo(cv::Scalar(0));
		channels.push_back(empty_channel);		// Blue  channel
		channels.push_back(empty_channel);		// Green channel
		channels.push_back(shades_img);			// Red   channel
		cv::Mat color_img;
		cv::merge(channels, color_img);
		aia::imshow("50 shades of red (BGR-based)", color_img);

		// second attempt: use HSV color space
		channels.clear();
		channels.push_back(empty_channel);		// Hue  channel
		cv::Mat full_channel = shades_img.clone();
		full_channel.setTo(cv::Scalar(255));
		channels.push_back(shades_img);			// Saturation  channel
		channels.push_back(full_channel);		// Value      channel
		cv::merge(channels, color_img);
		cv::cvtColor(color_img, color_img, cv::COLOR_HSV2BGR);
		aia::imshow("50 shades of red (HSV-based)", color_img);

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
