// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace eiid
{
	// since we work with a GUI, one possible solution to pass parameters
	// to/from the GUI/application core functions is to store parameters 
	// (including images) in global variables
	cv::Mat img;

	// the points define the piecewise linear stretching with N = 3
	int s1 = 0;
	int s2 = 255;
	int t1 = 0;
	int t2 = 255;

	// optionally (percentile_mode = true), s1 and s2 could be specified 
	// in terms of percentiles
	bool percentile_mode = false;
	int s1perc = 5;
	int s2perc = 95;

	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	//       see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void linearStretchingCallback(int pos, void* userdata) 
	{
		// precondition checks
		if(img.channels() != 1 && img.channels() != 3)
		{
			printf("image has %d channel, only 1- and 3-channel image are supported", img.channels());
			return;
		}
		if(img.depth() != CV_8U)
		{
			printf("unsupported pixel type (only 8-bit unsigned are supported)");
			return;
		}
		if(!img.data)
		{
			printf("image data empty");
			return;
		}

		// from now on, we operate on a clone of the original image
		// so that we can call this function multiple times with
		// different settings
		cv::Mat img_cpy = img.clone();

		// we apply contrast enhancement on the luminance channel only
		// if the image is grayscale, the luminance channel is the image itself
		// otherwise we have to extract is using an appropriate colorspace conversion
		cv::Mat img_lum;
		if(img_cpy.channels() == 3)
		{
			// here we use the Lab colorspace, but HSV is also a valid option
			cv::cvtColor(img_cpy, img_cpy, cv::COLOR_BGR2Lab);
			std::vector <cv::Mat> img_channels;
			cv::split(img_cpy, img_channels);

			img_lum = img_channels[0];
		}
		else
			img_lum = img_cpy;


		// in 'percentile mode', we determine s1 and s2 as
		// percentiles corresponding to s1perc and s2perc, respectively
		if(percentile_mode)
		{
			// the histogram is required to calculate percentiles...
			std::vector <int> histo = ucas::histogram(img_lum);

			// ...along with an accumulator
			int acc = 0;
			for(int i=0; i<histo.size(); i++)
			{
				acc += histo[i];
				if(acc >= s1perc *(img_lum.rows*img_lum.cols)/100.0f)
				{
					// we found the percentile value --> exit
					s1 = i;
					break;
				}
			}
			acc = 0;
			for(int i=0; i<histo.size(); i++)
			{
				acc += histo[i];
				if(acc >= s2perc *(img_lum.rows*img_lum.cols)/100.0f)
				{
					// we found the percentile value --> exit
					s2 = i;
					break;
				}
			}
		}

		// apply piecewise (N=3) linear stretching
		for(int y=0; y<img_lum.rows; y++)
		{
			unsigned char* data_row = img_lum.ptr<unsigned char>(y);
			for(int x=0; x<img_lum.cols; x++)
				if(data_row[x] <= s1)
					data_row[x] = float(data_row[x] - 0)  * (t1  - 0 )/(s1 -  0)  + 0;
				else if(data_row[x] > s1 && data_row[x] <= s2)
					data_row[x] = float(data_row[x] - s1) * (t2  - t1)/(s2 -  s1) + t1;
				else
					data_row[x] = float(data_row[x] - s2) * (255 - t2)/(255 - s2) + t2;
		}

		// if image is 3-channels, split img_cpy again and set the new luminance channel,
		// then merge and set the original colorspace back
		if(img_cpy.channels() == 3)
		{
			std::vector <cv::Mat> img_channels;
			cv::split(img_cpy, img_channels);
			img_channels[0] = img_lum;
			cv::merge(img_channels, img_cpy);
			cv::cvtColor(img_cpy, img_cpy, cv::COLOR_Lab2BGR);
		}

		// show enhanced image and the corresponding histogram
		cv::imshow("linear stretching", img_cpy);
		cv::imshow("histogram", ucas::imhist(img_lum));
	}
}


int main() 
{
	try
	{	
		// load the image
		// suggestion: try with
		// - lowcontrast.png (grayscale, 8 bits)
		// - retina_lowcontrast.bmp (color, 8 bits)
		// - moon.png (grayscale, 8 bits)
		std::string img_name = "retina_lowcontrast.bmp";
		eiid::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_UNCHANGED);
		if(!eiid::img.data)
			throw ucas::Error("cannot load image");

		// enable/disable percentile mode
		eiid::percentile_mode = true;

		// create the window and insert the trackbars
		cv::namedWindow("linear stretching");
		if(eiid::percentile_mode)
		{
			cv::createTrackbar("s1perc", "linear stretching", &eiid::s1perc, 100, eiid::linearStretchingCallback);
			cv::createTrackbar("s2perc", "linear stretching", &eiid::s2perc, 100, eiid::linearStretchingCallback);
		}
		else
		{
			cv::createTrackbar("s1", "linear stretching", &eiid::s1, 255, eiid::linearStretchingCallback);
			cv::createTrackbar("s2", "linear stretching", &eiid::s2, 255, eiid::linearStretchingCallback);
		}
		cv::createTrackbar("t1", "linear stretching", &eiid::t1, 255, eiid::linearStretchingCallback);
		cv::createTrackbar("t2", "linear stretching", &eiid::t2, 255, eiid::linearStretchingCallback);
		
		// call function with default parameters so it is updated right after the app starts
		eiid::linearStretchingCallback(0,0);
		
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

