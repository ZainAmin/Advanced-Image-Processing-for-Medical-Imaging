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

	bool gamma_correction = false;		// whether we are applying gamma correction or logarithmic transform
	int gammaX100 = 100;				// gamma parameter x100 (since OpenCV trackbars support integers only)

	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	//       see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void nonlinearTransformationCallback(int pos, void* userdata) 
	{
		// precondition checks
		if(img.channels() != 1 && img.channels() != 3)
		{
			printf("image has %d channel, only 1- and 3-channel image are supported", img.channels());
			return;
		}
		if(img.depth() != CV_8U && img.depth() != CV_16U)
		{
			printf("unsupported pixel type (only 8- and 16-bit unsigned are supported)");
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

		// automatically detect the number of gray levels
		int L = std::pow(2, ucas::imdepth_detect(img_cpy));

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


		// logarithmic transform (8- and 16-bit versions)
		if(gamma_correction == false)
		{
			float c = (L-1)/log(L);
			if(img_lum.depth() == CV_8U)
			{
				for(int y=0; y<img_lum.rows; y++)
				{
					unsigned char* data_row = img_lum.ptr<unsigned char>(y);
					for(int x=0; x<img_lum.cols; x++)
						data_row[x] = c*log(1+data_row[x]);
				}
			}
			else
			{
				for(int y=0; y<img_lum.rows; y++)
				{
					unsigned short* data_row = img_lum.ptr<unsigned short>(y);
					for(int x=0; x<img_lum.cols; x++)
						data_row[x] = c*log(1+data_row[x]);
				}
			}
		}
		// gamma correction (8- and 16-bit versions)
		else
		{
			float c = std::pow(L-1, 1-gammaX100/100.0f);
			if(img_lum.depth() == CV_8U)
			{
				for(int y=0; y<img_lum.rows; y++)
				{
					unsigned char* data_row = img_lum.ptr<unsigned char>(y);
					for(int x=0; x<img_lum.cols; x++)
						data_row[x] = c*std::pow(data_row[x], gammaX100/100.0f);
				}
			}
			else
			{
				for(int y=0; y<img_lum.rows; y++)
				{
					unsigned short* data_row = img_lum.ptr<unsigned short>(y);
					for(int x=0; x<img_lum.cols; x++)
						data_row[x] = c*std::pow(data_row[x], gammaX100/100.0f);
				}
			}
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
		cv::imshow("nonlinear enhancement", img_cpy);
		cv::imshow("histogram", ucas::imhist(img_lum));
	}
}


int main() 
{
	try
	{	
		// load the image
		// suggestion: try with
		// - lowcontrast.png (grayscale, 8 bits) --> poor result expected (low contrast)
		// - retina_lowcontrast.bmp (color, 8 bits)
		// - raw_mammogram.tif (grayscale, 16 bits stored / 14 bits used)
		// - moon.png (grayscale, 8 bits) 
		std::string img_name = "raw_mammogram.tif";
		eiid::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_UNCHANGED);
		if(!eiid::img.data)
			throw ucas::Error("cannot load image");

		// enable / disable gamma correction
		eiid::gamma_correction = false;

		// if you are loading the mammogram (too big to fit the screen)
		// you might want to:
		if(img_name == "raw_mammogram.tif")
		{
			// 1) resize it to 20% of its original size
			cv::resize(eiid::img, eiid::img, cv::Size(0,0), 0.2, 0.2);
			// 2) rescale from 14-bits to 16-bits
			cv::normalize(eiid::img, eiid::img, 0, 65535, cv::NORM_MINMAX);
		}

		// create the window and insert the trackbar
		cv::namedWindow("nonlinear enhancement");
		cv::createTrackbar("gamma", "nonlinear enhancement", &eiid::gammaX100, 200, eiid::nonlinearTransformationCallback);
		
		// call function with default parameters so it is updated right after the app starts
		eiid::nonlinearTransformationCallback(0,0);
		
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

