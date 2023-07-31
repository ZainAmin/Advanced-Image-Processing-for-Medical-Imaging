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

	bool median = true;					// whether we are applying median (true) or gaussian (false) filtering
	int salt_and_pepper_level = 0;		// level in [0,100] of salt-and-pepper noise
	int filter_size = 3;

	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	//       see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void denoiseSaltAndPepper(int pos, void* userdata) 
	{
		if(img.channels() != 1)
		{
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}
		if(filter_size % 2 == 0)
			return;

		cv::Mat img_noise = img.clone();
		for(int y=0; y<img_noise.rows; y++)
		{
			unsigned char* img_noise_row = img_noise.ptr<unsigned char>(y);
			for(int x=0; x<img_noise.cols; x++)
			{
				// rand()%2        is a random number in {0,1}
				// 255* (rand()%2) is a random number in {0,255}
				// rand()%101 is a random number in [0, 100]
				int r = rand()%101;
				if(r && r <= salt_and_pepper_level)
					img_noise_row[x] = 255* (rand()%2);
			}
		}

		// 
		cv::Mat img_denoised;
		if(eiid::median)
			cv::medianBlur(img_noise, img_denoised, filter_size);
		else
			cv::GaussianBlur(img_noise, img_denoised, cv::Size(filter_size, filter_size), 0, 0);

		cv::imshow("denoising", img_noise);
		cv::imshow("denoised", img_denoised);
	}
}


int main() 
{
	try
	{	
		// load the image
		std::string img_name = "lena.png";
		eiid::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_UNCHANGED);
		if(!eiid::img.data)
			throw ucas::Error("cannot load image");

		// enable / disable median filtering
		eiid::median = false;

		// create the window and insert the trackbar
		cv::namedWindow("denoising");
		cv::createTrackbar("salt_and_pepper", "denoising", &eiid::salt_and_pepper_level, 100, eiid::denoiseSaltAndPepper);
		cv::createTrackbar("filter_size", "denoising", &eiid::filter_size, 100, eiid::denoiseSaltAndPepper);

		// call function with default parameters so it is updated right after the app starts
		eiid::denoiseSaltAndPepper(0,0);

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