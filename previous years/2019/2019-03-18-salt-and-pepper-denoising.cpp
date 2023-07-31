// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	// since we work with a GUI, one possible solution to pass parameters
	// to/from the GUI/application core functions is to store parameters 
	// (including images) in global variables
	cv::Mat img;
	bool median = true;				// whether we are applying median (true) or gaussian (false) filtering
	int salt_pepper_perc;
	int filter_size;

	// callback function
	void denoiseSaltAndPepper(int pos, void* userdata) 
	{
		// clone image
		cv::Mat img_copy = img.clone();

		// add salt-and-pepper noise
		for(int y=0; y<img.rows; y++)
		{
			unsigned char* yThRow = img_copy.ptr<unsigned char>(y);
			for(int x=0; x<img.cols; x++)
				if(1+rand()%101 <= salt_pepper_perc)
					yThRow[x] = (rand()%2)*255;
		}

		cv::Mat img_denoised;
		if(filter_size % 2 != 0 && filter_size >= 3)
		{
			if(median)
				cv::medianBlur(img_copy, img_denoised, filter_size);
			else
				cv::GaussianBlur(img_copy, img_denoised, cv::Size(filter_size, filter_size), 0, 0);

			cv::imshow("denoised", img_denoised);
		}
		cv::imshow("denoising", img_copy);
	}
}


int main() 
{
	try
	{	
		// load the image in grayscale
		std::string img_name = "lena.png";
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw ucas::Error("cannot load image");

		// enable / disable median filtering
		median = true;

		// create the window and insert the trackbar(s)
		cv::namedWindow("denoising");
		salt_pepper_perc = 0;
		filter_size = 3;
		cv::createTrackbar("salt and pepper", "denoising", &salt_pepper_perc, 100, denoiseSaltAndPepper);
		cv::createTrackbar("filter size", "denoising", &filter_size, 150, denoiseSaltAndPepper);

		// call function with default parameters so it is updated right after the app starts
		denoiseSaltAndPepper(0,0);

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