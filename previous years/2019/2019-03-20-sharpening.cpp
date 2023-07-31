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
	bool laplacian = true;				// whether we are applying laplacian (true) unsharp masking (false)
	int k_x10;

	// callback function
	void sharpen(int pos, void* userdata) 
	{
		float k = k_x10/10.0f;
		cv::Mat sharpened;

		if(laplacian)
		{
			// define sharpening kernel based on laplacian operator
			cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
				-k,    -k,  -k,
				-k, 1+8*k,  -k,
				-k,    -k,  -k);

			
			cv::filter2D(img, sharpened, CV_8U, kernel);
			//cv::normalize(sharpened, sharpened, 0, 255, cv::NORM_MINMAX);
			//sharpened.convertTo(sharpened, CV_8U);
		}
		else
		{
			cv::Mat low_frew_img;
			cv::GaussianBlur(img, low_frew_img, cv::Size(7,7), 0, 0);
			sharpened = img + k*(img - low_frew_img);
		}

		cv::imshow("sharpening", sharpened);
	}
}


int main() 
{
	try
	{	
		// load the image in grayscale
		std::string img_name = "eye.blurry.png";
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_UNCHANGED);
		if(!img.data)
			throw ucas::Error("cannot load image");

		// enable / disable median filtering
		laplacian = false;

		// create the window and insert the trackbar(s)
		cv::namedWindow("sharpening");
		k_x10 = 0;
		cv::createTrackbar("sharpening factor", "sharpening", &k_x10, 100, sharpen);
		
		// call function with default parameters so it is updated right after the app starts
		sharpen(0,0);

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