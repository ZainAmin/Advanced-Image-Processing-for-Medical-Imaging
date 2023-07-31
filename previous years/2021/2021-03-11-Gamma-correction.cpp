// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int percentile(std::vector<int> & histogram, int percVal)
{
	int sum = 0;
	for(auto & h : histogram)
		sum += h;

	int perc = 0;
	for(int acc = 0; ; perc++)
	{
		acc += histogram[perc];
		if(acc >= (percVal/100.0f)*sum)
			break;
	}

	return perc;
}

namespace
{
	cv::Mat input_img;
	std::string win_name;
	int gamma_x10 = 10;

	void linearStretchingCallback(int pos, void* userdata)
	{
		float gamma = gamma_x10/10.0f;

		cv::Mat enhanced_img = input_img.clone();
		int L = 256;
		float c = std::pow(L-1, 1-gamma);
		for(int y=0; y<enhanced_img.rows; y++)
		{
			unsigned char* yRow = enhanced_img.ptr<unsigned char>(y);
			for(int x=0; x<enhanced_img.cols; x++)
				yRow[x] = ucas::round<float>(c*std::pow(yRow[x], gamma));
		}

		cv::imshow(win_name, enhanced_img);
		cv::imshow("Original histogram", ucas::imhist(input_img));
		cv::imshow("Transformed histogram", ucas::imhist(enhanced_img));
	}
}

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		// load the image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		// if the image is too big, we can resize it
		cv::resize(input_img, input_img, cv::Size(0, 0), 0.5, 0.5, CV_INTER_AREA);

		// create the UI
		win_name = "Gamma correction";
		cv::namedWindow(win_name);
		cv::createTrackbar("gamma", win_name, &gamma_x10, 100, linearStretchingCallback);

		// start the UI
		linearStretchingCallback(0, 0);

		// waits for user to press a button and exit from the app
		cv::waitKey(0);

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
