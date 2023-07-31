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
	int perc1 = 5;
	int perc2 = 95;
	int t1 = 0;
	int t2 = 255;
	std::vector<int> hist;

	void linearStretchingCallback(int pos, void* userdata)
	{
		int s1 = percentile(hist, perc1);
		int s2 = percentile(hist, perc2);
		printf("%d%% percentile = %d\n", perc1, s1);
		printf("%d%% percentile = %d\n\n", perc2, s2);

		if(s2-s1 <= 0 || t2-t1 <= 0)
			return;

		cv::Mat enhanced_img = input_img.clone();
		for(int y=0; y<enhanced_img.rows; y++)
		{
			unsigned char* yRow = enhanced_img.ptr<unsigned char>(y);
			for(int x=0; x<enhanced_img.cols; x++)
			{
				if(yRow[x] <= s1)
					yRow[x] = float(yRow[x] - 0)  * (t1  - 0 )/(s1 -  0)  + 0;
				else if(yRow[x] > s1 && yRow[x] <= s2)
					yRow[x] = float(yRow[x] - s1) * (t2  - t1)/(s2 -  s1) + t1;
				else
					yRow[x] = float(yRow[x] - s2) * (255 - t2)/(255 - s2) + t2;
			}
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
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		// compute once for all the image histogram
		hist = ucas::histogram(input_img);

		// create the UI
		win_name = "Linear stretching";
		cv::namedWindow(win_name);
		cv::createTrackbar("s1", win_name, &perc1, 100, linearStretchingCallback);
		cv::createTrackbar("s2", win_name, &perc2, 100, linearStretchingCallback);
		cv::createTrackbar("t1", win_name, &t1, 255, linearStretchingCallback);
		cv::createTrackbar("t2", win_name, &t2, 255, linearStretchingCallback);

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
