// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "ucasImageUtils.h"

// include my project functions
#include "functions.h"

// calculates the given percentile from the given histogram and image size
// 'perc_val' is a real number in [0, 100]
int percentile(float perc_val, std::vector<int> & img_histo, int img_size)
{
	int perc_int = 0;	// intensity (to be computed) corresponding to perc_val
	int sum = 0;		// used to accumulate histogram values

	for(int i=0; i<img_histo.size(); i++)
	{
		sum += img_histo[i];
		if(sum >= (perc_val/100)*img_size)
		{
			perc_int = i;
			break;
		}
	}

	return perc_int;
}

// global variables and functions
// visible in this module only (unnamed namespace)
namespace
{
	cv::Mat img;				// image to be processed
	int pa, pb, c, d;			// parameters for piecewise linear stretching with N=3
								// pa and pb are left and right percentiles (x axis)
								// pc and pd are the bottom and upper percentiles (y axis)
	std::vector<int> histo[3];	// image histograms for BGR channels (computed once)
	std::vector<cv::Mat> img_channels;	// image channels

	// callback function
	// it is automatically called when the user interacts with the trackbar
	void piecewiseLinearStretching(int pos, void* userdata)
	{
		cv::Mat img_channels_copy[3];

		for(int chan=0; chan<3; chan++)
		{
			// define the 4 parameters of N=3 piecewise linear stretching
			// in this application, we calculate them from their corresponding percentile values
			float a = percentile(pa, histo[chan], img.rows*img.cols);
			float b = percentile(pb, histo[chan], img.rows*img.cols);
			float cf = (c/100.0)*255;
			float df = (d/100.0)*255;

			// copy image
			img_channels_copy[chan] = img_channels[chan].clone();

			// apply N=3 piecewise linear stretching
			for(int y=0; y<img.rows; y++)
			{
				unsigned char* ythRow = img_channels_copy[chan].ptr<unsigned char>(y);
				for(int x=0; x<img.cols; x++)
				{
					if(ythRow[x] < a)
						ythRow[x] = ucas::round(ythRow[x]*cf/a);
					else if(ythRow[x] > b)
						ythRow[x] = ucas::round((ythRow[x]-b)*(255-df)/(255-b))+df;
					else
						ythRow[x] = ucas::round((ythRow[x]-a)*(df-cf)/(b-a))+cf;
				}
			}
		}

		cv::Mat img_enhanced;
		cv::merge(img_channels_copy, 3, img_enhanced);
		cv::imshow("Image", img_enhanced);
	}
};

int main() 
{
	try
	{
		// which image to load?
		std::string imgPath = std::string(EXAMPLE_IMAGES_PATH) + "/retina_lowcontrast.bmp";

		// load image and check it has been loaded correctly
		img = cv::imread(imgPath, CV_LOAD_IMAGE_UNCHANGED);
		if(!img.data)
			throw ucas::Error("Cannot read image");

		// calculate histogram here, so we will not calculate it every
		// time the trackbar callback function is called
		cv::split(img, img_channels);
		histo[0] = ucas::histogram(img_channels[0]);	// remember OpenCV uses BGR
		histo[1] = ucas::histogram(img_channels[1]);
		histo[2] = ucas::histogram(img_channels[2]);

		// first create a window
		cv::namedWindow("Image");

		// create trackbars
		cv::createTrackbar("pa", "Image", &pa, 100, piecewiseLinearStretching);
		cv::createTrackbar("pb", "Image", &pb, 100, piecewiseLinearStretching);
		cv::createTrackbar("c", "Image", &c, 100, piecewiseLinearStretching);
		cv::createTrackbar("d", "Image", &d, 100, piecewiseLinearStretching);

		// launch application GUI
		piecewiseLinearStretching(0, 0);

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

