// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: show how three different methods (Gradient, Marr-Hildreth/LoG, Canny) for edge detection work
//       we also build a minimal Graphical User Interface (GUI) so that we can interactively select
//       the parameters and see the result on the image

// since we work with a GUI, we need parameters (and the images) to be stored in global variables
namespace aia
{
	// the images we need to keep in memory
	cv::Mat img;				// original image

	// parameters of gradient-based edge detection
	int thresholdGrad;

	// parameters of Marr-Hildreth/LoG edge detection
	int sigmaLoGx10;
	int thresholdLoG;

	// parameters of Canny edge detection
	int sigmaCannyx10;
	int thresholdCannyL;

	// edge detection using gradient / first-order derivatives
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void edgeDetectionGrad(int, void*)
	{
		cv::Mat imgEdges;       // we will store here the binary image after edge detection to be displayed
		
		cv::Mat dx, dy;
		cv::Sobel(img, dx, CV_32F, 1, 0);
		cv::Sobel(img, dy, CV_32F, 0, 1);
		cv::Mat grad_mag;
		cv::magnitude(dx, dy, grad_mag);
	
		//cv::normalize(grad_mag, grad_mag, 0, 255, cv::NORM_MINMAX);
		//grad_mag.convertTo(grad_mag, CV_8U);

		double min_val, max_val;
		cv::minMaxLoc(grad_mag, &min_val, &max_val);

		cv::threshold(grad_mag, grad_mag, thresholdGrad*0.01*max_val, 255, CV_THRESH_BINARY);
		grad_mag.convertTo(grad_mag, CV_8U);

		//printf("thresholdGrad = %d\n", thresholdGrad);
		cv::imshow("Edge detection (gradient)", grad_mag);
	}


	// edge detection using Marr-Hildreth / LoG
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void edgeDetectionLoG(int, void*)
	{
		cv::Mat imgEdges;              // we will store here the binary image after edge detection to be displayed

		// first step: gaussian filtering
		cv::Mat blurred;
		cv::GaussianBlur(img, blurred, cv::Size(0,0), sigmaLoGx10/10.0, sigmaLoGx10/10.0);

		// second step: laplacian filtering
		cv::Mat laplacian = (cv::Mat_<float>(3, 3) <<
			1,  1,  1,
			1, -8,  1,
			1,  1,  1);
		cv::Mat LoG_output;
		cv::filter2D(blurred, LoG_output, CV_32F, laplacian);

		// third step: zero-crossing detection
		double min_val, max_val;
		cv::minMaxLoc(LoG_output, &min_val, &max_val);
		imgEdges = img.clone();
		imgEdges.setTo(cv::Scalar(0));
		float T = 0.01*max_val*thresholdLoG;
		for(int y=1; y<LoG_output.rows-1; y++)
		{
			float *prevRow =  LoG_output.ptr<float>(y-1);
			float *currRow =  LoG_output.ptr<float>(y);
			float *nextRow =  LoG_output.ptr<float>(y+1);

			unsigned char *imgEdgeData = imgEdges.ptr<unsigned char>(y);
			for(int x=1; x<LoG_output.cols-1; x++)
			{
				float N  = prevRow[x];
				float NE = prevRow[x+1];
				float E  = currRow[x+1];
				float SE = nextRow[x+1];
				float S  = nextRow[x];
				float SW = nextRow[x-1];
				float W  = currRow[x-1];
				float NW = prevRow[x-1];

				if( fabs(N-S) > T ||
					fabs(E-W) > T ||
					fabs(NW-SE) > T ||
					fabs(NE-SW) > T)
					imgEdgeData[x] = 255;
			}
		}

		cv::imshow("Edge detection (LoG)", imgEdges);
	}


	// edge detection using Canny
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void edgeDetectionCanny(int, void*)
	{
		cv::Mat imgEdges;              // we will store here the binary image after edge detection to be displayed

		// first step: gaussian filtering
		cv::Mat blurred;
		cv::GaussianBlur(img, blurred, cv::Size(0,0), sigmaCannyx10/10.0, sigmaCannyx10/10.0);


		cv::Canny(blurred, imgEdges, thresholdCannyL, 3*thresholdCannyL);

		
		cv::imshow("Edge detection (Canny)", imgEdges);
	}
}


int main() 
{
	try
	{
		// load an image
		aia::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!aia::img.data)
			throw aia::error("Cannot open image");

		// show the image
		cv::imshow("Image", aia::img);

		// set default parameters
		aia::thresholdGrad = 33;
		aia::thresholdLoG = 4;
		aia::sigmaLoGx10 = 10;
		aia::sigmaCannyx10 = 10;
		aia::thresholdCannyL = 100;

		// create windows and insert the trackbars
		// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
		cv::namedWindow("Edge detection (gradient)");
		cv::namedWindow("Edge detection (LoG)");
		cv::namedWindow("Edge detection (Canny)");
		cv::createTrackbar("threshold",  "Edge detection (gradient)", &aia::thresholdGrad,  100, aia::edgeDetectionGrad);
		cv::createTrackbar("sigmaX10",  "Edge detection (LoG)", &aia::sigmaLoGx10,  100, aia::edgeDetectionLoG);
		cv::createTrackbar("threshold",  "Edge detection (LoG)", &aia::thresholdLoG,  100, aia::edgeDetectionLoG);
		cv::createTrackbar("sigmaX10",  "Edge detection (Canny)", &aia::sigmaCannyx10,  100, aia::edgeDetectionCanny);
		cv::createTrackbar("threshold",  "Edge detection (Canny)", &aia::thresholdCannyL,  100, aia::edgeDetectionCanny);

		// run edge detection for the first time with default parameters
		aia::edgeDetectionGrad(1,0);
		aia::edgeDetectionLoG(1,0);
		aia::edgeDetectionCanny(1,0);

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