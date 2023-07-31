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
	int sigmaGradX10;           // standard deviation of the gaussian smoothing applied as denoising prior to the calculation of the image derivative
								// 'X10' means it is multiplied by 10: unfortunately the OpenCV GUI does not support real-value trackbar, so we have to deal with integers
	int thresholdGrad;			// threshold applied on the gradient magnitude image normalized in [0, 255]

	// parameters of Marr-Hildreth/LoG edge detection
	int sigmaLoGX10;			// similar to 'sigmaGradX10'
	int thresholdLoG;			// zero-crossing threshold

	// parameters of Canny edge detection
	int sigmaCannyX10;			// similar to 'sigmaGradX10'
	int thresholdCanny;			// low threshold of the Canny's hysteresis step 



	// edge detection using gradient / first-order derivatives
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void edgeDetectionGrad(int, void*)
	{
		cv::Mat imgEdges;       // we will store here the binary image after edge detection to be displayed

		// if 'sigmaGradX10' is valid, we apply gaussian smoothing
		if(sigmaGradX10 > 0)
			cv::GaussianBlur(img, imgEdges, cv::Size(0,0), (sigmaGradX10/10.0), (sigmaGradX10/10.0));
		// otherwise we simply clone the image as it is
		else
			imgEdges = img.clone();

		// compute first-order derivatives along X and Y
		cv::Mat img_dx, img_dy;
		cv::Sobel(imgEdges, img_dx, CV_32F, 1, 0);
		cv::Sobel(imgEdges, img_dy, CV_32F, 0, 1);

		// compute gradient magnitude and angle (not used, but could also be used for thresholding)
		cv::Mat mag, angle;
		cv::cartToPolar(img_dx, img_dy, mag, angle);

		// generate a binary image from gradient magnitude 
		// by taking pixels whose gradient magnitude is higher than the specified threshold
		for(int y=0; y<imgEdges.rows; y++)
		{
			aia::uint8* imgEdgesYthRow = imgEdges.ptr<aia::uint8>(y); 
			float* magYthRow = mag.ptr<float>(y); 

			for(int x=0; x<imgEdges.cols; x++)
			{
				if(magYthRow[x] > thresholdGrad)
					imgEdgesYthRow[x] = 255;
				else
					imgEdgesYthRow[x] = 0;
			}
		}

		cv::imshow("Edge detection (gradient)", imgEdges);
	}


	// edge detection using Marr-Hildreth / LoG
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void edgeDetectionLoG(int, void*)
	{
		cv::Mat imgEdges;              // we will store here the binary image after edge detection to be displayed

		// calculate gaussian kernel size so that 99% of data are under the gaussian 
		int n = 6* (sigmaLoGX10/10.0);
		if(n % 2 == 0)
			n++;
		// if 'sigmaLoGX10' is valid, we apply gaussian smoothing
		if(sigmaLoGX10 > 0)
			cv::GaussianBlur(img, imgEdges, cv::Size(n,n), (sigmaLoGX10/10.0), (sigmaLoGX10/10.0));
		// otherwise we simply clone the image as it is
		else
			imgEdges = img.clone();

		// create 45° degrees-rotation-invariant Laplacian kernel
		cv::Mat laplacianKernel = (cv::Mat_<float>(3, 3) <<
			1,  1,   1,
			1,  -8,  1,
			1,  1,   1);

		// 'filter2D' is the convolution operation
		// in this case, it is the convolution of 'f' with the kernel 'laplacianKernel', 
		// and the result is stored into 'LoGresult' of type 'CV_32F' (real-valued image)
		cv::Mat LoGresult;
		cv::filter2D(imgEdges, LoGresult, CV_32F, laplacianKernel);

		// zero-crossing detection with 3x3 window (i.e. we detected positive/negative patterns along 4 directions)
		imgEdges.setTo(cv::Scalar(0));	// reset to 0s
		for(int y=1; y<imgEdges.rows-2; y++)
		{
			unsigned char* img_edges_curr_row = imgEdges.ptr<unsigned char>(y);

			float* LoG_curr_row = LoGresult.ptr<float>(y);
			float* LoG_prev_row = LoGresult.ptr<float>(y-1);
			float* LoG_next_row = LoGresult.ptr<float>(y+1);

			for(int x=1; x<imgEdges.cols-2; x++)
			{
				// get 8 neighbors
				float N		= LoG_prev_row[	x];
				float NE	= LoG_prev_row[	x+1];
				float E		= LoG_curr_row[	x+1];
				float SE	= LoG_next_row[	x+1];
				float S		= LoG_next_row[	x];
				float SO	= LoG_next_row[	x-1];
				float O		= LoG_curr_row[	x-1];
				float NO	= LoG_prev_row[	x-1];

				if( (N>0 && S<0 && fabs(N-S) >= thresholdLoG) ||
					(S>0 && N<0 && fabs(N-S) >= thresholdLoG) ||
					(E>0 && O<0 && fabs(E-O) >= thresholdLoG) ||
					(O>0 && E<0 && fabs(E-O) >= thresholdLoG) ||
					(NE>0 && SO<0 && fabs(NE-SO) >= thresholdLoG) ||
					(SO>0 && NE<0 && fabs(NE-SO) >= thresholdLoG) ||
					(NO>0 && SE<0 && fabs(SE-NO) >= thresholdLoG) ||
					(SE>0 && NO<0 && fabs(SE-NO) >= thresholdLoG))
					img_edges_curr_row[x] = 255;
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

		// calculate gaussian kernel size so that 99% of data are under the gaussian 
		int n = 6* (sigmaCannyX10/10.0);
		if(n % 2 == 0)
			n++;
		// if 'sigmaCannyX10' is valid, we apply gaussian smoothing
		if(sigmaCannyX10 > 0)
			cv::GaussianBlur(img, imgEdges, cv::Size(n,n), (sigmaCannyX10/10.0), (sigmaCannyX10/10.0));
		// otherwise we simply clone the image as it is
		else
			imgEdges = img.clone();

		// NOTE: OpenCV Canny function does not include gaussian smoothing: that's why we must did it before calling cv::Canny
		cv::Canny(imgEdges, imgEdges, thresholdCanny, 3*thresholdCanny);
		//                                            /\
		//                                            || suggested by Canny: 2 * low threshold <= high threshold <= 3 * low threshold

		cv::imshow("Edge detection (Canny)", imgEdges);
	}
}


int main() 
{
	try
	{
		// load an image
		aia::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/brain_ct.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!aia::img.data)
			throw aia::error("Cannot open image");

		// show the image
		cv::imshow("Image", aia::img);

		// set default parameters
		aia::sigmaGradX10 = aia::sigmaLoGX10 = aia::sigmaCannyX10 = 10;
		aia::thresholdGrad = aia::thresholdLoG = aia::thresholdCanny = 60;

		// create windows and insert the trackbars
		// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
		cv::namedWindow("Edge detection (gradient)");
		cv::namedWindow("Edge detection (LoG)");
		cv::namedWindow("Edge detection (Canny)");
		cv::createTrackbar("sigma(x10)", "Edge detection (gradient)", &aia::sigmaGradX10,   100, aia::edgeDetectionGrad);
		cv::createTrackbar("threshold",  "Edge detection (gradient)", &aia::thresholdGrad,  100, aia::edgeDetectionGrad);
		cv::createTrackbar("sigma(x10)", "Edge detection (LoG)",      &aia::sigmaLoGX10,    100, aia::edgeDetectionLoG);
		cv::createTrackbar("threshold",  "Edge detection (LoG)",      &aia::thresholdLoG,   100, aia::edgeDetectionLoG);
		cv::createTrackbar("sigma(x10)", "Edge detection (Canny)",    &aia::sigmaCannyX10,  100, aia::edgeDetectionCanny);
		cv::createTrackbar("tLOW",       "Edge detection (Canny)",    &aia::thresholdCanny, 100, aia::edgeDetectionCanny);
		
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