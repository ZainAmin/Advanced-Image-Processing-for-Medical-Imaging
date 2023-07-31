// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: show how the Hough transform can be used to detect lines
//       we also build a minimal Graphical User Interface (GUI) so that we can interactively select
//       the parameters and see the result on the image

// NOTE: the Hough transform works on binary images only. To generate a binary image suited for Hough,
//       we have to detect edges first. Here, this is done with image derivatives, but in general it
//       can also be done with other methods (e.g. Canny, Marr-Hildtreth, custom criteria, etc.)

// since we work with a GUI, we need parameters (and the images) to be stored in global variables
namespace aia
{
	// the images we need to keep in memory
	cv::Mat img;            // original image
	cv::Mat imgEdges;       // binary image after edge detection

	// parameters of edge detection
	int stdevX10;           // standard deviation of the gaussian smoothing applied as denoising prior to the calculation of the image derivative
	// 'X10' means it is multiplied by 10: unfortunately the OpenCV GUI does not support real-value trackbar, so we have to deal with integers
	int threshold;          // threshold applied on the gradient magnitude image normalized in [0, 255]
	int alpha0;             // filter on gradient orientation: only gradients whose orientation is between [alpha0,alpha1] are considered
	int alpha1;             // filter on gradient orientation: only gradients whose orientation is between [alpha0,alpha1] are considered

	// parameters of Hough line detection
	int drho;               // quantization along rho axis
	int dtheta;             // quantization along theta axis
	int accum;              // accumulation threshold
	int n;                  // if != 0, we take the 'n' most voted (highest accumulation) lines

	// edge detection using gradient / first-order derivatives
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void edgeDetectionGrad(int, void*)
	{
		// if 'stdevX10' is valid, we apply gaussian smoothing
		if(stdevX10 > 0)
			cv::GaussianBlur(img, imgEdges, cv::Size(0,0), (stdevX10/10.0), (stdevX10/10.0));
		// otherwise we simply clone the image as it is
		else
			imgEdges = img.clone();
		// NOTE: we store the result into 'imgEdges', that we will re-use after
		//       in this way we can avoid allocating multiple cv::Mat and make the processing faster

		// compute first-order derivatives along X and Y
		cv::Mat img_dx, img_dy;
		cv::Sobel(imgEdges, img_dx, CV_32F, 1, 0);
		cv::Sobel(imgEdges, img_dy, CV_32F, 0, 1);

		// compute gradient magnitude and angle
		cv::Mat mag, angle;
		cv::cartToPolar(img_dx, img_dy, mag, angle, true);

		// generate a binary image from gradient magnitude and angle matrices
		// how?
		// - take pixels whose gradient magnitude is higher than the specified threshold
		//   AND
		// - take pixels whose angle is within the specified range
		for(int y=0; y<imgEdges.rows; y++)
		{
			aia::uint8* imgEdgesYthRow = imgEdges.ptr<aia::uint8>(y); 
			float* magYthRow = mag.ptr<float>(y); 
			float* angleYthRow = angle.ptr<float>(y); 

			for(int x=0; x<imgEdges.cols; x++)
			{
				if(magYthRow[x] > threshold && (angleYthRow[x] >= alpha0 || angleYthRow[x] <= alpha1))
					imgEdgesYthRow[x] = 255;
				else
					imgEdgesYthRow[x] = 0;
			}
		}

		cv::imshow("Edge detection (gradient)", imgEdges);
	}


	// line detection using Hough transform
	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void Hough(int,void*)
	{
		// in case we have invalid parameters, we do nothing
		if(drho <= 0)
			return;
		if(dtheta <= 0)
			return; 
		if(accum <= 0)
			return;

		// Hough returns a vector of lines represented by (rho,theta) pairs
		// the vector is automatically sorted by decreasing accumulation scores
		// this means the first 'n' lines are the most voted
		std::vector<cv::Vec2f> lines;
		cv::HoughLines(imgEdges, lines, drho, dtheta/180.0, accum);

		// we draw the first 'n' lines
		cv::Mat img_copy = img.clone();
		for(int k=0; k< std::min(size_t(n), lines.size()); k++)
		{
			float rho   = lines[k][0];
			float theta = lines[k][1];

			if (theta < aia::PI/4. || theta > 3.*aia::PI/4.) 
			{ // ~vertical line

				// point of intersection of the line with first row
				cv::Point pt1(rho/cos(theta),0);        
				// point of intersection of the line with last row
				cv::Point pt2((rho-img_copy.rows*sin(theta))/cos(theta),img_copy.rows);
				// draw a white line
				cv::line( img_copy, pt1, pt2, cv::Scalar(0,0,255), 1); 
			} 
			else
			{ // ~horizontal line

				// point of intersection of the line with first column
				cv::Point pt1(0,rho/sin(theta));        
				// point of intersection of the line with last column
				cv::Point pt2(img_copy.cols,(rho-img_copy.cols*cos(theta))/sin(theta));
				// draw a white line
				cv::line( img_copy, pt1, pt2, cv::Scalar(0,0,255), 2, CV_AA); 
			}
		}

		cv::imshow("Line detection (Hough)", img_copy);
	}
}


int main() 
{
	try
	{
		// load an image where there are lines that can be detected
		aia::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		if(!aia::img.data)
			throw aia::error("Cannot open image");

		// set default parameters
		aia::stdevX10 = 10;
		aia::threshold = 60;
		aia::alpha0 = 0;
		aia::alpha1 = 360;
		aia::drho = 1;
		aia::dtheta = 1;
		aia::accum = 1;
		aia::n = 10;

		// create a window named 'Edge detection (gradient)' and insert the trackbars
		// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
		cv::namedWindow("Edge detection (gradient)");
		cv::createTrackbar("stdev(x10)", "Edge detection (gradient)", &aia::stdevX10, 100, aia::edgeDetectionGrad);
		cv::createTrackbar("threshold", "Edge detection (gradient)", &aia::threshold, 100, aia::edgeDetectionGrad);
		cv::createTrackbar("alpha0", "Edge detection (gradient)", &aia::alpha0, 360, aia::edgeDetectionGrad);
		cv::createTrackbar("alpha1", "Edge detection (gradient)", &aia::alpha1, 360, aia::edgeDetectionGrad);

		// create another window named 'Line detection (Hough)' and insert the trackbars
		cv::namedWindow("Line detection (Hough)");
		cv::createTrackbar("drho", "Line detection (Hough)", &aia::drho, 100, aia::Hough);
		cv::createTrackbar("dtheta", "Line detection (Hough)", &aia::dtheta, 100, aia::Hough);
		cv::createTrackbar("accum", "Line detection (Hough)", &aia::accum, 100, aia::Hough);
		cv::createTrackbar("n", "Line detection (Hough)", &aia::n, 50, aia::Hough);

		// run edge detection + Hough for the first time with default parameters
		aia::edgeDetectionGrad(1,0);
		aia::Hough(1,0);

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