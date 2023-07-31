// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// include opencv image processing module
#include <opencv2/imgproc/imgproc.hpp>

// GOAL: use Hu shape descriptors to build a basic shape recognition system
namespace smartshape
{
	// here we store pairs (shape name, shape contour), which represents the prior knowledge for this problem
	static std::map< std::string, std::vector < cv::Point> > shape_library;

	// system initialization: load all example shape images and extract their contours
	void init(std::string folderPath, std::string ext) throw (aia::error);

	// single-frame processor: takes a frame in input, and returns an image with the detected/recognized shapes
	cv::Mat shape_recognizer(const cv::Mat & frame);
}

int main() 
{
	try
	{	
		// initialize the system
		smartshape::init(std::string(EXAMPLE_IMAGES_PATH) + "/other shapes", ".bmp");
		printf("%d shapes loaded in the library\n", smartshape::shape_library.size());

		// launch the system on the video acquired in real-time from the camera
		aia::processVideoStream("", smartshape::shape_recognizer);

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


// system initialization: load all example shape images and extract their contours
void smartshape::init(std::string folderPath, std::string ext) throw (aia::error)
{
	// check folder exist
	if(!ucas::isDirectory(folderPath))
		throw aia::error(aia::strprintf("in smartshape::init(): cannot open folder at \"%s\"", folderPath.c_str()));

	// get all files within folder
	std::vector < std::string > files;
	cv::glob(folderPath, files);

	// process files
	for(auto & f : files)
	{
		// discard files that do not contain 'ext'
		if(f.find(ext) == std::string::npos)
			continue;

		// in order to extract the template shapes, we don't need the color information
		// hence, it's ok to load the image in grayscale
		cv::Mat img = cv::imread(f, CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error(aia::strprintf("Cannot read image \"%s\"", f.c_str()));

		// inverted thresholding because we assume objects are black on a white background
		cv::Mat binarized;
		cv::threshold(img, binarized, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
		
		// extract connected components / contours
		std::vector < std::vector <cv::Point> > contours;
		cv::findContours(binarized, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		// check there is only one contour
		if(contours.size() != 1)
			throw aia::error(aia::strprintf("more than one contours found in init(), image = %s", f.c_str()));

		// add (shape name, shape contour) to the library
		smartshape::shape_library[aia::getFileName(f, false)] = contours[0];	// 1 contour only --> it is in contours[0]
	}
}

// single-frame processor: takes a frame in input, and returns an image with the detected/recognized shapes
cv::Mat smartshape::shape_recognizer(const cv::Mat & frame)
{
	// where we will store the output/processed image
	cv::Mat output;

	// convert to gray (we need to binarize)
	cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);

	// binarize with Otsu
	cv::threshold(output, output, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

	// extract all contours (including internal ones, with CV_RETR_LIST)
	std::vector < std::vector <cv::Point> > contours;
	cv::findContours(output.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// copy frame into output (we want to draw stuff on top of the original image content)
	output = frame.clone();

	
	// if there are not contours --> return immediately (nothing to recognize)
	if(contours.empty())
		return output;

	// draw all contours with BLACK color (optional)
	//cv::drawContours(output, contours, -1, cv::Scalar(0, 0, 0), 1);

	// draw contours that meet all detection criteria
	for(int i=0; i<contours.size(); i++)
	{
		//double area = cv::contourArea(contours[i]);

		// discard small objects = noise (we do not consider them)
		//if(area < 1000)
			//continue;

		// discard too big objects (perhaps it is the whole image or the background)
		//if(area > 0.5 * frame.rows * frame.cols)
			//continue;

		// discard contours that do not contain the image center
		if(cv::pointPolygonTest(contours[i], cv::Point2f(frame.cols/2.0f, frame.rows/2.0f), false) < 0)
			continue;

		// find the shape from the shape library that better matches with the current contour
		// i.e. the one with the minimum distance
		double min_shape_distance = aia::inf<double>();
		std::string shape_name = "unknown";
		for(auto & el : smartshape::shape_library)
		{
			double shape_distance = cv::matchShapes(contours[i], el.second, CV_CONTOURS_MATCH_I1, 0);
			if(shape_distance < min_shape_distance)
			{
				min_shape_distance = shape_distance;
				shape_name = el.first;
			}
		}

		// discard object with poor matching (high distance)
		if(min_shape_distance > 0.5)
			continue;

		// draw current object contour (blue outline / yellow transparent filling)
		cv::drawContours(output, contours, i, cv::Scalar(0, 255, 255), -1, CV_AA);
		cv::addWeighted(output, 0.5, frame, 0.5, 0, output);
		cv::drawContours(output, contours, i, cv::Scalar(255, 0, 0), 2, CV_AA);

		// draw current object barycentre (red)
		cv::Moments moments = cv::moments(contours[i], true);
		cv::Point barycentre = cv::Point(moments.m10/moments.m00, moments.m01/moments.m00);
		cv::circle(output, barycentre, 5, cv::Scalar(0,0,255), CV_FILLED);
		
		// draw current object name (blue)
		cv::putText(output, shape_name, barycentre, 2, 1, cv::Scalar(255, 0, 0), 2, CV_AA);
	}

	// draw cross at the center
	cv::line(output, cv::Point(output.cols/2 - 5, output.rows/2), cv::Point(output.cols/2 + 5, output.rows/2), cv::Scalar(0,0,255), 2, CV_AA);
	cv::line(output, cv::Point(output.cols/2, output.rows/2 -5), cv::Point(output.cols/2, output.rows/2 + 5), cv::Scalar(0,0,255), 2, CV_AA);

	return output;
}
