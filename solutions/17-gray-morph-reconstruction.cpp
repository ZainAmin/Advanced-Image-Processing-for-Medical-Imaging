// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: stars/galaxy discrimination in astronomical images
int main() 
{
	try
	{
		// load image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy.jpg");
		if(!img.data)
			throw aia::error("Cannot open image");
		aia::imshow("Image", img);

		// Morphological reconstruction builds on the definition of a marker image 'F'
		// containing the starting points (seeds), and a mask image 'G' which (usually) is
		// the original binary image

		// The marker image should satisfy the following conditions:
		// - it must be contained in the mask image 'G', i.e. F <= G
		// - it should contain seeds for only 'some' objects (those we want to reconstruct)
		//   and be zero (or low) for all pixels of the 'other' objects (those we are not interested to)

		// in this example, we are interested to stars
		// thus, opening the image with a circular SE of diameter slightly larger than the biggest star
		// will remove stars / shave the peaks
		cv::Mat marker;
		cv::morphologyEx(img, marker, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(30,30)));
		aia::imshow("Marker", marker);
		//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy_marker.jpg", marker);


		// Morphological reconstruction by conditional dilation:
		// iteratively dilate the marker 'F' under the constraint 'F' <= 'G'
		// ...until 'F' does not change anymore (reached stability = reconstruction ended)
		cv::Mat marker_prev;	// we need to store the previous marker in the iteration
		std::vector<cv::Mat> marker_split, marker_split_prec;	// and do the same for each channel since here we deal with a color image
		int it = 0;
		do
		{
			// make a backup copy of the previous marker
			marker_prev = marker.clone();

			// geodesic dilation ( = dilation + pointwise minimum with mask)
			cv::morphologyEx(marker, marker, CV_MOP_DILATE, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(5,5)));
			marker = cv::min(marker, img);

			// display reconstruction in progress
			printf("it = %d\n", ++it);
			//aia::imshow("marker", marker);

			// *** ONLY FOR COLOR MORPHOLOGICAL RECONSTRUCTION ***
			// split channels
			cv::split(marker, marker_split);
			cv::split(marker_prev, marker_split_prec);
		}
		while( cv::countNonZero(marker_split[0]-marker_split_prec[0]) > 0 ||
			   cv::countNonZero(marker_split[1]-marker_split_prec[1]) > 0 ||
			   cv::countNonZero(marker_split[2]-marker_split_prec[2]) > 0) ;
			 // check for each channel if there are any changes


		aia::imshow("Reconstructed", marker);
		aia::imshow("G - reconstructed", img-marker);
		//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy_rec.jpg", marker);
		//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy_stars.jpg", img-marker);

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

