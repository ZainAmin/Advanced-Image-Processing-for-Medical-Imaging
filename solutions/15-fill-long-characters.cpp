// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main() 
{
	try
	{	
		// load image containing text
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/text.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot load image");
		aia::imshow("Image", img);

		// binarize with Otsu
		cv::Mat img_bin;
		cv::threshold(img, img_bin, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY_INV);
		//                                                           /\
		//                                                           || _INV (INVERSE) because in OpenCV foreground is 'white', but here text is 'black'
		ucas::imshow("Image binarized", img_bin);

		// STEP 1: MORPHOLOGICAL RECONSTRUCTION / (in this case also known as ) OPENING BY RECONSTRUCTION
		// Morphological reconstruction builds on the definition of a marker image 'F'
		// containing the starting points (seeds), and a mask image 'G' which (usually) is
		// the original binary image

		// The marker image should satisfy the following conditions:
		// - it must be a binary image contained in the mask image 'G', i.e. F ? G
		// - it should contain seeds for only 'some' objects (those we want to reconstruct)
		//   and be = 0 for all pixels of the 'other' objects (those we are not interested to.

		// in this example, we are interested to 'vertical' characters
		// thus, eroding the image with a 'vertical' structuring element (SE) 3x24 will:
		// - erase the parts of the foreground objects where the SE does *not* fit
		//   ...this will completely erase characters such as 'a', 'c', 'e', etc.
		// - erode the parts of the foreground objects where the SE does fit
		//   ...this will leave some seed areas within characters such as 'l', 'b', 'k', etc.
		cv::Mat marker;
		cv::erode(img_bin, marker, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(1, 24)));
		aia::imshow("Reconstruction (marker)", marker);

		// Morphological reconstruction by conditional dilation:
		// iteratively dilate the marker 'F' under the constraint 'F' ? 'G'
		// ...until 'F' does not change anymore (reached stability = reconstruction ended)
		cv::Mat marker_prev;	
		do 
		{
			// we keep a 'copy' of the previous marker (i.e. before the ith conditional dilation)
			marker_prev = marker.clone();

			// dilation...
			cv::dilate(marker, marker, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
			//                                                        /\
			//                                                        || see final comment and try to use a CV_SHAPE_CROSS instead

			// ...under the constraint 'F' ? 'G'
			marker = marker & img_bin;

			// display intermediate results with a delay of 200ms between two iterations
			cv::imshow("Reconstruction (in progress)", marker);
			if (cv::waitKey(200)>=0)
				cv::destroyWindow("Reconstruction (in progress)");

		} while (cv::countNonZero(marker - marker_prev) > 0);
		//                 /\
		//                 || counts the number of pixels different than '0'
		//                 || 'marker - marker_prev' contains at least one nonzero pixel if (and only if)
		//                    'marker' and 'marker_prev' are not equal, i.e. if 'marker_prev' has changed
		//                    so this conditions reads 'go on while the marker keeps changing' or
		//                    'stop when the marker does not change anymore

		aia::imshow("Reconstruction (result)", marker);
		// NOTE: the result may erroneously detected/reconstructed 'a' characters close to 't', since
		// in these cases, by a close look to the 'ta' sequences, we see 't' and 'a' are 8-connected
		// as a result of a nonideal binarization / acquisition (low resolution).
		// SOLUTION: use a 4-connected SE (CV_SHAPE_CROSS) for conditional dilation


		// STEP 2: FILLING
		img = marker.clone();
		marker.setTo(cv::Scalar(0));
		cv::rectangle(marker, cv::Rect(0,0,marker.cols-1,marker.rows-1), cv::Scalar(255));
		ucas::imshow("Hole filling (marker)", marker);
		cv::Mat mask = 255 - img;
		ucas::imshow("Hole filling (mask)", mask);
		do 
		{
			// we keep a 'copy' of the previous marker (i.e. before the ith conditional dilation)
			marker_prev = marker.clone();

			// dilation...
			cv::dilate(marker, marker, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));

			// ...under the constraint 'F' ? 'G'
			marker = marker & mask;

			// display intermediate results with a delay of 50ms between two iterations
			cv::imshow("Hole filling (in progress)", marker);
			if (cv::waitKey(50)>=0)
				cv::destroyWindow("Hole filling (in progress)");

		} while (cv::countNonZero(marker - marker_prev) > 0);

		ucas::imshow("Hole filling (result)", 255-marker);


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

