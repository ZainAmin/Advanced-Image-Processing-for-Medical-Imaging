// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main()
{
	try
	{
		// EXAMPLE 1: load and show an RGB image
		cv::Mat imgRGB = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", cv::IMREAD_UNCHANGED);
		//                    /\
		//                    || like MATLAB! We just need to provide the image full path
		//                    || cv::IMREAD_UNCHANGED means "load the image as it is, without grayscale-to-color
		//                    || conversion (default) and 16-to-8-bit conversion (default). Not a problem in this case,
		//                    || but it can be in EXAMPLE 2 and 3!
		// check if image data was successfully loaded. If not, we throw an error exception
		if (!imgRGB.data)
			throw aia::error("Cannot load image");
		// print some image characteristics (X-Y dimensions, number of channels, bitdepth)
		printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",
			imgRGB.rows, imgRGB.cols, imgRGB.channels(), aia::bitdepth(imgRGB.depth()));
		aia::imshow("An RGB image", imgRGB);
		//     /\
		//     || aia::imshow is a slightly modified version of cv::imshow (see EXAMPLE 3 for the additional arguments)


		//// EXAMPLE 2: load and show a grayscale image (8 bits per pixel)
		//cv::Mat imgGray8 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png", cv::IMREAD_UNCHANGED);
		//if(!imgGray8.data)
		//	throw aia::error("Cannot load image");
		//printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",
		//	imgGray8.rows, imgGray8.cols, imgGray8.channels(), aia::bitdepth(imgGray8.depth()));
		//aia::imshow("An 8-bit grayscale image", imgGray8);


		//// EXAMPLE 3: load and show a grayscale image (16 bits per pixel)
		//cv::Mat imgGray16 = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram.tif", cv::IMREAD_UNCHANGED);
		//if(!imgGray16.data)
		//	throw aia::error("Cannot load image");
		//printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n",
		//	imgGray16.rows, imgGray16.cols, imgGray16.channels(), aia::bitdepth(imgGray16.depth()));
		//aia::imshow("An 16-bit grayscale image", imgGray16, true, 0.3f);
		////                                                   /\    /\
		////                                                   ||    || scale factor (mammograms are too big to fit in the screeen!)
		////                                                   || wait for the user to press a key before closing the window

		//// EXAMPLE 4: do some image processing and save the result
		//// histogram equalization is just one of MANY functions available in OpenCV
		//cv::equalizeHist(imgGray8, imgGray8);
		//cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.EQ.png", imgGray8);


		// EXAMPLE 5: video processing: Face Detection
		// - requires a (working) camera
		// - requires a Face Detector (pre-trained classifier): automatically loaded within faceRectangles()
		/* comment this if you do not want to run it / it does not work */
		//aia::processVideoStream("", aia::project0::faceRectangles);
		//                      /\                       /\
		//                      || empty video input = use camera
		//                                               || function / processing to be applied to each frame of the video sequence

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

