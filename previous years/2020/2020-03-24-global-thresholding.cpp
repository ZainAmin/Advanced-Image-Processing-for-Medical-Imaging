// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() 
{
	try
	{
		//cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.bmp", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img, true, 1.0f);

		auto histo = ucas::histogram(img);

		cv::Mat img_bin;

		int T = ucas::getOtsuAutoThreshold(histo);
		printf("otsu T = %d\n", T);
		img_bin = ucas::binarize(img.clone(), T);
		aia::imshow("Otsu Binarization", img_bin, true, 1.0f);

		T = ucas::getTriangleAutoThreshold(histo);
		printf("triangle T = %d\n", T);
		img_bin = ucas::binarize(img.clone(), T);
		aia::imshow("Triangle Binarization", img_bin, true, 1.0f);

		T = ucas::getRenyiEntropyAutoThreshold(histo);
		printf("reny T = %d\n", T);
		img_bin = ucas::binarize(img.clone(), T);
		aia::imshow("Renyi Entropy Binarization", img_bin, true, 1.0f);

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

