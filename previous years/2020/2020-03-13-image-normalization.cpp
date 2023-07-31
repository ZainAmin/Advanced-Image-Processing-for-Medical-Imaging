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
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png", CV_LOAD_IMAGE_GRAYSCALE);

		aia::imshow("Immagine originale", img);
		aia::imshow("Istogramma originale", ucas::imhist(img));

		cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);

		aia::imshow("Immagine normalizata", img);
		aia::imshow("Istogramma post normalizzazione", ucas::imhist(img));

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

