// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace eiid
{
	cv::Mat img;
	int soglia;

	void aTrackbarCallback(int pos, void* userdata) 
	{
		cv::Mat img_sogliata;
		cv::threshold(img, img_sogliata, soglia, 255, cv::THRESH_BINARY);
		cv::imshow("Lena", img_sogliata);
	}
}


int main() 
{
	try
	{	
		eiid::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!eiid::img.data)
			throw ucas::Error("immagine non letta");
		ucas::imshow("Lena_original", eiid::img);

		cv::namedWindow("Lena");
		cv::createTrackbar("valore soglia T", "Lena", &eiid::soglia, 255, eiid::aTrackbarCallback);
		eiid::aTrackbarCallback(0,0);
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

