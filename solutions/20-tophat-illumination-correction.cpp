// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: correct nonuniform illumination field
int main() 
{
	try
	{
		// load image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw aia::error("Cannot open image");


		// calculate illumination field by opening with a big SE
		cv::Mat illumination_img;
		cv::morphologyEx(img, illumination_img, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(40,40)));
		aia::imshow("Illumination Image", illumination_img);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/rice_ill.png", illumination_img);
		

		// suppress background with top-hat
		cv::Mat img_corrected;
		cv::morphologyEx(img, img_corrected, CV_MOP_TOPHAT, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(40,40)));
		aia::imshow("Top-hat", img_corrected);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/rice_tophat.png", img_corrected);


		// restore background by adding the average intensity of the illumination field
		img_corrected = img_corrected + cv::mean(illumination_img)[0];
		aia::imshow("Corrected image", img_corrected);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/rice_corrected.png", img_corrected);

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