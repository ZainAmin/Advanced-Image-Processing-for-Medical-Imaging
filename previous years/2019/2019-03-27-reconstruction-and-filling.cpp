// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

int main() 
{
	try
	{	
		// load the binary vessel tree image
		std::string img_name = "text.png";
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw ucas::Error("cannot load image");
		ucas::imshow("original image", img);

		// inverse thresholding
		cv::threshold(img, img, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
		ucas::imshow("binarized", img);
		cv::imwrite("C:/work/text.binarized.png", img);

		// generate marker image
		cv::Mat marker;
		cv::erode(img, marker, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 25)));
		ucas::imshow("marker", marker);
		cv::Mat reconstructed = marker.clone();
		cv::Mat reconstructed_prev;
		do 
		{
			reconstructed_prev = reconstructed.clone();

			cv::morphologyEx(reconstructed, 
				reconstructed, 
				cv::MORPH_DILATE, 
				cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3)));

			reconstructed = reconstructed & img;

			cv::imshow("reconstruction", reconstructed);
			if (cv::waitKey(100)>=0)
				cv::destroyWindow("reconstruction");

		} while (cv::countNonZero(reconstructed-reconstructed_prev) > 0);
		//ucas::imshow("result", reconstructed);

		img = reconstructed.clone();

		// filling
		// generate marker
		cv::Mat mask(img.rows, img.cols, CV_8U, cv::Scalar(0));
		marker = mask.clone();
		cv::rectangle(mask, cv::Rect(0, 0, img.cols, img.rows), cv::Scalar(255), 1);
		img = 255 - img;
		img.copyTo(marker, mask);
		// reconstruction
		reconstructed = marker.clone();
		reconstructed_prev;
		do 
		{
			reconstructed_prev = reconstructed.clone();

			cv::morphologyEx(reconstructed, 
				reconstructed, 
				cv::MORPH_DILATE, 
				cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

			reconstructed = reconstructed & img;

			cv::imshow("reconstruction", reconstructed);
			if (cv::waitKey(10)>=0)
				cv::destroyWindow("reconstruction");

		} while (cv::countNonZero(reconstructed-reconstructed_prev) > 0);
		ucas::imshow("final result", 255-reconstructed);

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