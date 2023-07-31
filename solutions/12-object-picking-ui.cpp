// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	
	cv::Mat img;
	std::vector < std::vector <cv::Point> > contours;
	int clicked_contour_idx = -1;

	void aMouseCallback(int event, int x, int y, int, void* userdata)
	{
		if(event == cv::EVENT_LBUTTONDOWN)
		{
			clicked_contour_idx = -1;
			for(int i=0; i<contours.size(); i++)
				if(cv::pointPolygonTest(contours[i], cv::Point2f(x,y), false) >= 0)
				{
					clicked_contour_idx = i;
					break;
				}
		}
		cv::Mat img_copy = img.clone();

		if(clicked_contour_idx != -1)
		{
			cv::drawContours(img_copy, contours, clicked_contour_idx, cv::Scalar(255,0,0), 2, CV_AA);

			cv::Mat img_mask(img.rows, img.cols, CV_8U, cv::Scalar(0));
			cv::drawContours(img_mask, contours, clicked_contour_idx, cv::Scalar(255), CV_FILLED);

			cv::Mat img_copy_2 = img_copy.clone();
			img_copy_2.setTo(cv::Scalar(255,255,255), img_mask);
			cv::addWeighted(img_copy, 0.5, img_copy_2, 0.5, 0, img_copy);
		}
		
		cv::imshow("Picking", img_copy);
	}
}

int main() 
{
	try
	{	
		// load the image
		std::string img_name = "tools.bmp";
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_UNCHANGED);
		if(!img.data)
			throw ucas::Error("cannot load image");
		printf("image.depth = %d\n", ucas::imdepth(img.depth()));
		printf("image.channels = %d\n", img.channels());

		// binarize the image
		cv::Mat img_gray;
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		std::vector <int> histo = ucas::histogram(img_gray);
		int T = ucas::getTriangleAutoThreshold(histo);
		cv::threshold(img_gray, img_gray, T, 255, cv::THRESH_BINARY);
		ucas::imshow("binarized", img_gray);

		// remove binarization artifacts
		cv::medianBlur(img_gray, img_gray, 7);
		ucas::imshow("binarized + median", img_gray);

		// connected component extraction
		cv::findContours(img_gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		/*for(int i=0; i<contours.size(); i++)
		{
			cv::Mat img_copy = img.clone();
			cv::drawContours(img_copy, contours, i, cv::Scalar(255,0,0), 2, CV_AA);
			ucas::imshow("ith contour", img_copy);
		}*/

		cv::namedWindow("Picking");
		cv::setMouseCallback("Picking", aMouseCallback);
		aMouseCallback(0,0,0,0,0);
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

