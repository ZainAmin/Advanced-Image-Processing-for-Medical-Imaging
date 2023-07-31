// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	cv::Mat input_img;
	std::string win_name = "Select object";
	std::vector < std::vector <cv::Point> > objects_filtered;

	void objectSelectionCallback(int event, int x, int y, int, void* userdata)
	{
		int selected_obj_index = -1;

		for(int i=0; i<objects_filtered.size(); i++)
			if(cv::pointPolygonTest(objects_filtered[i], cv::Point2f(x,y), false) >= 0)
			{
				selected_obj_index = i;

				double A = cv::contourArea(objects_filtered[i]);
				double P = cv::arcLength(objects_filtered[i], true);
				printf("\nObject selected\nArea = %.0f\nAngle = %.0f\nCircularity = %.3f\n",
					A,
					cv::minAreaRect(objects_filtered[i]).angle,
					(4*ucas::PI*A)/(P*P) );


				break;
			}
		
		cv::Mat selected_obj_img = input_img.clone();
		
		if(selected_obj_index >= 0)
		{
			
			cv::drawContours(selected_obj_img, objects_filtered, selected_obj_index,
				cv::Scalar(255, 255, 255), -1, CV_AA);
			cv::addWeighted(selected_obj_img, 0.5, input_img, 0.5, 0, selected_obj_img);
		}
		

		cv::imshow(win_name, selected_obj_img);
	}
}

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		// load the image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.bmp");
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		cv::Mat input_img_gray;
		cv::cvtColor(input_img, input_img_gray, cv::COLOR_BGR2GRAY);

		//aia::imshow("Original image", input_img);
		aia::imshow("Histogram", ucas::imhist(input_img_gray));

		// Otsu-binarization does not work well on this image (not a bimodal histogram)
		/*int T = ucas::getOtsuAutoThreshold(ucas::histogram(input_img));
		printf("Otsu T = %d\n", T);
		cv::Mat binarized_img;
		cv::threshold(input_img, binarized_img, T, 255, CV_THRESH_BINARY);
		aia::imshow("Otsu-binarized image", binarized_img);*/

		// Triangle-binarization works better (dominant dark background)
		int T = ucas::getTriangleAutoThreshold(ucas::histogram(input_img_gray));
		//printf("Triangle T = %d\n", T);
		cv::Mat binarized_img;
		cv::threshold(input_img_gray, binarized_img, T, 255, CV_THRESH_BINARY);
		aia::imshow("Triangle-binarized image", binarized_img);

		// connectect compontent extraction
		std::vector < std::vector <cv::Point> > objects;
		cv::findContours(binarized_img, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
		printf("Extracted %d objects\n", objects.size());

		// connected component filtering
		for(auto & obj : objects)
			if(cv::contourArea(obj) > 50)
				objects_filtered.push_back(obj);
		printf("Objects after filtering = %d\n", objects_filtered.size());

		// object selection GUI
		cv::namedWindow(win_name);
		cv::setMouseCallback(win_name, objectSelectionCallback);

		// launch GUI
		objectSelectionCallback(-1, 0, 0, 0, 0);

		// waits for user to press a button and exit from the app
		cv::waitKey(0);

		return EXIT_SUCCESS;
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
