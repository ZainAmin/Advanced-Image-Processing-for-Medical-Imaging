// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	std::string win_name = "Object picking";
	cv::Mat img;
	std::vector <std::vector <cv::Point>> objects_filtered;

	void objectPicking(int event, int x, int y, int, void* userdata)
	{
		cv::Mat img_copy = img.clone();
		if (event == cv::EVENT_LBUTTONDOWN)
		{
			for (int i = 0; i < objects_filtered.size(); i++)
				if (cv::pointPolygonTest(objects_filtered[i], cv::Point(x, y), false) > 0)
					cv::drawContours(img_copy, objects_filtered, i, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
			
		}
		cv::imshow(win_name, img_copy);
	}
}

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main()
{
	// load the image
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.png", cv::IMREAD_GRAYSCALE);

	int T = ucas::getTriangleAutoThreshold(ucas::histogram(img));
	cv::Mat binarized_img;
	cv::threshold(img, binarized_img, T, 255, cv::THRESH_BINARY);

	std::vector <std::vector <cv::Point>> objects;
	cv::findContours(binarized_img, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	printf("objects (before filtering) = %d\n", objects.size());

	objects_filtered;
	for (int i = 0; i < objects.size(); i++)
	{
		double A = cv::contourArea(objects[i]);
		if (A > 100)
			objects_filtered.push_back(objects[i]);
	}
	printf("objects (after filtering) = %d\n", objects_filtered.size());

	cv::namedWindow(win_name);
	cv::setMouseCallback(win_name, objectPicking);
	objectPicking(0, 0, 0, 0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}
