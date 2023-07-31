// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/imgproc.hpp>

// GOAL: perform real-time object shape recognition (12 classes)

typedef std::vector <cv::Point> contour;
typedef std::vector <contour> contourSet;

namespace aia
{
	std::map<std::string, contour> shapes_library;

	bool sortByDescendingArea(contour & first, contour & second)
	{
		return cv::contourArea(first) > contourArea(second);
	}

	cv::Mat shapeRecognizer(const cv::Mat & frame)
	{
		cv::Mat result = frame.clone();
		cv::Point center(frame.cols/2, frame.rows/2);
		cv::circle(result, center, 6, cv::Scalar(0, 0, 255), CV_FILLED, CV_AA);

		// Otsu binarization in grayscale
		cv::Mat frame_gray;
		cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
		cv::threshold(frame_gray, frame_gray, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

		// connected component extraction
		contourSet objects;
		cv::findContours(frame_gray, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// find object where the reference point fits into
		for(int k=0; k<objects.size(); k++)
		{
			if(cv::pointPolygonTest(objects[k], center, false) > 0)
			{
				cv::drawContours(result, objects, k, cv::Scalar(255, 0, 0), 2, CV_AA);

				// 1-nearest-neighbor classification
				double minDist = std::numeric_limits<double>::infinity();
				std::string shape_class;
				for(auto & shape : shapes_library)
				{
					double dist = cv::matchShapes(objects[k], shape.second, CV_CONTOURS_MATCH_I1, 0);
					if(dist < minDist)
					{
						minDist = dist;
						shape_class = shape.first;
					}
				}

				cv::putText(result, shape_class, objects[k][0], 2, 1, cv::Scalar(255, 0, 0), 1, CV_AA);
			}
		}

		return result;
	}
}

int main() 
{
	try
	{
		// retrieve all .bmp files in the shape folder
		std::vector < std::string> filenames;
		cv::glob(std::string(EXAMPLE_IMAGES_PATH) + "/other shapes/*.bmp", filenames);

		// build our training set (template set) shapes library
		for(auto & f: filenames)
		{
			cv::Mat img = cv::imread(f, CV_LOAD_IMAGE_GRAYSCALE);

			// binarization
			cv::threshold(img, img, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

			// connected component extraction
			contourSet objects;
			cv::findContours(img, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

			// select largest area object
			std::sort(objects.begin(), objects.end(), aia::sortByDescendingArea);

			// visual debugging
			//img.setTo(cv::Scalar(0));
			//cv::drawContours(img, objects, 0, cv::Scalar(255), CV_FILLED);
			//aia::imshow("Object", img);

			aia::shapes_library[ucas::getFileName(f, false)] = objects[0];
		}

		
		aia::processVideoStream("", aia::shapeRecognizer);


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