#pragma once

#include <aiaConfig.h>

namespace aia
{
	typedef std::vector< cv::Point > contour;
	typedef std::vector< std::vector <cv::Point> > contourSet;

	// utility functions
	contourSet & filterObjects(contourSet & objs, bool (*pred)(contour & obj));
	cv::RotatedRect correctedMinAreaRect(contour & obj);

	// plate detection criteria
	bool areaCriterion(contour & obj);
	bool orientationCriterion(contour & obj);
	bool aspectRatioCriterion(contour & obj);
	bool rectangularityCriterion(contour & obj);

	// plate characters detection criteria
	bool circularityCriterion(contour & obj);
	bool sortByDescendingArea(contour & first, contour & second);
	bool sortByAscendingX(contour & first, contour & second);

	// plate characters recognition criteria
	std::string computeModeCharwise(std::vector <std::string> & plate_recognitions);
	
	class OCR
	{
		private:

			cv::Mat letters_image;
			cv::Mat digits_image;
			contourSet letters_shapes;
			contourSet digits_shapes;

			OCR();	// constructor inaccessible

		public:

			static OCR& instance();	// singleton design pattern

			// intensity-based matching
			char intensity_matching (cv::Mat & image, bool isLetter) throw (aia::error);

			// shape-based matching
			char shape_matching (contour & shape, bool isLetter) throw (aia::error);
	};
}
