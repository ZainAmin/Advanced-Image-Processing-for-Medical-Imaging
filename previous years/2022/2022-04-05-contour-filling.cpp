// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	cv::Mat img(800, 800, CV_8U, cv::Scalar(0));
	cv::rectangle(img, cv::Rect(50, 50, 100, 150), cv::Scalar(255), 2);
	cv::rectangle(img, cv::Rect(400, 400, 200, 250), cv::Scalar(255), 2);
	aia::imshow("Image", img);
	
	// STEP 2: FILLING
	cv::Mat marker = img.clone();
	marker.setTo(cv::Scalar(0));
	cv::rectangle(marker, cv::Rect(0, 0, marker.cols - 1, marker.rows - 1), cv::Scalar(255));
	ucas::imshow("Hole filling (marker)", marker);
	cv::Mat mask = 255 - img;
	ucas::imshow("Hole filling (mask)", mask);
	cv::Mat marker_prev;
	do
	{
		// we keep a 'copy' of the previous marker (i.e. before the ith conditional dilation)
		marker_prev = marker.clone();

		// dilation...
		cv::dilate(marker, marker, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

		// ...under the constraint 'F' ? 'G'
		marker = marker & mask;

		// display intermediate results with a delay of 50ms between two iterations
		cv::imshow("Hole filling (in progress)", marker);
		if (cv::waitKey(50) >= 0)
			cv::destroyWindow("Hole filling (in progress)");

	} while (cv::countNonZero(marker - marker_prev) > 0);

	ucas::imshow("Hole filling (result)", 255 - marker);

	return EXIT_SUCCESS;
}