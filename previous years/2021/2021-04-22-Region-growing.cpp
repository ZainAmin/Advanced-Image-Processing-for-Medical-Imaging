// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat predicateLocalVariance(const cv::Mat & img, int neighborhood = 5)
{
	cv::Mat sq_avg;
	cv::boxFilter(img, sq_avg, CV_32F, cv::Size(neighborhood, neighborhood));
	cv::pow(sq_avg, 2, sq_avg);

	cv::Mat avg_sq;
	cv::Mat img_sq;
	img.convertTo(img_sq, CV_32F);
	cv::pow(img_sq, 2, img_sq);
	cv::boxFilter(img_sq, avg_sq, CV_32F, cv::Size(neighborhood, neighborhood));

	cv::Mat stdev;
	cv::sqrt(avg_sq-sq_avg, stdev);

	/*cv::Mat varianceImg(img.rows, img.cols, CV_32F, cv::Scalar(0));
	for(int i=neighborhood/2; i<img.rows-neighborhood; i++)
	{
		float* iRow = varianceImg.ptr<float>(i);
		for(int j=neighborhood/2; j<img.cols-neighborhood; j++)
		{
			cv::Scalar mean, stddev;
			cv::meanStdDev(
				img(cv::Rect(i-neighborhood/2, j-neighborhood/2, neighborhood, neighborhood)),
				mean, stddev);
			iRow[j] = stddev[0];
		}
	}*/

	cv::normalize(stdev, stdev, 0, 255, cv::NORM_MINMAX);
	stdev.convertTo(stdev, CV_8U);

	return stdev;
}

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		
		// seeds extraction
		cv::Mat seeds;
		cv::threshold(img, seeds, 254, 255, cv::THRESH_BINARY);

		cv::Mat seeds_prev;
		do 
		{
			// save current seeds for later comparison
			seeds.copyTo(seeds_prev);

			// find candidates with 3x3 seeds morphological dilation
			cv::Mat candidates;
			cv::dilate(seeds, candidates,
				cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
			candidates -= seeds;

			// generate predicate image(s)
			cv::Mat predicate1_img;
			cv::threshold(predicateLocalVariance(img), predicate1_img, 20, 255, cv::THRESH_BINARY);
			cv::Mat predicate2_img;
			cv::threshold(img, predicate2_img, 50, 255, cv::THRESH_BINARY);

			// candidate selection
			candidates = candidates & predicate1_img & predicate2_img;

			// region growing
			seeds += candidates;

			cv::imshow("Region growing", seeds);
			cv::waitKey(10);

		} while (cv::countNonZero(seeds-seeds_prev));

		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_result.png", seeds);

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

	return EXIT_SUCCESS;
}


