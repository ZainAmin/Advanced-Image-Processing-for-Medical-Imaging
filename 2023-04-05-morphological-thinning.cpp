// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
cv::Mat rotate90(cv::Mat img, int step)
{
	cv::Mat img_rot;

	// adjust step in case it is negative
	if (step < 0)
		step = -step;
	// adjust step in case it exceeds 4
	step = step % 4;

	// no rotation
	if (step == 0)
		img_rot = img;
	// 90° CW rotation
	else if (step == 1)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 1);
	}
	// 180° CW rotation
	else if (step == 2)
		cv::flip(img, img_rot, -1);
	// 270° CW rotation
	else if (step == 3)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 0);
	}

	return img_rot;
}



int main()
{
	cv::Mat img(600, 600, CV_8U, cv::Scalar(0));
	cv::rectangle(img, cv::Rect(200, 200, 300, 200), cv::Scalar(255), cv::FILLED);
	aia::imshow("Original image", img);

	cv::Mat thinning_SE_90 = (cv::Mat_<int>(3, 3)
		<< -1, -1, -1,
		    0,  1,  0,
		    1,  1,  1);
	cv::Mat thinning_SE_45 = (cv::Mat_<int>(3, 3)
		<<  0, -1, -1,
		    1,  1, -1,
		    1,  1,  0);
	std::vector<cv::Mat> thinning_SEs;
	thinning_SEs.push_back(thinning_SE_90);
	thinning_SEs.push_back(rotate90(thinning_SE_90, 1));
	thinning_SEs.push_back(rotate90(thinning_SE_90, 2));
	thinning_SEs.push_back(rotate90(thinning_SE_90, 3));
	thinning_SEs.push_back(thinning_SE_45);
	thinning_SEs.push_back(rotate90(thinning_SE_45, 1));
	thinning_SEs.push_back(rotate90(thinning_SE_45, 2));
	thinning_SEs.push_back(rotate90(thinning_SE_45, 3));

	cv::Mat current = img.clone();
	cv::Mat previous;
	do 
	{
		previous = current.clone();
		for (auto SE : thinning_SEs)
		{
			cv::Mat hitmiss_result;
			cv::morphologyEx(current, hitmiss_result, cv::MORPH_HITMISS, SE);
			current -= hitmiss_result;
			cv::imshow("Thinning in progress", current);
			cv::waitKey(10);
		}
	} 
	while (cv::countNonZero(previous-current));


	aia::imshow("Skeletonization result", current);

	return EXIT_SUCCESS;
}
