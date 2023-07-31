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
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree.tif", cv::IMREAD_GRAYSCALE);

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
			//cv::waitKey(10);
		}
	} 
	while (cv::countNonZero(previous-current));

	cv::Mat skeleton = current;
	aia::imshow("Skeletonization result", skeleton);
	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_skeleton.png", skeleton);


	cv::Mat pruning_SE_90 = (cv::Mat_<int>(3, 3)
		<<   0,  0,  0,
		    -1,  1, -1,
		    -1, -1, -1);
	std::vector<cv::Mat> pruning_SEs;
	pruning_SEs.push_back(pruning_SE_90);
	pruning_SEs.push_back(rotate90(pruning_SE_90, 1));
	pruning_SEs.push_back(rotate90(pruning_SE_90, 2));
	pruning_SEs.push_back(rotate90(pruning_SE_90, 3));

	current = skeleton.clone();
	int pruning_iters = 5;
	for(int k=0; k < pruning_iters; k++)
	{
		previous = current.clone();
		for (auto SE : pruning_SEs)
		{
			cv::Mat hitmiss_result;
			cv::morphologyEx(current, hitmiss_result, cv::MORPH_HITMISS, SE);
			current -= hitmiss_result;
			cv::imshow("Pruning in progress", current);
			cv::waitKey(10);
		}
	}

	cv::Mat pruned = current;
	aia::imshow("Pruning result", pruned);
	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_pruning.png", pruned);


	cv::Mat junction_SE_1 = (cv::Mat_<int>(3, 3)
		<<  1,  1,  1,
		   -1,  1, -1,
		    0,  1,  0);
	std::vector<cv::Mat> junction_SEs;
	junction_SEs.push_back(junction_SE_1);
	junction_SEs.push_back(rotate90(junction_SE_1, 1));
	junction_SEs.push_back(rotate90(junction_SE_1, 2));
	junction_SEs.push_back(rotate90(junction_SE_1, 3));

	cv::Mat match_result(img.rows, img.cols, CV_8U, cv::Scalar(0));
	for (auto SE : junction_SEs)
	{
		cv::Mat hitmiss_result;
		cv::morphologyEx(pruned, hitmiss_result, cv::MORPH_HITMISS, SE);
		match_result += hitmiss_result;
	}


	cv::cvtColor(pruned, pruned, cv::COLOR_GRAY2BGR);
	pruned.setTo(cv::Scalar(0, 0, 255), match_result);

	cv::morphologyEx(match_result, match_result, cv::MORPH_DILATE,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

	cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	img.setTo(cv::Scalar(0, 0, 255), match_result);


	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_junctions.png", img);
	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_junctions_skeleton.png", pruned);

	return EXIT_SUCCESS;
}
