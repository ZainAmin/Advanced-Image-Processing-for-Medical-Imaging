// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat rotate90(const cv::Mat & mat, int k)
{
	// check/adjust input correctness
	if(k < 0 )
		k*= -1;
	k = k%4;

	cv::Mat mat_rotated;

	// no rotation
	if(k == 0)
		mat_rotated = mat;
	// 90° rotation
	else if(k == 1)
	{
		//1 2 3
		//x x x
		//x x x
		//// transpose
		//1 x x
		//2 x x
		//3 x x
		//// flip around y-axis
		//x x 1
		//x x 2
		//x x 3
		//// = 90° rotation

		cv::transpose(mat, mat_rotated);
		cv::flip(mat_rotated, mat_rotated, 1);
	}
	// 180° rotation
	else if(k == 2)
	{
		//1 2 3
		//4 5 6
		//7 8 9
		//// flip around X
		//7 8 9
		//4 5 6
		//1 2 3
		//// flip around Y
		//9 8 7
		//6 5 4
		//3 2 1
		//// = 180° rotation

		cv::flip(mat, mat_rotated, -1);
	}
	// 270° rotation
	else
	{
		//1 2 3
		//x x x
		//x x x
		//// transpose
		//1 x x
		//2 x x
		//3 x x
		//// flip around x-axis
		//3 x x
		//2 x x
		//1 x x
		//// = 270° rotation
		cv::transpose(mat, mat_rotated);
		cv::flip(mat_rotated, mat_rotated, 0);
	}

	return mat_rotated;
}

int main() 
{
	try
	{
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree.tif", CV_LOAD_IMAGE_GRAYSCALE);
		aia::imshow("Image", img);

		// STEP 1: skeletonization
		std::vector <cv::Mat> thinning_SEs;
		thinning_SEs.push_back(
			(cv::Mat_<char>(3,3) <<
				-1, -1, -1,
				 0,  1,  0,
				 1,  1,  1
			));
		thinning_SEs.push_back(
			(cv::Mat_<char>(3,3) <<
				 0, -1, -1,
				 1,  1, -1,
				 1,  1,  0
			));

		cv::Mat prev = img.clone();
		cv::Mat curr = prev.clone();
		cv::Mat hit_miss_res;
		int iters = 0;
		do 
		{
			printf("thinning iteration # %d\n", ++iters);
			curr.copyTo(prev);
			for(auto & base_kernel : thinning_SEs)
				for(int k=0; k<4; k++)
				{
					cv::morphologyEx(curr, hit_miss_res, cv::MORPH_HITMISS, rotate90(base_kernel, k));
					curr = curr - hit_miss_res;
				}
			cv::imshow("Thinning", curr);
			cv::waitKey(100);

		} while (cv::countNonZero(prev-curr));
		printf("\n");
		aia::imshow("Thinning result", curr);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_skeleton.png", curr);
		

		// STEP 2: pruning
		std::vector <cv::Mat> pruning_SEs;
		pruning_SEs.push_back(
			(cv::Mat_<char>(3,3) <<
				 0,  0,  0,
				-1,  1, -1,
				-1, -1, -1
			));

		int pruning_iters = 10;
		iters = 0;
		do 
		{
			printf("pruning iteration # %d\n", ++iters);
			curr.copyTo(prev);
			for(auto & base_kernel : pruning_SEs)
				for(int k=0; k<4; k++)
				{
					cv::morphologyEx(curr, hit_miss_res, cv::MORPH_HITMISS, rotate90(base_kernel, k));
					curr = curr - hit_miss_res;
				}
				cv::imshow("Pruning", curr);
				cv::waitKey(100);

		} while (iters <= pruning_iters);
		printf("\n");
		aia::imshow("Pruning result", curr);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_pruned.png", curr);


		// STEP 3: search pattern "junction"
		std::vector <cv::Mat> junctions_SEs;
		junctions_SEs.push_back(
			(cv::Mat_<char>(3,3) <<
				 0, -1,  0,
				 1,  1,  1,
				-1,  1, -1
			));

		cv::Mat junctions = curr.clone();
		junctions.setTo(cv::Scalar(0));

		for(auto & base_kernel : junctions_SEs)
			for(int k=0; k<4; k++)
			{
				cv::morphologyEx(curr, hit_miss_res, cv::MORPH_HITMISS, rotate90(base_kernel, k));
				junctions += hit_miss_res;
			}

		// result visualization #1: red circles
		cv::Mat junctions_circles;
		cv::dilate(junctions, junctions_circles, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9,9)));
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		cv::Mat junctions_img_red = img.clone();
		junctions_img_red.setTo(cv::Scalar(0, 0, 255), junctions_circles);
		aia::imshow("Junctions (red circles)", junctions_img_red);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_junctions.png", junctions_img_red);

		// result visualization #2: blue boxes
		std::vector <std::vector <cv::Point> > junctions_objects;
		cv::findContours(junctions, junctions_objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		for(auto & j : junctions_objects)
		{
			cv::Rect bbox = cv::boundingRect(j);
			bbox.x -= 4;
			bbox.y -= 4;
			bbox.width = 8;
			bbox.height = 8;
			cv::rectangle(img, bbox, cv::Scalar(255, 0, 0));
			
			curr.at<unsigned char>(j[0].y, j[0].x) = 120;
			//cv::Mat roi = curr(bbox);
			//aia::imshow("Junction", roi, true, 25);
		}
		aia::imshow("Junctions (boxes)", img);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_junctions_boxes.png", img);

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

