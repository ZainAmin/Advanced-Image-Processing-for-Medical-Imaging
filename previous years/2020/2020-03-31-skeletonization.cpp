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
		cv::flip(mat, mat_rotated, 0);
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
		img.setTo(cv::Scalar(0));
		cv::rectangle(img, cv::Point(50,50), cv::Point(img.cols-50, img.rows-50), cv::Scalar(255), CV_FILLED);
		aia::imshow("Image", img);

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

		cv::Mat prev = img;
		cv::Mat curr = prev.clone();
		cv::Mat hit_miss_res;
		int iters = 0;
		do 
		{
			printf("iteration # %d\n", ++iters);
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

		aia::imshow("Thinning result", curr);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_skeleton.png", curr);
		//

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

