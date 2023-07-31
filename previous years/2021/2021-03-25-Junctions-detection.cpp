// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"


// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
cv::Mat rotate90(cv::Mat img, int step);

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		//cv::Mat img(500, 500, CV_8U, cv::Scalar(0));
		//cv::rectangle(img, cv::Rect(150, 150, 200, 200), cv::Scalar(255), CV_FILLED);

		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree.tif", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat img_copy = img.clone();
		aia::imshow("Image", img);

		// STEP 1: fill single pixel holes
		std::vector <cv::Mat> filling_SEs;
		cv::Mat isolated_pattern_1 = (cv::Mat_<int>(3, 3) <<
			1,  1,  1,
			1, -1,  0,
			1,  1,  1);
		filling_SEs.push_back(isolated_pattern_1);
		filling_SEs.push_back(rotate90(isolated_pattern_1, 1));
		filling_SEs.push_back(rotate90(isolated_pattern_1, 2));
		filling_SEs.push_back(rotate90(isolated_pattern_1, 3));
		cv::Mat isolated_pattern_2 = (cv::Mat_<int>(3, 3) <<
			1,  1,  1,
			1, -1,  1,
			1,  1,  0);
		filling_SEs.push_back(isolated_pattern_2);
		filling_SEs.push_back(rotate90(isolated_pattern_2, 1));
		filling_SEs.push_back(rotate90(isolated_pattern_2, 2));
		filling_SEs.push_back(rotate90(isolated_pattern_2, 3));
		for(int i=0; i<3; i++)
		{
			// filling
			for(int i=0; i<filling_SEs.size(); i++)
			{
				cv::Mat hit_or_miss_res;
				cv::morphologyEx(img, hit_or_miss_res, cv::MORPH_HITMISS, filling_SEs[i]);
				img += hit_or_miss_res;
			}
		}
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_filled.tif", img);


		// STEP 2: skeletonization
		std::vector <cv::Mat> skeletonization_SEs;
		cv::Mat edge_pattern_1 = (cv::Mat_<int>(3, 3) <<
			-1, -1, -1,
			 0,  1,  0,
			 1,  1,  1);
		skeletonization_SEs.push_back(edge_pattern_1);
		skeletonization_SEs.push_back(rotate90(edge_pattern_1, 1));
		skeletonization_SEs.push_back(rotate90(edge_pattern_1, 2));
		skeletonization_SEs.push_back(rotate90(edge_pattern_1, 3));
		cv::Mat edge_pattern_2 = (cv::Mat_<int>(3, 3) <<
			 0, -1, -1,
			 1,  1, -1,
			 1,  1,  0);
		skeletonization_SEs.push_back(edge_pattern_2);
		skeletonization_SEs.push_back(rotate90(edge_pattern_2, 1));
		skeletonization_SEs.push_back(rotate90(edge_pattern_2, 2));
		skeletonization_SEs.push_back(rotate90(edge_pattern_2, 3));
		cv::Mat img_prev;
		do
		{
			// save current image into img_prev
			img_prev = img.clone();

			// thinning
			for(int i=0; i<skeletonization_SEs.size(); i++)
			{
				cv::Mat hit_or_miss_res;
				cv::morphologyEx(img, hit_or_miss_res, cv::MORPH_HITMISS, skeletonization_SEs[i]);
				img -= hit_or_miss_res;
			}

			cv::waitKey(100);
			cv::imshow("Iterative thinning", img);
		}
		while(cv::countNonZero(img_prev - img));
		aia::imshow("Skeletonization result", img);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_skeleton.tif", img);

		// STEP 3: pruning (to remove skeleton protrusions which might result into false junctions)
		std::vector <cv::Mat> pruning_SEs;
		cv::Mat endpoint_pattern = (cv::Mat_<int>(3, 3) <<
			 0,  0,  0,
			-1,  1, -1,
			-1, -1, -1);
		pruning_SEs.push_back(endpoint_pattern);
		pruning_SEs.push_back(rotate90(endpoint_pattern, 1));
		pruning_SEs.push_back(rotate90(endpoint_pattern, 2));
		pruning_SEs.push_back(rotate90(endpoint_pattern, 3));
		for(int i=0; i<10; i++)
		{
			// pruning
			for(int i=0; i<pruning_SEs.size(); i++)
			{
				cv::Mat hit_or_miss_res;
				cv::morphologyEx(img, hit_or_miss_res, cv::MORPH_HITMISS, pruning_SEs[i]);
				img -= hit_or_miss_res;
			}

			cv::waitKey(50);
			cv::imshow("Iterative pruning", img);
		}
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_skeleton_pruned.tif", img);


		// STEP 4: junction detection
		std::vector <cv::Mat> junction_SEs;
		cv::Mat junction_pattern_1 = (cv::Mat_<int>(3, 3) <<
			 0,  0,  0,
			 1,  1,  1,
			 0,  1, -1);
		junction_SEs.push_back(junction_pattern_1);
		junction_SEs.push_back(rotate90(junction_pattern_1, 1));
		junction_SEs.push_back(rotate90(junction_pattern_1, 2));
		junction_SEs.push_back(rotate90(junction_pattern_1, 3));
		cv::Mat junction_pattern_2 = (cv::Mat_<int>(3, 3) <<
			 0,  0,  0,
			 1,  1,  1,
		    -1,  1,  0);
		junction_SEs.push_back(junction_pattern_2);
		junction_SEs.push_back(rotate90(junction_pattern_2, 1));
		junction_SEs.push_back(rotate90(junction_pattern_2, 2));
		junction_SEs.push_back(rotate90(junction_pattern_2, 3));
		cv::Mat junctions_image = img.clone();
		junctions_image.setTo(cv::Scalar(0));
		for(int i=0; i<junction_SEs.size(); i++)
		{
			cv::Mat hit_or_miss_res;
			cv::morphologyEx(img, hit_or_miss_res, cv::MORPH_HITMISS, junction_SEs[i]);
			junctions_image += hit_or_miss_res;
		}
		cv::dilate(junctions_image, junctions_image,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		img.setTo(cv::Scalar(0, 0, 255), junctions_image);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina_tree_junctions.tif", img);

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

// utility function that rotates 'img' by step*90°
// step = 0 --> no rotation
// step = 1 --> 90° CW rotation
// step = 2 --> 180° CW rotation
// step = 3 --> 270° CW rotation
cv::Mat rotate90(cv::Mat img, int step)
{
	cv::Mat img_rot;

	// adjust step in case it is negative
	if(step < 0)
		step = -step;
	// adjust step in case it exceeds 4
	step = step%4;

	// no rotation
	if(step == 0)
		img_rot = img;
	// 90° CW rotation
	else if(step == 1)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 1);
	}
	// 180° CW rotation
	else if(step == 2)
		cv::flip(img, img_rot, -1);
	// 270° CW rotation
	else if(step == 3)
	{
		cv::transpose(img, img_rot);
		cv::flip(img_rot, img_rot, 0);
	}

	return img_rot;
}
