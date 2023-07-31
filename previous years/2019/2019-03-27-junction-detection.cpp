// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
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
}

int main() 
{
	try
	{	
		// load the binary vessel tree image
		std::string img_name = "retina_tree.tif";
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_GRAYSCALE);
		if(!img.data)
			throw ucas::Error("cannot load image");
		ucas::imshow("original image", img);

		// for debugging purposes
		//img.setTo(cv::Scalar(0));
		//cv::rectangle(img, cv::Rect(200, 200, 250, 250), cv::Scalar(255), CV_FILLED);
		//cv::circle(img, cv::Point(200,200), 100, cv::Scalar(255), CV_FILLED);

		// skeletonization
		std::vector <cv::Mat> thinning_SEs;
		thinning_SEs.push_back((cv::Mat_<float>(3, 3) <<
		   -1, -1, -1,
			0,  1,  0,
			1,  1,  1));
		thinning_SEs.push_back(rotate90(thinning_SEs[0], 1));
		thinning_SEs.push_back(rotate90(thinning_SEs[0], 2));
		thinning_SEs.push_back(rotate90(thinning_SEs[0], 3));
		thinning_SEs.push_back((cv::Mat_<float>(3, 3) <<
			0, -1, -1,
			1,  1, -1,
			1,  1,  0));
		thinning_SEs.push_back(rotate90(thinning_SEs[4], 1));
		thinning_SEs.push_back(rotate90(thinning_SEs[4], 2));
		thinning_SEs.push_back(rotate90(thinning_SEs[4], 3));
		cv::Mat thinned = img.clone();
		cv::Mat thinned_prev;
		do 
		{
			thinned_prev = thinned.clone();

			for(int i=0; i<thinning_SEs.size(); i++)
			{
				cv::Mat hit_or_miss_res;
				cv::morphologyEx(thinned, hit_or_miss_res, cv::MORPH_HITMISS, thinning_SEs[i]);
				thinned -= hit_or_miss_res;
			}

			cv::imshow("skeletonization", thinned);
			if (cv::waitKey(20)>=0)
				cv::destroyWindow("skeletonization");
			
		} while (cv::countNonZero(thinned_prev-thinned) > 0);
		cv::imwrite("C:/work/skeleton.png", thinned);

		// pruning
		std::vector <cv::Mat> pruning_SEs;
		pruning_SEs.push_back((cv::Mat_<float>(3, 3) <<
			0,  0,  0,
		   -1,  1, -1,
		   -1, -1, -1));
		pruning_SEs.push_back(rotate90(pruning_SEs[0], 1));
		pruning_SEs.push_back(rotate90(pruning_SEs[0], 2));
		pruning_SEs.push_back(rotate90(pruning_SEs[0], 3));
		cv::Mat pruned = thinned.clone();
		for(int k=0; k<10; k++) 
		{
			for(int i=0; i<pruning_SEs.size(); i++)
			{
				cv::Mat hit_or_miss_res;
				cv::morphologyEx(pruned, hit_or_miss_res, cv::MORPH_HITMISS, pruning_SEs[i]);
				pruned -= hit_or_miss_res;
			}

			cv::imshow("Pruning", pruned);
			if (cv::waitKey(50)>=0)
				cv::destroyWindow("Pruning");
			
		}
		cv::imwrite("C:/work/skeleton.pruned.png", pruned);

		// detect T-junctions
		std::vector <cv::Mat> junctions_SEs;
		junctions_SEs.push_back((cv::Mat_<float>(3, 3) <<
			 0, -1,  0,
			 1,  1,  1,
			-1,  1, -1));
		junctions_SEs.push_back(rotate90(junctions_SEs[0], 1));
		junctions_SEs.push_back(rotate90(junctions_SEs[0], 2));
		junctions_SEs.push_back(rotate90(junctions_SEs[0], 3));
		
		cv::Mat junctions_image(img.rows, img.cols, CV_8U, cv::Scalar(0));
		for(int i=0; i<junctions_SEs.size(); i++)
		{
			cv::Mat hit_or_miss_res;
			cv::morphologyEx(pruned, hit_or_miss_res, cv::MORPH_HITMISS, junctions_SEs[i]);
			junctions_image += hit_or_miss_res;
		}
		cv::dilate(junctions_image, junctions_image, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		img.setTo(cv::Scalar(0,0,255), junctions_image);
		cv::imwrite("C:/work/junctions.png", img);

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