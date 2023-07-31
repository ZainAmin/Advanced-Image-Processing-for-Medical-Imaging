// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

namespace
{
	cv::Mat img;
	int threshold = 50;
	std::string win_name = "Region splitting";

	bool splitPredicate(const cv::Mat & img, cv::Rect & region)
	{
		double minV, maxV;
		cv::minMaxLoc(img(region), &minV, &maxV);
		return maxV-minV >= threshold;
	}

	bool mergePredicate(const cv::Mat & img, aia::QuadTreeNode* region1, aia::QuadTreeNode* region2)
	{
		return true;
	}

	void splitting(int pos, void* userdata)
	{
		ucas::Timer timer;
		aia::QuadTreeNode *root = new aia::QuadTreeNode(cv::Rect(0, 0, img.cols, img.rows));
		root->split(splitPredicate, img, 50);
		printf("split time = %.3f\n", timer.elapsed<float>());

		std::vector <aia::QuadTreeNode*> leaves;
		root->getLeaves(leaves);
		printf("# regions = %d\n", leaves.size());

		// debug
		cv::Mat regions_img(img.rows, img.cols, CV_8UC(3), cv::Scalar(0,0,0));
		for(auto & leaf : leaves)
			cv::rectangle(regions_img, leaf->roi, cv::Scalar(rand()%256, rand()%256, rand()%256), CV_FILLED);
		cv::imshow(win_name, regions_img);

		delete root;
	}
}

int main() 
{
	try
	{
		/*aia::QuadTreeNode n1(cv::Rect(0, 0, 10, 10));
		aia::QuadTreeNode n2(cv::Rect(10, 0, 10, 10));
		printf("adjacent = %d\n", n1.isAdjacent(&n2));
		return 1;*/

		// load image
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//img = cv::Mat(512, 512, CV_8U, cv::Scalar(0));
		/*cv::rectangle(img, cv::Rect(20, 20, 100, 300), cv::Scalar(255), CV_FILLED);
		cv::rectangle(img, cv::Rect(20, 350, 150, 100), cv::Scalar(200), CV_FILLED);
		cv::rectangle(img, cv::Rect(300, 400, 150, 50), cv::Scalar(160), CV_FILLED);*/
		//cv::rectangle(img, cv::Rect(256, 256, 128, 128), cv::Scalar(160), CV_FILLED);
		//cv::rectangle(img, cv::Rect(256+128, 256-64, 64, 128), cv::Scalar(200), CV_FILLED);

		aia::imshow("Image", img);


		cv::namedWindow(win_name);
		cv::createTrackbar("threshold", win_name, &threshold, 100, splitting);

		splitting(0, 0);
		cv::waitKey(0);

		// re-run splitting with the 'accepted' configuration
		aia::QuadTreeNode *root = new aia::QuadTreeNode(cv::Rect(0, 0, img.cols, img.rows));
		root->split(splitPredicate, img);
		std::vector <aia::QuadTreeNode*> leaves;
		root->getLeaves(leaves);

		ucas::Timer timer;

		// run merging

		// STEP 1: find adjacencies
		int count = 0;
		for(int i=0; i<leaves.size(); i++)
			for(int j=0; j<leaves.size(); j++)
				if(i!=j && leaves[i]->isAdjacent(leaves[j]))
					leaves[i]->neighbors.push_back(leaves[j]);
		printf("find adjacencies = %.3f\n", timer.elapsed<float>());

		// STEP 2: merging
		bool merging_occurred = false;
		do 
		{
			for(int i=0; i<leaves.size(); i++)
				for(auto neighbor : leaves[i]->neighbors)
					if(mergePredicate(img, leaves[i], neighbor))
					{
						merging_occurred = true;
						leaves[i]->merge(neighbor);
					}

		} while (merging_occurred);
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
