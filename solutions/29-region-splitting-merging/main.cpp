// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

using namespace ucas;
using namespace aia;

// put everything in a local namespace to avoid naming conflicts
namespace
{
	// global images
	cv::Mat img_gray;						// grayscale version
	cv::Mat img_color;						// color version

	// parameters
	int split_threshold = 25;				// splitting parameter
	float scaling_factor = 1.0;				// for visualization only


	// splitting predicate
	bool aSplittingPredicate(const cv::Mat& img)
	{
		double minv, maxv;
		cv::minMaxLoc(img, &minv, &maxv);
		return maxv - minv < split_threshold;

		//cv::Scalar mean, stdev;
		//cv::meanStdDev(img, mean, stdev);
		//return stdev[0] < stdev_threshold;
	}


	// merging predicate
	bool aMergingPredicate(const cv::Mat& img, Region* r1, Region* r2)
	{
		//// difference between max and min
		//double minV, maxV;
		//cv::minMaxLoc(img, &minV, &maxV, 0, 0, r1->mask | r2->mask);
		//return maxV-minV < 100;

		//// homogeneity testing (standard deviation)
		//cv::Scalar mean, stdev;
		//cv::meanStdDev(img, mean, stdev, r1->mask | r2->mask);
		//return stdev[0] < 20;*/

		// difference between means
		cv::Scalar mean1, stdev1;
		cv::meanStdDev(img, mean1, stdev1, r1->mask);
		cv::Scalar mean2, stdev2;
		cv::meanStdDev(img, mean2, stdev2, r2->mask);
		return abs(mean1[0] - mean2[0]) < 58;

		//// hypothesis testing
		//cv::Scalar mu0, mu1, mu2, s0, s1, s2;
		//cv::meanStdDev(img, mu0, s0, r1->mask | r2->mask);
		//cv::meanStdDev(img, mu1, s1, r1->mask);
		//cv::meanStdDev(img, mu2, s2, r2->mask);
		//int n1 = cv::countNonZero(r1->mask);
		//int n2 = cv::countNonZero(r2->mask);
		//float L = std::pow(s0[0], n1+n2) / ( std::pow(s1[0], n1) * std::pow(s2[0], n2) );
		//return L < 100000.0;
	}


	// trackbar interaction
	void split_update(int, void*)
	{
		// build Quad Tree from root node (= whole image)
		QuadTreeNode* quadtree = new QuadTreeNode(img_gray);

		// split
		quadtree->split(aSplittingPredicate);

		// get leaves (= image blocks)
		std::vector <QuadTreeNode*> leaves;
		quadtree->getLeaves(leaves);
		printf("# leaves after split = %d\n", leaves.size());

		// random coloring of each block
		cv::Mat img_split = img_color.clone();
		for (auto b : leaves)
		{
			cv::Point offset = ucas::imOffsetInParent(b->img_roi);
			cv::rectangle(img_split, cv::Rect(offset.x, offset.y, b->img_roi.cols, b->img_roi.rows), cv::Scalar(rand() % 256, rand() % 256, rand() % 256), -1);
		}

		// display result overlaid (=blending) on the original image
		cv::addWeighted(img_color, 0.6, img_split, 0.4, 0, img_split);
		aia::imshow("Region Splitting", img_split, false, scaling_factor);

		// release allocated memory / avoid memory leaks
		delete quadtree;
	}
}


int main()
{
	try
	{
		// load image
		img_color = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/skin_lesion.png");
		if (!img_color.data)
			throw aia::error("Cannot open image");
		//cv::pyrMeanShiftFiltering(img_color, img_color, 30, 50, 0);
		cv::resize(img_color, img_color, cv::Size(0, 0), 0.5, 0.5);
		cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);
		aia::imshow("Image", img_color, true, scaling_factor);

		// generate a binary image for debugging purposes
		// (it is much easier to grasp the expected splitting/merging for binary images)
		//cur_img.setTo(cv::Scalar(0,0,0));
		//cv::rectangle(cur_img, cv::Rect(64, 64, 100, 100), cv::Scalar(255,255,255), -1);

		// create window and insert the trackbar
		aia::imshow("Region Splitting", img_color, false, scaling_factor);
		cv::createTrackbar("threshold", "Region Splitting", &split_threshold, 100, split_update);

		// run
		split_update(1, 0);

		// wait for key press = windows stay opened until the user presses any key
		cv::waitKey(0);

		// re-run splitting with last saved settings
		QuadTreeNode* quadtree = new QuadTreeNode(img_gray);
		quadtree->split(aSplittingPredicate);
		std::vector <QuadTreeNode*> leaves;
		quadtree->getLeaves(leaves);

		// generate regions for merging
		std::vector<Region*> regions;
		for (auto leaf : leaves)
		{
			cv::Mat region_mask(img_color.rows, img_color.cols, CV_8U, cv::Scalar(0));
			cv::Point offset = ucas::imOffsetInParent(leaf->img_roi);
			cv::Rect region_rect = cv::Rect(offset.x, offset.y, leaf->img_roi.cols, leaf->img_roi.rows);
			cv::rectangle(region_mask, region_rect, cv::Scalar(255), -1);
			regions.push_back(new Region(&img_gray, region_mask, region_rect));
		}

		// run merging
		merge(regions, aMergingPredicate);

		// display result
		printf("# regions after merge = %d\n", regions.size());
		cv::Mat img_merged = img_color.clone();
		for (auto r : regions)
			img_merged.setTo(cv::Scalar(rand() % 256, rand() % 256, rand() % 256), r->mask);
		aia::imshow("Region Merging (result)", img_merged, true, scaling_factor);
		cv::addWeighted(img_color, 0.6, img_merged, 0.4, 0, img_merged);
		aia::imshow("Region Merging (overlaied)", img_merged, true, scaling_factor);

		return 1;
	}
	catch (aia::error& ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}
