#pragma once

#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/core/core.hpp>
#include <set>

using ucas::range;	// tell compiler we will use this type

// open namespace "aia"
namespace aia
{
	// REGION SPLITTING
	// QuadTree node class
	// (remember a 'struct' in C++ is a 100% public 'class')
	struct QuadTreeNode 
	{
		// members
		cv::Mat img_roi;							// image roi
		std::vector <QuadTreeNode*> children;		// children nodes

		// constructor / deconstructor
		QuadTreeNode(cv::Mat roi);
		~QuadTreeNode();

		// split method
		void split(
			bool (*predicate)(const cv::Mat&),		// predicate function (false = split)
			cv::Size block_size = cv::Size(2,2)		// minimum block size
		);

		// get all leaves starting from current node
		void getLeaves(std::vector<QuadTreeNode*> & leaves);
	};


	// REGION MERGING
	// Region class
	struct Region
	{
		cv::Mat *img;					// image
		cv::Mat mask;					// binary mask (=1 for pixels within the region)
		std::set<Region*> neighbors;	// list of neighbors (std::set = unique elements)
		bool destroy;					// whether the region is scheduled for destroy
		cv::Rect rect;					// region rectangle (valid only before merging occurs)
		cv::Rect dilated_rect;			// region rectangle dilated with 3x3 SE (used for adjacency checks)

		// constructor / deconstructor
		Region(cv::Mat * _img, cv::Mat _mask, cv::Rect _rect);
		~Region();

		// merge with 'r'
		void mergeWith(Region* r);

		// return true if current region is adjacent with 'r'
		bool isAdjacent(Region* r);
	};

	// merge method
	void merge(
		std::vector <Region*> & regions,					// regions to merge (result is also stored here)
		bool (*predicate)(const cv::Mat&, Region*, Region*)	// predicate function (true = merge)
	);	
}