#pragma once

#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/core/core.hpp>
#include <list>

// open namespace "aia"
namespace aia
{
	struct QuadTreeNode
	{
		// attributes
		cv::Rect roi;			// region (quadrant)
		QuadTreeNode* children;	// up to 4 children nodes (subquadrants)
		
		// attributes for Merging
		std::list<QuadTreeNode*> neighbors;
		std::list<QuadTreeNode*> collage;

		// constructors
		QuadTreeNode();
		QuadTreeNode(cv::Rect _roi);

		// deconstructor
		~QuadTreeNode();

		// region splitting
		// first parameter: function pointer (function = predicate)
		// second parameter: region rectangle
		// third parameter: minimum region/block size
		void split( 
			bool (*predicate)(const cv::Mat & img, cv::Rect & region), 
			const cv::Mat & img,
			int min_block_size = 2
		);

		// get all regions after split
		void getLeaves(std::vector < QuadTreeNode* > & leaves);

		// detect adjacency between quadrants
		bool isAdjacent(QuadTreeNode* region);

		void merge(QuadTreeNode* region);
	};
}