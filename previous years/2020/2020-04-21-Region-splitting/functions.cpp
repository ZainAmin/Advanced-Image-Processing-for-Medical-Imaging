#include <iostream>
#include "functions.h"

using namespace aia;

QuadTreeNode::QuadTreeNode()
{
	children = 0;
}
QuadTreeNode::QuadTreeNode(cv::Rect _roi)
{
	roi = _roi;
	children = 0;
}

// deconstructor
QuadTreeNode::~QuadTreeNode()
{
	if(children)
		delete[] children;
}

void QuadTreeNode::split( 
	bool (*predicate)(const cv::Mat & img, cv::Rect & region), 
	const cv::Mat & img,
	int min_block_size
	)
{
	// block size limit
	if(roi.width <= min_block_size && roi.height <= min_block_size)
		return;

	bool hasToSplit = predicate(img, this->roi);
	if(hasToSplit)
	{
		children = new QuadTreeNode[4];
		
		for(int i=0; i<2; i++)
			for(int j=0; j<2; j++)
			{
				cv::Rect new_roi;
				new_roi.x = this->roi.x + j*(roi.width/2);
				new_roi.y = this->roi.y + i*(roi.height/2);
				new_roi.width  = (j == 0 ? roi.width/2  : roi.width  - roi.width/2);
				new_roi.height = (i == 0 ? roi.height/2 : roi.height - roi.height/2);
				
				children[i*2+j] = QuadTreeNode(new_roi);
				children[i*2+j].split(predicate, img);
			}
	}
}

void QuadTreeNode::getLeaves(std::vector < QuadTreeNode* > & leaves)
{
	if(children)
	{
		for(int i=0; i<4; i++)
			children[i].getLeaves(leaves);
	}
	else
		leaves.push_back(this);
}