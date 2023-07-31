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

				//// top-left element adjacency relationships
				//if(i==0 && j == 0)
				//{
				//	children[0].neighbors.push_back(&children[1]);	// top-right
				//	children[0].neighbors.push_back(&children[2]);	// bottom-left

				//	// scan all neighbors of the current node (parent)
				//	// and add adjacent nodes to the neighbors lists
				//	for(auto neighbor : neighbors)
				//		if( children[0].isAdjacent(neighbor) ) // it is adjacent
				//		{
				//			children[0].neighbors.push_back(neighbor);
				//			neighbor->neighbors.push_back(&children[0]);
				//		}
				//}
				//// top-right element adjacency relationships
				//else if(i==0 && j == 1)
				//{
				//	children[1].neighbors.push_back(&children[0]);	// top-left
				//	children[1].neighbors.push_back(&children[3]);	// bottom-right
				//}
				//// bottom-right element adjacency relationships
				//else if(i==1 && j == 1)
				//{
				//	children[3].neighbors.push_back(&children[1]);	// top-right
				//	children[3].neighbors.push_back(&children[2]);	// bottom-left
				//}
				//// bottom-left element adjacency relationships
				//else
				//{
				//	children[2].neighbors.push_back(&children[0]);	// top-left
				//	children[2].neighbors.push_back(&children[3]);	// bottom-right
				//}
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

bool QuadTreeNode::isAdjacent(QuadTreeNode* region)
{
	// add one row(column) to the top(left) and one row(column) to the bottom(right)
	// this makes the roi 1 pixel larger in the vertical direction
	// then intersection between enlarged roi and region->roi is not null if they are adjacent

	roi.y -= 1;
	roi.height += 2;
	bool vertical_adjacency = (roi & region->roi).area() > 0;
	roi.y += 1;
	roi.height -= 2;

	roi.x -= 1;
	roi.width += 2;
	bool horizontal_adjacency = (roi & region->roi).area() > 0;
	roi.x += 1;
	roi.width -= 2;

	return vertical_adjacency || horizontal_adjacency;
}

void QuadTreeNode::merge(QuadTreeNode* region)
{

}