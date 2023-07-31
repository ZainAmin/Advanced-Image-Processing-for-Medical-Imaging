#include <iostream>
#include "functions.h"

using namespace aia;

// constructor
QuadTreeNode::QuadTreeNode(cv::Mat roi)
{
	img_roi = roi;
}

// deconstructor (recursive solution)
QuadTreeNode::~QuadTreeNode()
{
	for(auto b : children)
		delete b;	// this also calls b deconstructor
}

// splitting method
void QuadTreeNode::split(bool (*predicate)(const cv::Mat&), cv::Size block_size)
{
	// no split if predicate is true
	// or block size is smaller than minimum block size
	if( predicate(img_roi) ||
		img_roi.rows <= block_size.height ||img_roi.cols <= block_size.width)
		return;
	// otherwise we split current block into 2x2=4 sub-blocks
	else
	{
		std::vector<range> x_blocks = ucas::partition(range(0, img_roi.cols), 2);
		std::vector<range> y_blocks = ucas::partition(range(0, img_roi.rows), 2);
		for(int y = 0; y < y_blocks.size(); y++)
			for(int x = 0; x < x_blocks.size(); x++)
				children.push_back(new QuadTreeNode (img_roi(cv::Rect(x_blocks[x].start, y_blocks[y].start, x_blocks[x].size(), y_blocks[y].size()))));

		// call split recursively on each child
		for(auto child : children)
			child->split(predicate, block_size);
	}
}

// get all leaves starting from current node
void QuadTreeNode::getLeaves(std::vector<QuadTreeNode*> & leaves)
{
	// stop when a leaf is found
	// leaf = no children
	if(children.empty())
	{
		leaves.push_back(this);
		return;	// stop recursion
	}
	else
		for(auto child : children)
			child->getLeaves(leaves);	// recursion
}







// constructor
Region::Region(cv::Mat * _img, cv::Mat _mask, cv::Rect _rect)
{
	img = _img;
	mask = _mask;
	rect = _rect;
	dilated_rect = cv::Rect(rect.x-1, rect.y-1, rect.width+2, rect.height+2);
	destroy = false;
}

// deconstructor
Region::~Region()
{
	// nothing to do
}

// merge with 'r'
void Region::mergeWith(Region* r)
{
	// do nothing if r is not set
	if(!r)
		return;

	// merge masks
	r->mask = r->mask | mask;

	// merge neighbors
	for(auto n : neighbors)
		if(n != r)
			r->neighbors.insert(n);

	// remove current region from other neighbors lists
	for(auto n : neighbors)
		n->neighbors.erase(this);

	// schedule for destroy
	destroy = true;
}


// return true if current region is adjacent with 'r'
bool Region::isAdjacent(Region* r)
{
	// trick: dilate current region, and check intersection
	// if adjacent, intersection is nonnull
	//cv::Mat dilated_mask;
	//cv::dilate(mask, dilated_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
	//return cv::countNonZero(dilated_mask & r->mask);

	return ((dilated_rect & r->rect).area() > 0);
}


// merge method
void aia::merge(
	std::vector <Region*> & regions,					// regions to merge (result is also stored here)
	bool (*predicate)(const cv::Mat&, Region*, Region*)	// predicate function (true = merge)
	)
{
	// find neighbors of each region
	ucas::Timer timer;
	for(int i=0; i<regions.size(); i++)
		for(int j=0; j<i; j++)
		{
			if(regions[i]->isAdjacent(regions[j]))
			{
				regions[i]->neighbors.insert(regions[j]);
				regions[j]->neighbors.insert(regions[i]);
			}
		}
	printf("Find neighbors = %.1f seconds\n", timer.elapsed<float>());
			
	// go on with merging until no more merging occurs
	bool merged;
	do 
	{
		timer.restart();
		// try to merge neighboring regions
		merged = false;
		for(auto r1 : regions)
		{
			for(auto r2 : r1->neighbors)
			{
				// skip if one of the two regions is scheduled for destroy
				if(r1->destroy || r2->destroy)
					continue;

				// merge if predicate function returns true
				if(predicate(*r1->img, r1, r2))
				{
					merged = true;
					r1->mergeWith(r2);
				}
			}
		}

		// destroy regions that have been merged into other regions
		for(auto it = regions.begin(); it != regions.end(); )
		{
			if((*it)->destroy)
			{
				//delete *it;
				it = regions.erase(it);
			}
			else 
				++it;
		}


		printf("Merge cycle = %.1f seconds, %d regions\n", timer.elapsed<float>(), regions.size());

		// display merging
		/*{
			printf("regions.size = %d\n", regions.size());
			cv::Mat img_copy = parent_img.clone();
			for(auto r : regions)
				img_copy.setTo(cv::Scalar(rand()%256, rand()%256, rand()%256), r->mask);
			aia::imshow("Region Merging (in progress)", img_copy, true, scaling_factor);
		}*/
	} 
	while (merged);



	//printf("regions.size = %d\n", regions.size());
}