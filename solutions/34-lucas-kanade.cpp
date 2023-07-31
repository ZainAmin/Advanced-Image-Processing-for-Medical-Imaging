// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv image processing module
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace aia
{
	// global variables
	std::vector<cv::Point2f> prev_points;		// points to track in the previous image
	cv::Mat prev_img;							// previous image
	cv::Mat traj_img;							// trajectory image

	// single-frame processor
	cv::Mat track_moving_objects(const cv::Mat& frame);
}


int main()
{
	// launch the system
	std::string path = std::string(EXAMPLE_IMAGES_PATH) + "/traffic1.avi";
	//std::string path = "";
	aia::processVideoStream(path, aia::track_moving_objects, "", false);

	return EXIT_SUCCESS;
}


// utility functions
bool isInvalid(cv::Point2f p) {
	return p.x == aia::inf<float>() || p.y == aia::inf<float>();
}

void invalidate(cv::Point2f& p) {
	p.x = p.y = aia::inf<float>();
}

bool isWithinImage(cv::Point2f& p, const cv::Mat& img) {
	return p.x > 0 && p.x < img.cols&& p.y > 0 && p.y < img.rows;
}

float distance(cv::Point2f& p1, cv::Point2f& p2) {
	return ucas::distance<float>(p1.x, p1.y, p2.x, p2.y);
}

void alphaColorBlending(cv::Mat& img, cv::Scalar color, cv::Mat& alpha) {
	// @TODO: check preconditions 
	for (int y = 0; y < img.rows; y++)
	{
		cv::Vec3b* img_data = img.ptr<cv::Vec3b>(y);
		unsigned char* alpha_data = alpha.ptr<unsigned char>(y);
		for (int x = 0; x < img.cols; x++)
			if (alpha_data[x])
			{
				float w2 = alpha_data[x] / 255.0;
				float w1 = 1 - w2;
				for (int c = 0; c < 3; c++)
					img_data[x][c] = cv::saturate_cast<unsigned char>(w1 * img_data[x][c] + w2 * color[c]);
			}
	}
}


// single-frame processor
cv::Mat aia::track_moving_objects(const cv::Mat& frame)
{
	// allocate trajectory image once for all
	if (!traj_img.data)
		traj_img = cv::Mat(frame.rows, frame.cols, CV_8U, cv::Scalar(0));

	// prepare output image (we will draw the result on top if it)
	cv::Mat output = frame.clone();

	// convert frame to grayscale
	cv::Mat curr_img;
	cv::cvtColor(frame, curr_img, cv::COLOR_BGR2GRAY);

	// if there are points to track, apply Lucas-Kanade on each of them
	if (prev_points.size())
	{
		// apply Lucas-Kanade
		std::vector <cv::Point2f> curr_points;
		std::vector<uchar> status;
		std::vector<float> err;
		cv::calcOpticalFlowPyrLK(prev_img, curr_img, prev_points, curr_points, status, err, cv::Size(21, 21), 5);

		// remove point pairs not matching the criteria
		for (int i = 0; i < prev_points.size(); i++)
			if (!status[i] ||	// tracking has failed
				!isWithinImage(curr_points[i], frame) ||	// point outside image
				distance(prev_points[i], curr_points[i]) < 1 || // point is not moving
				distance(prev_points[i], curr_points[i]) > 100)// too fast: it must be an error
			{
				invalidate(prev_points[i]);
				invalidate(curr_points[i]);
			}
		prev_points.erase(std::remove_if(prev_points.begin(), prev_points.end(), isInvalid), prev_points.end());
		curr_points.erase(std::remove_if(curr_points.begin(), curr_points.end(), isInvalid), curr_points.end());

		// draw trajectories between point pairs
		for (int i = 0; i < prev_points.size(); i++)
		{
			cv::line(traj_img, prev_points[i], curr_points[i], cv::Scalar(255), 2, cv::LINE_AA);
			cv::circle(traj_img, curr_points[i], 3, cv::Scalar(255), -1, cv::LINE_AA);
		}

		// update previous points
		for (int i = 0; i < prev_points.size(); i++)
			prev_points[i] = curr_points[i];
	}

	// if there are too few points, we extract new keypoints with Harris
	if (prev_points.size() < 30)
	{
		std::vector <cv::Point2f> new_points;
		cv::goodFeaturesToTrack(curr_img, new_points, 100, 0.1, 5, cv::noArray(), 3, true);
		prev_points.insert(prev_points.end(), new_points.begin(), new_points.end());
	}

	// update previous image with current image data
	prev_img = curr_img;

	// aging of trajectories
	traj_img -= 30;

	// overlay trajectories
	alphaColorBlending(output, cv::Scalar(0, 255, 255), traj_img);

	return output;
}
