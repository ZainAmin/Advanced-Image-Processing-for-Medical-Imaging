// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: Magic-Wand-like functionality with Marker-controlled Watershed segmentation
namespace aia
{
	// since we work with a GUI, we need parameters (and the images) to be stored in global variables
	cv::Mat img;
	// nothing more to be added here...there are no parameters!!!

	// we will need a simple struct to store rgb triplets
	struct rgb{
		uint8 r, g, b;
		rgb() : r(0), g(0), b(0){}
		rgb(uint8 _r, uint8 _g, uint8 _b) : r(_r), b(_b), g(_g){}
	};

	// NOTE: this is a mouse callback function we will link to the GUI
	// see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html#setmousecallback
	void magicWand(
		int event,					// mouse event (e.g. mouse pressed, mouse released, mouse double clicked, ...)
		int x,						// mouse x-coordinate
		int y,						// mouse y-coordinate
		int flags, void* userdata)	// not used
	{
		// We need to store persistent metadata across all mouse interactions / updates of the GUI
		// Persistent variables = static variables in C

		// whether the user is drawing a curve (initially set to false)
		static bool is_drawing = false;

		// number of curves drawn (initially 0), will be incremented at each mouse stroke
		static int curve_count = 0;

		// watershed markers image (initially empty), will be populated with markers inputted with mouse strokes
		static cv::Mat inputted_markers(img.rows, img.cols, CV_32S, cv::Scalar(0));

		// the image displayed in the GUI (initially it is the original image)
		static cv::Mat img_displayed = img.clone();

		// the colored segmented regions displayed in the GUI (initially black)
		static cv::Mat regions_displayed(img.rows, img.cols, CV_8UC3, cv::Scalar(0,0,0));

		// the previous point in the curve points sequence, will be updated as long as the drawing proceeds
		static cv::Point prev(0,0);


		// we are now going to check the mouse event
		// mouse event = mouse left button pressed
		if  ( event == cv::EVENT_LBUTTONDOWN )
		{
			is_drawing = true;		// start drawing
			curve_count++;			// add a new curve
			prev = cv::Point(x,y);	// update previous point (this is the starting point)
		}
		// mouse event = mouse left button released
		else if  ( event == cv::EVENT_LBUTTONUP )
		{
			is_drawing = false;	// stop drawing

			// apply watershed
			// NOTE: we make a copy of the marker' because 'watershed(...)' stores the result in the given markers cv::Mat
			// and we do not want to modify the original user-inputted markers
			cv::Mat watershed_markers = inputted_markers.clone();
			cv::watershed(img, watershed_markers);

			// we want to display the segmented regions with distinct colors
			// an easy (but not 100% accurate) way is to assign each object label a random color
			std::map <int, rgb> object2colors;
			for(int i=1; i<=curve_count; i++)
				object2colors[i] = rgb(rand()%256, rand()%256, rand()%256);
			for(int y=0; y<regions_displayed.rows; y++)
			{
				uint8 *ythSegRow =  regions_displayed.ptr<uint8>(y);
				int   *ythWatRow =  watershed_markers.ptr<int>(y);

				for(int x=0; x<regions_displayed.cols; x++)
				{
					ythSegRow[3*x + 0] = object2colors[ ythWatRow[x] ].b;
					ythSegRow[3*x + 1] = object2colors[ ythWatRow[x] ].g;
					ythSegRow[3*x + 2] = object2colors[ ythWatRow[x] ].r;
				}
			}


			// we are now interested to contours overimposed on the original image,
			// so we can do some simple reasoning on how to rescale the values of 'markers' in the 8-bit range [0, 255]
			// pixels whose value is -1   : dams  -------------------->  -1*255+255 = 0
			// pixels whose value is >= 0 : objects and background --->   x*255+255 = 255 (saturation)
			watershed_markers.convertTo(watershed_markers, CV_8U, 255, 255);
			// image is now 'all white' except for the dams ('black') --> we have a binary image!

			// we can now find the contours on the inverted binary image
			std::vector < std::vector <cv::Point> > segmented_objects;
			watershed_markers = 255-watershed_markers;
			cv::findContours(watershed_markers, segmented_objects, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

			// and overimpose them on the original image (we use yellow color)
			cv::drawContours(img_displayed, segmented_objects, -1, cv::Scalar(0,255,255), 2, CV_AA);
		}
		// mouse event : mouse is moving
		else if  ( event == cv::EVENT_MOUSEMOVE )
		{
			// if drawing is active
			if(is_drawing)
			{
				// we add a new point into the 'inputted_marker' using the 'curve_count' as the marker label
				inputted_markers.at<int>(y,x) = curve_count;

				// we draw a segment between the previous and the current point
				cv::line(img_displayed, prev, cv::Point(x,y), cv::Scalar(0,0,255), 2);

				// and update the previous point with the current point
				prev = cv::Point(x,y);
			}
		}

		// update the display
		cv::imshow("Magic Wand", img_displayed);
		cv::imshow("Segmented regions", regions_displayed);
	}
}


int main() 
{
	try
	{
		// load an image
		aia::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg");
		if(!aia::img.data)
			throw aia::error("Cannot open image");

		// create a GUI window
		cv::namedWindow("Magic Wand");
		cv::namedWindow("Segmented regions");

		// ...and set a mouse callback
		cv::setMouseCallback("Magic Wand", aia::magicWand);

		// display original image
		cv::imshow("Magic Wand", aia::img);

		// wait for key press = windows stay opened until the user presses any key
		cv::waitKey(0);

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