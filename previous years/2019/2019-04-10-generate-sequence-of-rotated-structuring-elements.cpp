// create 'n' rectangular Structuring Elements (SEs) at different orientations spanning the whole 360°
cv::vector<cv::Mat>						// vector of 'width' x 'width' uint8 binary images with non-black pixels being the SE
	createTiltedStructuringElements(
	int width,							// SE width (must be odd)
	int height,							// SE height (must be odd)
	int n)								// number of SEs
	throw (ucas::Error)
{
	// check preconditions
	if( width%2 == 0 )
		throw ucas::Error(ucas::strprintf("Structuring element width (%d) is not odd", width));
	if( height%2 == 0 )
		throw ucas::Error(ucas::strprintf("Structuring element height (%d) is not odd", height));

	// draw base SE along x-axis
	cv::Mat base(width, width, CV_8U, cv::Scalar(0));
	// workaround: cv::line does not work properly when thickness > 1. So we draw line by line.
	for(int k=width/2-height/2; k<=width/2+height/2; k++)
		cv::line(base, cv::Point(0,k), cv::Point(width, k), cv::Scalar(255));

	// compute rotated SEs
	cv::vector <cv::Mat> SEs;
	SEs.push_back(base);
	double angle_step = 180.0/n;
	for(int k=1; k<n; k++)
	{
		cv::Mat SE;
		cv::warpAffine(base, SE, cv::getRotationMatrix2D(cv::Point2f(base.cols/2.0f, base.rows/2.0f), k*angle_step, 1.0), cv::Size(width, width), CV_INTER_NN);
		SEs.push_back(SE);
	}

	return SEs;	 
}