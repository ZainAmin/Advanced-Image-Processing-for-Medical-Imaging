// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// boost library for graph processing
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edge_list.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/graph_utility.hpp>

// boost graph typedefs
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
	boost::no_property,
	boost::property<boost::edge_index_t, std::size_t> >		GraphType;
typedef boost::graph_traits<GraphType>::vertex_descriptor	VertexDescriptor;
typedef boost::graph_traits<GraphType>::edge_descriptor		EdgeDescriptor;
typedef boost::graph_traits<GraphType>::vertices_size_type	VertexIndex;
typedef boost::graph_traits<GraphType>::edges_size_type		EdgeIndex;
typedef std::pair<int, int> Edge;

// adjacent pixels' similarity metric
double similarity(int pixel1, int pixel2)
{
	double s = 10, k = 1;
	return k * std::exp(-(std::pow(pixel1 - pixel2, 2)) / (2 * s * s));
}

// graph utility function
void AddBidirectionalEdge(GraphType& graph, unsigned int source, unsigned int target, float weight,
	std::vector<EdgeDescriptor>& reverseEdges, std::vector<float>& capacity, int nextEdgeId)
{
	// Add edges between grid vertices. We have to create the edge and the reverse edge,
	// then add the reverseEdge as the corresponding reverse edge to 'edge', and then add 'edge'
	// as the corresponding reverse edge to 'reverseEdge'
	EdgeDescriptor edge = add_edge(source, target, nextEdgeId, graph).first;
	EdgeDescriptor reverseEdge = add_edge(target, source, nextEdgeId + 1, graph).first;
	reverseEdges.push_back(reverseEdge);
	reverseEdges.push_back(edge);
	capacity.push_back(weight);
	capacity.push_back(weight);
}

// UI
namespace Paint
{
	cv::Mat img;
	cv::Mat seeds_FG;
	cv::Mat seeds_BG;
	std::string win_name = "Paint";

	void updatePaint(int event, int x, int y, int, void* userdata)
	{
		// one-time initialization
		if (!seeds_FG.data)
			seeds_FG = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
		if (!seeds_BG.data)
			seeds_BG = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));

		static cv::Point prevPos(0, 0);
		static bool drawing = false;
		static bool draw_FG = false;
		static cv::Mat img_copy = img.clone();

		// alternate background/foreground seeds drawing
		if (event == cv::EVENT_LBUTTONDOWN & !drawing)
		{
			prevPos.x = x;
			prevPos.y = y;
			drawing = true;
		}
		else if (event == cv::EVENT_LBUTTONUP && drawing)
		{
			drawing = false;
			if (draw_FG == false)
				draw_FG = true;
			else
				draw_FG = false;
		}
		else if (event == cv::EVENT_MOUSEMOVE && drawing)
		{
			// display markers on top of the image
			// and draw them in the binary masks
			if (draw_FG == false)
			{
				cv::line(img_copy, prevPos, cv::Point(x, y), cv::Scalar(255, 0, 0), 12, cv::LINE_AA);
				cv::line(seeds_BG, prevPos, cv::Point(x, y), cv::Scalar(255), 12);
			}
			else
			{
				cv::line(img_copy, prevPos, cv::Point(x, y), cv::Scalar(0, 0, 255), 12, cv::LINE_AA);
				cv::line(seeds_FG, prevPos, cv::Point(x, y), cv::Scalar(255), 12);
			}

			// previous pos update
			prevPos.x = x;
			prevPos.y = y;
		}

		cv::imshow(win_name, img_copy);
	}

}

int main()
{
	try
	{
		// load image
		Paint::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/girl.png");
		if (!Paint::img.data)
			throw aia::error("Cannot open image");

		// launch UI for seeds drawing
		cv::namedWindow(Paint::win_name);
		cv::setMouseCallback(Paint::win_name, Paint::updatePaint);
		Paint::updatePaint(0, 0, 0, 0, 0);
		cv::waitKey(0);

		// show seeds
		ucas::imshow("Seeds (foreground)", Paint::seeds_FG);
		ucas::imshow("Seeds (background)", Paint::seeds_BG);

		// bitdepth of the image 
		int L = 256;

		// definition of definitive image
		cv::Mat img_support = Paint::img.clone();
		cv::Mat img_result(Paint::img.rows, Paint::img.cols, CV_8UC(3), cv::Scalar(0, 0, 0));

		// grayscale conversion
		cv::cvtColor(Paint::img, Paint::img, cv::COLOR_BGR2GRAY);

		// show seeds histograms
		ucas::imshow("Foreground histogram", ucas::imhist(Paint::img, Paint::seeds_FG));
		ucas::imshow("Background histogram", ucas::imhist(Paint::img, Paint::seeds_BG));

		// normalize seeds histograms -> seeds pdfs
		std::vector<int> hist_FG = ucas::histogram_mask(Paint::img, Paint::seeds_FG);
		std::vector<int> hist_BG = ucas::histogram_mask(Paint::img, Paint::seeds_BG);
		int pixel_count_FG = 0;
		int pixel_count_BG = 0;
		for (int i = 0; i < L; i++)
		{
			pixel_count_FG += hist_FG[i];
			pixel_count_BG += hist_BG[i];
		}
		std::vector<double> pdf_FG(L);
		std::vector<double> pdf_BG(L);
		for (int i = 0; i < L; i++)
		{
			pdf_FG[i] = hist_FG[i] / double(pixel_count_FG);
			pdf_BG[i] = hist_BG[i] / double(pixel_count_BG);
		}


		// graph definition
		int height = Paint::img.rows;
		int width = Paint::img.cols;
		float s = height * width;		// source index
		float t = height * width + 1;    // sink index
		GraphType graph;
		std::list<float> weights;
		unsigned int numberOfVertices = (height * width) + 2;
		std::vector<int> groups(numberOfVertices);
		std::vector<EdgeDescriptor> reverseEdges;
		std::vector<float> capacity;

		// parameters
		float lambda = 0.0001;
		double k = -ucas::inf<double>();

		// insert n-links
		int counter = 0;
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				double sum = 0;

				for (int i = -1; i < 2; i++)
				{
					for (int j = -1; j < 2; j++)
					{
						if (x + j >= width || y + i >= height) // out of range
							continue;
						if (x + j < 0 || y + i < 0)			   // out of range
							continue;
						if (i == 0 && j == 0) // itself
							continue;

						// connections between adjacent pixels
						uchar pixel1 = Paint::img.at<uchar>((y * width + x));
						uchar pixel2 = Paint::img.at<uchar>((y + i) * width + x + j);
						weights.push_back(similarity(pixel1, pixel2));
						sum += weights.back();

						AddBidirectionalEdge(graph, (y * width + x), ((y + i) * width + x + j)
							, weights.back(), reverseEdges, capacity, counter);

						counter += 2;
					}
				}

				k = std::max(k, sum);

			}
		}

		// insert t-links
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				int seed_FG = (int)Paint::seeds_FG.at<uchar>(y * width + x);
				int seed_BG = (int)Paint::seeds_BG.at<uchar>(y * width + x);

				if (seed_FG)
				{
					weights.push_back(1 + k);        // {p, s}
					AddBidirectionalEdge(graph, (y * width + x), s
						, weights.back(), reverseEdges, capacity, counter);
					counter += 2;

					weights.push_back(0);          // {p, t}
					AddBidirectionalEdge(graph, (y * width + x), t
						, weights.back(), reverseEdges, capacity, counter);
					counter += 2;
				}

				else if (seed_BG)
				{
					weights.push_back(0);             // {p, s}
					AddBidirectionalEdge(graph, (y * width + x), s
						, weights.back(), reverseEdges, capacity, counter);
					counter += 2;

					weights.push_back(1 + k);           // {p, t}
					AddBidirectionalEdge(graph, (y * width + x), t
						, weights.back(), reverseEdges, capacity, counter);
					counter += 2;
				}

				else
				{
					int Ip = (int)Paint::img.at<uchar>((y * width + x));

					weights.push_back(lambda * (-log(pdf_BG[Ip] ? pdf_BG[Ip] : 10e-10)));        // {p, s}
					AddBidirectionalEdge(graph, (y * width + x), s
						, weights.back(), reverseEdges, capacity, counter);
					counter += 2;

					weights.push_back(lambda * (-log(pdf_FG[Ip] ? pdf_FG[Ip] : 10e-10)));
					AddBidirectionalEdge(graph, (y * width + x), t
						, weights.back(), reverseEdges, capacity, counter);
					counter += 2;
				}

			}
		}


		std::cout << "Number of vertices " << num_vertices(graph) << std::endl;
		std::cout << "Number of edges " << num_edges(graph) << std::endl;

		// min-cut (max-flow) algorithm
		std::vector<float> residual_capacity(num_edges(graph), 0);
		VertexDescriptor sourceVertex = vertex(s, graph);
		VertexDescriptor sinkVertex = vertex(t, graph);
		boost::boykov_kolmogorov_max_flow(graph,
			boost::make_iterator_property_map(&capacity[0], get(boost::edge_index, graph)),
			boost::make_iterator_property_map(&residual_capacity[0], get(boost::edge_index, graph)),
			boost::make_iterator_property_map(&reverseEdges[0], get(boost::edge_index, graph)),
			boost::make_iterator_property_map(&groups[0], get(boost::vertex_index, graph)),
			get(boost::vertex_index, graph),
			sourceVertex,
			sinkVertex);


		// display the segmentation
		std::cout << "Source group label " << groups[sourceVertex] << std::endl;
		std::cout << "Sink group label " << groups[sinkVertex] << std::endl;
		for (size_t index = 0; index < numberOfVertices - 2; ++index)
		{
			if (groups[index] == groups[sourceVertex])
				img_result.at<cv::Vec3b>(index) = img_support.at<cv::Vec3b>(index);
			else if (groups[index] == groups[sinkVertex])
				img_result.at<cv::Vec3b>(index) = cv::Vec3b(255, 0, 0);
			else
				img_result.at<cv::Vec3b>(index) = cv::Vec3b(150, 150, 150);
		}

		ucas::imshow("Result", img_result);

		return EXIT_SUCCESS;
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

