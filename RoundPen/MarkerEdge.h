#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>

// Include Markers. 
#include "Marker.h"

namespace rp {
	// Marker edges connect two markers over a distance.
	class MarkerEdge {
	public:
		Marker& m1;
		Marker& m2;
		double maxDistance;

		void add_to_markers();
		Marker* get_neighbor(Marker& m);
		bool valid_marker_positions(cv::Point2d p1, cv::Point2d p2);
	};
}
