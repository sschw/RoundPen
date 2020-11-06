#include "MarkerEdge.h"

namespace rp {
	void MarkerEdge::add_to_markers() {
		m1.append_neighbor(this);
		m2.append_neighbor(this);
	}

	Marker* MarkerEdge::get_neighbor(Marker& m) {
		// Check pointing to same instance.
		if (&m == &m1)
			return &m2;
		else if (&m == &m2)
			return &m1;
		return nullptr;
	}

	bool MarkerEdge::valid_marker_positions(cv::Point2d p1, cv::Point2d p2) {
		double distX = p1.x - p2.x;
		double distY = p1.y - p2.y;

		return distX * distX + distY * distY < maxDistance * maxDistance;
	}
}