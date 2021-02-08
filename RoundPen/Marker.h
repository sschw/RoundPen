#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>

namespace rp {
#define FIX_SUBR_H_RANGE(x) (((x) > 180) ? x - 76 : x)
#define FIX_ADD_H_RANGE(x) (((x) > 180) ? x + 76 : x)
#define FIX_SUBR_SV_RANGE(x, y) ((x < y) ? 0 : x - y)
#define FIX_ADD_SV_RANGE(x, y) ((x > (255 - y)) ? 255 : x + y)
	// We use MarkerEdges but Marker is included in MarkerEdge.
	class MarkerEdge;

	cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V);

	class Marker {
	public:
		Marker(cv::String name, cv::Vec3b hsvColor, bool alwaysVisible = false, uint8_t framesUntilLost = 5): mName(name), mHsvColor(hsvColor), mAlwaysVisible(alwaysVisible), mFramesUntilLost(framesUntilLost), mMarkerRadius(1), mTracked(false) {
			// Define minimal and maximal range.
			mCLow = cv::Vec3b(FIX_SUBR_H_RANGE(mHsvColor[0] - 10), FIX_SUBR_SV_RANGE(mHsvColor[1], 40), FIX_SUBR_SV_RANGE(mHsvColor[2], 40));
			mCHigh = cv::Vec3b(FIX_ADD_H_RANGE(mHsvColor[0] + 10), FIX_ADD_SV_RANGE(mHsvColor[1], 40), FIX_ADD_SV_RANGE(mHsvColor[2], 40));
		}

		void calibrate_marker_range(cv::Mat& img, uint8_t frameNr);
		void append_neighbor(MarkerEdge* edge);
		cv::Vec3b get_marker_color();
		cv::Vec3b get_marker_color_range_low();
		cv::Vec3b get_marker_color_range_high();
		cv::Point2d get_last_position();
		bool is_trackable(int currentFrame);
		bool is_tracked_for_frame(uint8_t currentFrame) { return currentFrame - mLastFrames[0] == 0; }
		cv::Point2d get_next_position(uint8_t currentFrame);
		void set_current_position(uint8_t currentFrame, cv::Point2d* pos);
		cv::String get_name() { return mName; }
		double get_marker_radius() { return mMarkerRadius; }

		int find_position(cv::Mat& frame, uint8_t currentFrame, cv::Point2d* position);

	private:
		cv::String mName;
		cv::Vec3b mHsvColor;
		cv::Vec3b mCLow;
		cv::Vec3b mCHigh;
		double mMarkerRadius;
		bool mAlwaysVisible;
		std::vector<MarkerEdge*> mNeighbors;

		cv::Point2d mLastPositions[5];
		uint8_t mLastFrames[5];
		int mLastPositionsNum = 0;
		cv::Point2d mLastVelocity;
		cv::Point2d mLastAcceleration;

		uint8_t mFramesUntilLost;

		bool mTracked;
	};
}