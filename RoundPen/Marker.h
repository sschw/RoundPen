#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>

namespace rp {
	// We use MarkerEdges but Marker is included in MarkerEdge.
	class MarkerEdge;

	class Marker {
	public:
		Marker(cv::String name, cv::Vec3b hsvColor, bool alwaysVisible = false, uint8_t framesUntilLost = 90): mName(name), mHsvColor(hsvColor), mAlwaysVisible(alwaysVisible), mFramesUntilLost(framesUntilLost), mLastFrame(255), mTracked(false) {
		}

		void calibrate_marker_range(cv::Mat& img);
		void append_neighbor(MarkerEdge* edge);
		cv::Vec3b get_marker_color();
		cv::Vec3b get_marker_color_range_low();
		cv::Vec3b get_marker_color_range_high();
		bool is_trackable();
		cv::Point2d get_next_position(uint8_t currentFrame);
		void set_current_position(uint8_t currentFrame, cv::Point2d* pos);

	private:
		cv::String mName;
		cv::Vec3b mHsvColor;
		cv::Vec3b mCLow;
		cv::Vec3b mCHigh;
		bool mAlwaysVisible;
		std::vector<MarkerEdge*> mNeighbors;

		cv::Point2d mLastPosition;
		cv::Point2d mLastVelocity;
		cv::Point2d mLastAcceleration;

		uint8_t mFramesUntilLost;
		uint8_t mLastFrame;

		bool mTracked;
	};
}