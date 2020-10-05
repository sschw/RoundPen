#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>

namespace rp {
	class Marker {
	public:
		Marker(cv::String name, cv::Vec3b hsvColor, bool alwaysVisible = false, uint8_t framesUntilLost = 90): mName(name), mHsvColor(hsvColor), mAlwaysVisible(alwaysVisible), mFramesUntilLost(framesUntilLost), mLastFrame(255), mTracked(false) {
		}

		void append_neighbor(MarkerEdge& edge) {
			mNeighbors.push_back(edge);
		}

		cv::Vec3b get_marker_color() {
			return mHsvColor;
		}

		bool is_trackable() {
			return mTracked;
		}

		cv::Point2d get_next_position(uint8_t currentFrame) {
			uint8_t frameDiff = currentFrame - mLastFrame;
			return mLastPosition + (mLastVelocity * frameDiff) + (mLastAcceleration * frameDiff * frameDiff / 2);
		}

		void set_current_position(uint8_t currentFrame, cv::Point2d* pos) {
			if (pos != nullptr) {
				uint8_t frameDiff = currentFrame - mLastFrame;
				mTracked = true;
				cv::Point2d newVelocity = (*pos - mLastPosition) / frameDiff;
				// If we have a previous velocity, we calc the acceleration. 
				// First acceleration after tracking is too high.
				if (mLastVelocity.x != 0 || mLastVelocity.y != 0) {
					mLastAcceleration = 2 * (newVelocity - mLastVelocity) / frameDiff;
				}
				else {
					mLastAcceleration = cv::Point2d(0, 0);
				}
				mLastVelocity = newVelocity;
				mLastPosition = *pos;
			}
			else {
				mTracked = mTracked && ((currentFrame - mLastFrame) < mFramesUntilLost);
				if (!mTracked) {
					mLastAcceleration = cv::Point2d(0, 0);
					mLastVelocity = cv::Point2d(0, 0);
				}
			}
		}

	private:
		cv::String mName;
		cv::Vec3b mHsvColor;
		bool mAlwaysVisible;
		std::vector<MarkerEdge&> mNeighbors;

		cv::Point2d mLastPosition;
		cv::Point2d mLastVelocity;
		cv::Point2d mLastAcceleration;

		uint8_t mFramesUntilLost;
		uint8_t mLastFrame;

		bool mTracked;
	};

	struct MarkerEdge {
		Marker& m1;
		Marker& m2;
		double maxDistance;

		void add_to_markers() {
			m1.append_neighbor(*this);
			m2.append_neighbor(*this);
		}

		Marker* get_neighbor(Marker& m) {
			// Check pointing to same instance.
			if (&m == &m1)
				return &m2;
			else if (&m == &m2)
				return &m1;
			return nullptr;
		}

		bool valid_marker_positions(cv::Point2d p1, cv::Point2d p2) {
			double distX = p1.x - p2.x;
			double distY = p1.y - p2.y;
			
			return distX * distX + distY * distY < maxDistance * maxDistance;
		}
	};
}