#include <iostream>
#include <stdio.h>

#include "Marker.h"

namespace rp {
	const int MINIMAL_HSV_RANGE = 3;
	const int START_H_RANGE = 10;
	const int START_S_RANGE = 40;
	const int START_V_RANGE = 40;

	// If marker color is unique, threshold will only use H channel.
	// If marker saturation is unique, threshold will use HS channel.
	// If marker brightness is unique, threshold will use HSV channel.
	void Marker::calibrate_marker_range(cv::Mat& img) {
		// We work on tmp.
		cv::Mat tmp;
		cv::Mat rangeOut;
		img.copyTo(tmp);
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		mCLow = cv::Vec3b(FIX_SUBR_H_RANGE(mHsvColor[0] - START_H_RANGE), FIX_SUBR_SV_RANGE(mHsvColor[1], START_S_RANGE), FIX_SUBR_SV_RANGE(mHsvColor[2], START_V_RANGE));
		mCHigh = cv::Vec3b(FIX_ADD_H_RANGE(mHsvColor[0] + START_H_RANGE), FIX_ADD_SV_RANGE(mHsvColor[1], START_S_RANGE), FIX_ADD_SV_RANGE(mHsvColor[2], START_V_RANGE));

		int i = 0;
		// Optimize H until good enough or impossible.
		do  {
			mCLow[0] = FIX_ADD_H_RANGE(mCLow[0] + 1);
			mCHigh[0] = FIX_SUBR_H_RANGE(mCHigh[0] - 1);

			cv::inRange(tmp, mCLow, mCHigh, rangeOut);

			cv::findContours(rangeOut, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

			i++;
		} while (mCLow[0] != mHsvColor[0] - MINIMAL_HSV_RANGE && mCHigh[0] != mHsvColor[0] + MINIMAL_HSV_RANGE && (contours.size() != 1 || cv::contourArea(contours[0]) < 200));

		// If no good optimization, optimize L until good enough or impossible.
		if (mCLow[0] == mHsvColor[0] - MINIMAL_HSV_RANGE || mCHigh[0] == mHsvColor[0] + MINIMAL_HSV_RANGE) {
			i = 0;
			do {
				mCLow[1] = mCLow[1] + 1;
				mCHigh[1] = mCHigh[1] - 1;

				cv::inRange(tmp, mCLow, mCHigh, rangeOut);

				cv::findContours(rangeOut, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

				i++;
			} while (mCLow[1] != mHsvColor[1] - MINIMAL_HSV_RANGE && mCHigh[1] != mHsvColor[1] + MINIMAL_HSV_RANGE && (contours.size() != 1 || cv::contourArea(contours[0]) < 200));

			// If no good optimization, optimize S until good enough.
			if (mCLow[1] == mHsvColor[1] - MINIMAL_HSV_RANGE || mCHigh[1] == mHsvColor[1] + MINIMAL_HSV_RANGE) {
				i = 0;
				do {
					mCLow[2] = mCLow[2] + 1;
					mCHigh[2] = mCHigh[2] - 1;

					cv::inRange(tmp, mCLow, mCHigh, rangeOut);

					cv::findContours(rangeOut, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

					i++;
				} while (mCLow[2] != mHsvColor[2] - MINIMAL_HSV_RANGE && mCHigh[2] != mHsvColor[2] + MINIMAL_HSV_RANGE && (contours.size() != 1 || cv::contourArea(contours[0]) < 200));
			}
		}

		// Range optimized on first frame.
		// Program itself has better detection as too large areas will just be ignored.
	}

	void Marker::append_neighbor(MarkerEdge* edge) {
		mNeighbors.push_back(edge);
	}

	cv::Vec3b Marker::get_marker_color() {
		return mHsvColor;
	}

	cv::Vec3b Marker::get_marker_color_range_low() {
		return mCLow;
	}

	cv::Vec3b Marker::get_marker_color_range_high() {
		return mCHigh;
	}

	cv::Point2d Marker::get_last_position() {
		return mLastPosition;
	}

	bool Marker::is_trackable(int currentFrame) {
		mTracked = mTracked && ((currentFrame - mLastFrame) < mFramesUntilLost);
		if (!mTracked) {
			mLastAcceleration = cv::Point2d(0, 0);
			mLastVelocity = cv::Point2d(0, 0);
		}
		return mTracked;
	}

	cv::Point2d Marker::get_next_position(uint8_t currentFrame) {
		uint8_t frameDiff = currentFrame - mLastFrame;
		return mLastPosition + (mLastVelocity * frameDiff);// +(mLastAcceleration * frameDiff * frameDiff / 2);
	}

	void Marker::set_current_position(uint8_t currentFrame, cv::Point2d* pos) {
		if (pos != nullptr) {
			uint8_t frameDiff = currentFrame - mLastFrame;
			mTracked = true;
			if (mLastPosition.x != 0 && mLastPosition.y != 0) {
				cv::Point2d newVelocity = (*pos - mLastPosition) / frameDiff;
				newVelocity.x = std::min(newVelocity.x, 15.0); // Max x velocity -> px per frame
				newVelocity.y = std::min(newVelocity.y, 15.0); // Max x velocity -> px per frame
				// If we have a previous velocity, we calc the acceleration. 
				// First acceleration after tracking is too high.
				if (mLastVelocity.x != 0 || mLastVelocity.y != 0) {
					mLastAcceleration = 2 * (newVelocity - mLastVelocity) / frameDiff;
				}
				else {
					mLastAcceleration = cv::Point2d(0, 0);
				}
				mLastVelocity = newVelocity;
			}
			mLastPosition = *pos;
			mLastFrame = currentFrame;
		}
	}

	void Marker::find_position(cv::Mat& frame, uint8_t currentFrame) {
		cv::Mat marker_area;
		cv::Mat marker_area_thresh;

		// Decrease search area.
		cv::Point2d predicted_center;
		if (is_trackable(currentFrame)) {
			predicted_center = get_next_position(currentFrame);
			int halfW = (int)std::min(200.0, std::max(20.0, abs(predicted_center.x - mLastPosition.x)));
			int halfH = (int)std::min(200.0, std::max(20.0, abs(predicted_center.y - mLastPosition.y)));
			int lx = (int)(std::max(0.0, std::min(predicted_center.x, (double)frame.cols) - halfW));
			int ly = (int)(std::max(0.0, std::min(predicted_center.y, (double)frame.rows) - halfH));
			int hx = (int)(std::min((double)frame.cols-1, std::max(0.0, predicted_center.x) + halfW));
			int hy = (int)(std::min((double)frame.rows-1, std::max(0.0, predicted_center.y) + halfH));
			
			marker_area = frame(cv::Rect(lx, ly, hx-lx, hy-ly));
		}
		else {
			frame.copyTo(marker_area);
		}

		inRange(marker_area, get_marker_color_range_low(), get_marker_color_range_high(), marker_area_thresh);
	}
	
	cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
		cv::Mat rgb;
		cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H, S, V));
		cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
		return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
	}
}

