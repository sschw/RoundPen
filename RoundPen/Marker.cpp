#include <iostream>
#include <stdio.h>

#include "Marker.h"

namespace rp {
	const int MINIMAL_HSV_RANGE = 3;
	const int START_H_RANGE = 10;
	const int START_S_RANGE = 40;
	const int START_V_RANGE = 40;

	void optimize_search_area_space(cv::Mat area, cv::Point offset, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Mat>& output, std::vector<cv::Point>& outOffset) {
		std::vector<cv::Rect> tempRects;
		for (int i = 0; i < contours.size(); i++) {
			cv::Rect r = cv::boundingRect(contours[i]);
			for (int j = 0; j < tempRects.size(); j++) {
				if ((r & tempRects[j]).area() > 0) {
					int xEnd = std::max(tempRects[j].x + tempRects[j].width, r.x + r.width);
					int yEnd = std::max(tempRects[j].y + tempRects[j].height, r.y + r.height);
					tempRects[j].x = std::min(r.x, tempRects[j].x);
					tempRects[j].y = std::min(r.y, tempRects[j].y);

					tempRects[j].width = xEnd - tempRects[j].x;
					tempRects[j].height = yEnd - tempRects[j].y;
				}
			}
		}
		for (int i = 0; i < tempRects.size(); i++) {
			output.push_back(area(tempRects[i]));
			outOffset.push_back(cv::Point(tempRects[i].x + offset.x, tempRects[i].y + offset.y));
		}
	}

	// If marker color is unique, threshold will only use H channel.
	// If marker saturation is unique, threshold will use HS channel.
	// If marker brightness is unique, threshold will use HSV channel.
	void Marker::calibrate_marker_range(cv::Mat& img, uint8_t frameNr) {
		// We work on tmp.
		cv::Mat tmp;
		cv::Mat rangeOut;
		std::vector<std::vector<cv::Point>> contours;
		mCLow = cv::Vec3b(FIX_SUBR_H_RANGE(mHsvColor[0] - START_H_RANGE), FIX_SUBR_SV_RANGE(mHsvColor[1], START_S_RANGE), FIX_SUBR_SV_RANGE(mHsvColor[2], START_V_RANGE));
		mCHigh = cv::Vec3b(FIX_ADD_H_RANGE(mHsvColor[0] + START_H_RANGE), FIX_ADD_SV_RANGE(mHsvColor[1], START_S_RANGE), FIX_ADD_SV_RANGE(mHsvColor[2], START_V_RANGE));
		img.copyTo(tmp);

		cv::inRange(tmp, mCLow, mCHigh, rangeOut);

		cv::findContours(rangeOut, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		std::vector<cv::Mat> contourAreas;
		std::vector<cv::Point> contourOffsets;

		optimize_search_area_space(tmp, cv::Point(0, 0), contours, contourAreas, contourOffsets);

		std::vector<std::vector<cv::Point>> contoursInsideContour;
		cv::Mat contourAreaThresh;

		int i = 0;
		// Optimize H until good enough or impossible.
		while (mCLow[0] != mHsvColor[0] - MINIMAL_HSV_RANGE && mCHigh[0] != mHsvColor[0] + MINIMAL_HSV_RANGE && (contourAreas.size() > 1 || (contourAreas.size() == 1 && contourAreas[0].rows * contourAreas[0].cols < 200))) {
			mCLow[0] = FIX_ADD_H_RANGE(mCLow[0] + 1);
			mCHigh[0] = FIX_SUBR_H_RANGE(mCHigh[0] - 1);

			for (int j = 0; j < contourAreas.size(); j++) {
				cv::inRange(contourAreas[j], mCLow, mCHigh, contourAreaThresh);

				cv::findContours(contourAreaThresh, contoursInsideContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
				optimize_search_area_space(contourAreas[j], contourOffsets[j], contoursInsideContour, contourAreas, contourOffsets);
				contourAreas.erase(contourAreas.begin() + j);
				contourOffsets.erase(contourOffsets.begin() + j);
			}

			i++;
		}

		// If no good optimization, optimize L until good enough or impossible.
		if (mCLow[0] == mHsvColor[0] - MINIMAL_HSV_RANGE || mCHigh[0] == mHsvColor[0] + MINIMAL_HSV_RANGE) {
			int i = 0;
			// Optimize H until good enough or impossible.
			while (mCLow[1] != mHsvColor[1] - MINIMAL_HSV_RANGE && mCHigh[1] != mHsvColor[1] + MINIMAL_HSV_RANGE && (contourAreas.size() > 1 || (contourAreas.size() == 1 && contourAreas[0].rows * contourAreas[0].cols < 200))) {
				mCLow[1] = mCLow[1] + 1;
				mCHigh[1] = mCHigh[1] - 1;

				for (int j = 0; j < contourAreas.size(); j++) {
					cv::inRange(contourAreas[j], mCLow, mCHigh, contourAreaThresh);

					cv::findContours(contourAreaThresh, contoursInsideContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
					optimize_search_area_space(contourAreas[j], contourOffsets[j], contoursInsideContour, contourAreas, contourOffsets);
					contourAreas.erase(contourAreas.begin() + j);
					contourOffsets.erase(contourOffsets.begin() + j);
				}

				i++;
			}

			// If no good optimization, optimize S until good enough.
			if (mCLow[1] == mHsvColor[1] - MINIMAL_HSV_RANGE || mCHigh[1] == mHsvColor[1] + MINIMAL_HSV_RANGE) {
				int i = 0;
				// Optimize H until good enough or impossible.
				while (mCLow[2] != mHsvColor[2] - MINIMAL_HSV_RANGE && mCHigh[2] != mHsvColor[2] + MINIMAL_HSV_RANGE && (contourAreas.size() > 1 || (contourAreas.size() == 1 && contourAreas[0].rows * contourAreas[0].cols < 200))) {
					mCLow[2] = mCLow[2] + 1;
					mCHigh[2] = mCHigh[2] - 1;

					for (int j = 0; j < contourAreas.size(); j++) {
						cv::inRange(contourAreas[j], mCLow, mCHigh, contourAreaThresh);

						cv::findContours(contourAreaThresh, contoursInsideContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
						optimize_search_area_space(contourAreas[j], contourOffsets[j], contoursInsideContour, contourAreas, contourOffsets);
						contourAreas.erase(contourAreas.begin() + j);
						contourOffsets.erase(contourOffsets.begin() + j);
					}

					i++;
				}
			}
		}

		if (contourAreas.size() == 1 && contourAreas[0].rows * contourAreas[0].cols < 200) {
			set_current_position(frameNr, &cv::Point2d(contourOffsets[0].x + contourAreas[0].cols / 2, contourOffsets[0].y + contourAreas[0].rows / 2));
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
		if (pos != nullptr || pos->x < 0 || pos->y < 0) {
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

	// Returns -1 if not found, 0 if found, 1 if previously tracked position still valid.
	int Marker::find_position(cv::Mat& frame, uint8_t currentFrame, cv::Point2d* position) {
		cv::Mat marker_area;
		cv::Mat marker_area_thresh;
		std::vector<std::vector<cv::Point>> cont;
		cv::Moments mu;
		int returnVal = -1;

		int lx = 0, ly = 0;
		// Decrease search area.
		cv::Point2d predicted_center;
		if (is_trackable(currentFrame)) {
			predicted_center = get_next_position(currentFrame);
			int halfW = (int)std::min(200.0, std::max(20.0, abs(predicted_center.x - mLastPosition.x)));
			int halfH = (int)std::min(200.0, std::max(20.0, abs(predicted_center.y - mLastPosition.y)));
			lx = (int)(std::max(0.0, std::min(predicted_center.x, (double)frame.cols) - halfW));
			ly = (int)(std::max(0.0, std::min(predicted_center.y, (double)frame.rows) - halfH));
			int hx = (int)(std::min((double)frame.cols-1, std::max(0.0, predicted_center.x) + halfW));
			int hy = (int)(std::min((double)frame.rows-1, std::max(0.0, predicted_center.y) + halfH));
			
			marker_area = frame(cv::Rect(lx, ly, hx-lx, hy-ly));
		}
		else {
			frame.copyTo(marker_area);
		}

		inRange(marker_area, get_marker_color_range_low(), get_marker_color_range_high(), marker_area_thresh);

		// TODO Replace it with something more efficient as on marker_area = frame will be too slow.
		//static cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		//	cv::Size(2 * 2 + 1, 2 * 2 + 1),
		//	cv::Point(2, 2));
		//cv::dilate(marker_area_thresh, marker_area_thresh, element);
		// Find the contour, using offset lx, ly to scale points back to real position.
		cv::findContours(marker_area_thresh, cont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(lx, ly));

		if (cont.size() == 0) {
			// Increase the color range so we hopefully will be able to track it again.
			if (mCHigh[0] - mCLow[0] < (START_H_RANGE << 1)) {
				mCLow[0] = FIX_SUBR_H_RANGE(mCLow[0] - 1);
				mCHigh[0] = FIX_ADD_H_RANGE(mCHigh[0] + 1);
			}
			if (mCHigh[1] - mCLow[1] < (START_S_RANGE << 1)) {
				mCLow[1] = FIX_SUBR_SV_RANGE(mCLow[1], 1);
				mCHigh[1] = FIX_ADD_SV_RANGE(mCHigh[1], 1);
			}
			if (mCHigh[2] - mCLow[2] < (START_V_RANGE << 1)) {
				mCLow[2] = FIX_SUBR_SV_RANGE(mCLow[2], 1);
				mCHigh[2] = FIX_ADD_SV_RANGE(mCHigh[2], 1);
			}
			if (predicted_center != cv::Point2d()) {
				position->x = predicted_center.x;
				position->y = predicted_center.y;
				returnVal = 1;
			}
		}
		else if (cont.size() >= 1) {
			int area;
			double contourX, contourY;
			double contourError;
			double error = INFINITY;
			cv::Mat labels = cv::Mat::zeros(marker_area.size(), CV_8UC1);

			for (int i = 0; i < cont.size(); i++) {
				area = contourArea(cont[i]);
				if (area < 200) {
					if (cont[i].size() == 1) {
						contourX = cont[i][0].x;
						contourY = cont[i][0].y;
					}
					else {
						mu = moments(cont[i]);
						if (mu.m10 < 0 || mu.m01 < 0 || mu.m00 <= 0) {
							contourX = cont[i][0].x;
							contourY = cont[i][0].y;
						}
						else {
							contourX = std::max(0.0, std::min(mu.m10 / mu.m00, (double)frame.cols));
							contourY = std::max(0.0, std::min(mu.m01 / mu.m00, (double)frame.rows));
						}
					}

					// Minimize error if we already know it. Otherwise take the one with the smallest color error.
					if (predicted_center != cv::Point2d()) {
						contourError = pow(contourX - predicted_center.x, 2) + pow(contourY - predicted_center.y, 2);
					}
					else {
						cv::drawContours(labels, cont, i, cv::Scalar(i), cv::FILLED);
						cv::Rect roi = cv::boundingRect(cont[i]);
						cv::Scalar mean = cv::mean(marker_area(roi), labels(roi) == i);
						
						contourError = pow(mean[0] - mHsvColor[0], 2) + pow(mean[1] - mHsvColor[1], 2) + pow(mean[2] - mHsvColor[2], 2);
					}
					if (contourError < error) {
						error = contourError;
						position->x = contourX;
						position->y = contourY;
						returnVal = 0;
					}
				}
			}
			//if (prevCenter != nullptr) {
			//    drawMarker(frame_points, (*prevCenter) / (((double)frame.cols) / 800), cv::Scalar(0, 0, 128), cv::MARKER_TILTED_CROSS, 5, 2);
			//}
			return returnVal;
		}
	}
	
	cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
		cv::Mat rgb;
		cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H, S, V));
		cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
		return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
	}
}

