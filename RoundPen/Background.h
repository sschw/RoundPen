#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>

namespace rp {
	class Background
	{
	public:
		Background(cv::Vec3b hsvColor) : mHsvColor(hsvColor) {
		}

		void calibrate_background_range(cv::Mat& reduced);

		cv::Vec3b get_background_color();
		cv::Vec3b get_background_color_range_low();
		cv::Vec3b get_background_color_range_high();
	private:
		cv::Vec3b mHsvColor;
		cv::Vec3b mCLow;
		cv::Vec3b mCHigh;
	};
}

