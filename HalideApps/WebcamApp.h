#pragma once

class WebcamApp
{
public:
	WebcamApp();

	int width() { return (int)cap.get(CV_CAP_PROP_FRAME_WIDTH); }
	int height() { return (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT); }
	int channels() { return 3; }

	Halide::Image<float> readFrame();
	void showImage(const Halide::Image<float>& im);

private:
	void showImage3D(const Halide::Image<float>& im);
	void showImage2D(const Halide::Image<float>& im);
	cv::VideoCapture cap;
};
