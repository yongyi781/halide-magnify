#pragma once

class WebcamApp
{
public:
	WebcamApp(std::string name = "Webcam app");

	int width() { return (int)cap.get(CV_CAP_PROP_FRAME_WIDTH); }
	int height() { return (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT); }
	int channels() { return 3; }

	Halide::Image<float> readFrame();

private:

	std::string name;
	cv::VideoCapture cap;
};
