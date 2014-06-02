#include "stdafx.h"
#include "WebcamApp.h"

using namespace Halide;

WebcamApp::WebcamApp(std::string name) : name(name), cap(0)
{
	if (!cap.isOpened())
		throw std::exception("Cannot open webcam.");
	cv::namedWindow(name);
}

Image<float> WebcamApp::readFrame()
{
	static Func convert;
	static ImageParam ip(UInt(8), 3);
	static Var x, y, c;

	if (!convert.defined())
	{
		convert(x, y, c) = ip(c, x, y) / 255.0f;
		convert.vectorize(x, 4).parallel(y, 4);
	}

	cv::Mat frame;
	cap >> frame;
	ip.set(Buffer(UInt(8), frame.channels(), frame.cols, frame.rows, 0, frame.data));
	return convert.realize(frame.cols, frame.rows, frame.channels());
}
