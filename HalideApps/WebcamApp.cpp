#include "stdafx.h"
#include "WebcamApp.h"

using namespace Halide;

WebcamApp::WebcamApp() : cap(0)
{
	if (!cap.isOpened())
		throw std::exception("Cannot open webcam.");
	cv::namedWindow("Webcam app");
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

void WebcamApp::showImage(const Image<float>& im)
{
	switch (im.dimensions())
	{
	case 2:
		showImage2D(im);
		break;
	case 3:
		showImage3D(im);
		break;
	default:
		throw std::exception("Image is neither 2- or 3-dimensional.");
	}
}

void WebcamApp::showImage3D(const Image<float>& im)
{
	static Func convert;
	static ImageParam ip(Float(32), 3);
	static Var x, y, c;

	if (!convert.defined())
	{
		convert(c, x, y) = cast<uint8_t>(ip(x, y, c) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC3, cv::Scalar(0));
	convert.realize(Buffer(UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
	cv::imshow("Webcam app", mat);
}

void WebcamApp::showImage2D(const Image<float>& im)
{
	static Func convert;
	static ImageParam ip(Float(32), 2);
	static Var x, y;

	if (!convert.defined())
	{
		convert(x, y) = cast<uint8_t>(ip(x, y) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC1, cv::Scalar(0));
	convert.realize(Buffer(UInt(8), im.width(), im.height(), 0, 0, mat.data));
	cv::imshow("Webcam app", mat);
}
