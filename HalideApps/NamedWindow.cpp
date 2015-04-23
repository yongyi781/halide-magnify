#include "stdafx.h"
#include "NamedWindow.h"

NamedWindow::NamedWindow(std::string name, int flags) : name(name)
{
	cv::namedWindow(name, flags);
}

void NamedWindow::resize(int width, int height)
{
	cv::resizeWindow(name, width, height);
}

void NamedWindow::move(int x, int y)
{
	cv::moveWindow(name, x, y);
}

void NamedWindow::showImage(Halide::Image<float> im)
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
		throw std::exception("Image must be either 2- or 3-dimensional.");
	}
}

void NamedWindow::showImage(cv::Mat mat)
{
	cv::imshow(name, mat);
}

void NamedWindow::close()
{
	cv::destroyWindow(name);
}

void NamedWindow::showImage3D(Halide::Image<float> im)
{
	static Halide::Func convert("convertToMat3D");
	static Halide::ImageParam ip(Halide::Float(32), 3);
	static Halide::Var x, y, c;

	if (!convert.defined())
	{
		convert(c, x, y) = Halide::cast<uint8_t>(ip(x, y, 2 - c) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC3, cv::Scalar(0));
	convert.realize(Halide::Buffer(Halide::UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
	cv::imshow(name, mat);
}

void NamedWindow::showImage3D(Halide::Image<uint8_t> im)
{
	static Halide::Func convert("convertToMat3D");
	static Halide::ImageParam ip(Halide::UInt(8), 3);
	static Halide::Var x, y, c;

	if (!convert.defined())
	{
		convert(c, x, y) = ip(x, y, 2 - c);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC3, cv::Scalar(0));
	convert.realize(Halide::Buffer(Halide::UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
	cv::imshow(name, mat);
}

void NamedWindow::showImage2D(Halide::Image<float> im)
{
	static Halide::Func convert("convertToMat2D");
	static Halide::ImageParam ip(Halide::Float(32), 2);
	static Halide::Var x, y;

	if (!convert.defined())
	{
		convert(x, y) = Halide::cast<uint8_t>(ip(x, y) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC1, cv::Scalar(0));
	convert.realize(Halide::Buffer(Halide::UInt(8), im.width(), im.height(), 0, 0, mat.data));
	cv::imshow(name, mat);
}

void NamedWindow::showImage2D(Halide::Image<uint8_t> im)
{
	static Halide::Func convert("convertToMat2D");
	static Halide::ImageParam ip(Halide::UInt(8), 2);
	static Halide::Var x, y;

	if (!convert.defined())
	{
		convert(x, y) = ip(x, y);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC1, cv::Scalar(0));
	convert.realize(Halide::Buffer(Halide::UInt(8), im.width(), im.height(), 0, 0, mat.data));
	cv::imshow(name, mat);
}
