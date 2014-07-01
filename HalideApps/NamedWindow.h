#pragma once
class NamedWindow
{
public:
	NamedWindow(std::string name = "Window", int flags = cv::WINDOW_AUTOSIZE);
	void resize(int width, int height);
	void showImage(const Halide::Image<float>& im);
	void showImage(const cv::Mat& mat);
	void close();

private:
	void showImage3D(const Halide::Image<float>& im);
	void showImage2D(const Halide::Image<float>& im);

	std::string name;
};

