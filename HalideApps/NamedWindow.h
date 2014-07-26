#pragma once
class NamedWindow
{
public:
	NamedWindow(std::string name = "Window", int flags = cv::WINDOW_AUTOSIZE);
	void resize(int width, int height);
	void move(int x, int y);
	void showImage(Halide::Image<float> im);
	void showImage(cv::Mat mat);
	void close();

private:
	void showImage3D(Halide::Image<float> im);
	void showImage2D(Halide::Image<float> im);

	std::string name;
};

