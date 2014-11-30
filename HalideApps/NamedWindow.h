#pragma once
class NamedWindow
{
public:
	NamedWindow(std::string name = "Window", int flags = cv::WINDOW_AUTOSIZE);
	void resize(int width, int height);
	void move(int x, int y);
	void showImage(Halide::Image<float> im);
	void showImage(cv::Mat mat);
	void showImage2D(Halide::Image<uint8_t> im);
	void showImage3D(Halide::Image<uint8_t> im);
	void close();

private:
	void showImage3D(Halide::Image<float> im);
	void showImage2D(Halide::Image<float> im);

	std::string name;
};

