#pragma once

// Returns initialSize / 2^level. Used for pyramids.
inline int scaleSize(int initialSize, int level)
{
	while (--level >= 0)
		initialSize /= 2;
	return initialSize;
}

// Downsample with a 1 2 1 filter
template<typename F>
Halide::Func downsample(F f)
{
	Halide::Func downx("downx"), downy("downy");
	Halide::Var x, y;

	downx(x, y, Halide::_) = (f(2 * x - 1, y, Halide::_) + 2.0f * f(2 * x, y, Halide::_) + f(2 * x + 1, y, Halide::_)) / 4.0f;
	downy(x, y, Halide::_) = (downx(x, 2 * y - 1, Halide::_) + 2.0f * downx(x, 2 * y, Halide::_) + downx(x, 2 * y + 1, Halide::_)) / 4.0f;

	return downy;
}

// Downsample with a 1 4 6 4 1 filter
template<typename F>
Halide::Func downsample5(F f)
{
	Halide::Func downx("downx"), downy("downy");
	Halide::Var x, y;

	downx(x, y, Halide::_) = (f(2 * x - 2, y, Halide::_) + 4 * f(2 * x - 1, y, Halide::_) + 6 * f(2 * x, y, Halide::_) + 4 * f(2 * x + 1, y, Halide::_) + f(2 * x + 2, y, Halide::_)) / 16.0f;
	downy(x, y, Halide::_) = (downx(x, 2 * y - 2, Halide::_) + 4 * downx(x, 2 * y - 1, Halide::_) + 6 * downx(x, 2 * y, Halide::_) + 4 * downx(x, 2 * y + 1, Halide::_) + downx(x, 2 * y + 2, Halide::_)) / 16.0f;

	return downy;
}

// Downsample with a Gaussian 5x5 filter
template<typename F>
Halide::Func downsampleG5(F f)
{
	Halide::Func downx("downx"), downy("downy");
	Halide::Var x, y;

	float coeffs[] = { 0.054488684549644346f, 0.24420134200323348f, 0.40261994689424435f, 0.24420134200323348f, 0.054488684549644346f };

	Expr xExpr = 0.0f;
	for (int i = 0; i < 5; i++)
		xExpr += coeffs[i] * f(2 * x - 2 + i, y, Halide::_);
	downx(x, y, Halide::_) = xExpr;

	Expr yExpr = 0.0f;
	for (int i = 0; i < 5; i++)
		yExpr += coeffs[i] * downx(x, 2 * y - 2 + i, Halide::_);
	downy(x, y, Halide::_) = yExpr;

	return downy;
}

// Upsample using bilinear interpolation
template<typename F>
Halide::Func upsample(F f)
{
	Halide::Func upx("upx"), upy("upy");
	Halide::Var x, y;

	upx(x, y, Halide::_) = 0.25f * f((x / 2) - 1 + 2 * (x % 2), y, Halide::_) + 0.75f * f(x / 2, y, Halide::_);
	upy(x, y, Halide::_) = 0.25f * upx(x, (y / 2) - 1 + 2 * (y % 2), Halide::_) + 0.75f * upx(x, y / 2, Halide::_);

	return upy;
}

inline Halide::Func clipToEdges(Halide::Image<float> im)
{
	Halide::Var x, y;
	return lambda(x, y, Halide::_, im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1), Halide::_));
}

inline Halide::Func clipToEdges(Halide::Func f, int width, int height)
{
	Halide::Var x, y;
	return lambda(x, y, Halide::_, f(clamp(x, 0, width - 1), clamp(y, 0, height - 1), Halide::_));
}

// Extern function to copy data to an external circular buffer of images.
// p is the offset in the third dimension to copy to (for circular buffers). Set to 0 if unused.
extern "C" __declspec(dllexport) int copyFloat32(int p, buffer_t* copyTo, buffer_t* in, buffer_t* out);

std::vector<Halide::Func> makeFuncArray(int pyramidLevels, std::string name);

// Creates a function which copies the output of a function to a circular buffer of images.
// pParam is the index in the circular buffer.
Halide::Func copyToCircularBuffer(Halide::Func input, const Halide::Image<float>& buffer, Halide::Param<int> pParam, std::string name);

std::vector<Halide::Func> copyPyramidToCircularBuffer(int pyramidLevels, const std::vector<Halide::Func>& input, const std::vector<Halide::Image<float>>& buffer, Halide::Param<int> pParam, std::string name);
