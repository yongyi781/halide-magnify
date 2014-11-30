#pragma once

const float gaussian5Coeffs[] = { 0.40261994689424435f, 0.24420134200323348f, 0.054488684549644346f };
//const float gaussian5Coeffs[] = { 0.375f, 0.25f, 0.0625f };

// Returns initialSize / 2^level. Used for pyramids.
inline Halide::Expr scaleSize(Halide::Expr initialSize, int level)
{
	return initialSize / (int)pow(2, level);
}

// Returns initialSize / 2^level. Used for pyramids.
inline int scaleSize(int initialSize, int level)
{
	return initialSize / (int)pow(2, level);
}

// Downsample with a 1 2 1 filter
inline Halide::Func downsample(Halide::Func f)
{
	Halide::Func downx("downx"), downy("downy");
	Halide::Var x, y;

	downx(x, y, Halide::_) = (f(2 * x - 1, y, Halide::_) + 2.0f * f(2 * x, y, Halide::_) + f(2 * x + 1, y, Halide::_)) / 4.0f;
	downy(x, y, Halide::_) = (downx(x, 2 * y - 1, Halide::_) + 2.0f * downx(x, 2 * y, Halide::_) + downx(x, 2 * y + 1, Halide::_)) / 4.0f;

	return downy;
}

// Downsample with a 1 4 6 4 1 filter
inline Halide::Func downsample5(Halide::Func f)
{
	Halide::Func downx("downx"), downy("downy");
	Halide::Var x, y;

	downx(x, y, Halide::_) =
		(f(2 * x - 2, y, Halide::_) + f(2 * x + 2, y, Halide::_)
		+ 4 * (f(2 * x - 1, y, Halide::_) + f(2 * x + 1, y, Halide::_))
		+ 6 * f(2 * x, y, Halide::_)) / 16;
	downy(x, y, Halide::_) =
		(downx(x, 2 * y - 2, Halide::_) + downx(x, 2 * y + 2, Halide::_)
		+ 4 * (downx(x, 2 * y - 1, Halide::_) + downx(x, 2 * y + 1, Halide::_))
		+ 6 * downx(x, 2 * y, Halide::_)) / 16;

	return downy;
}

// Downsample with a Gaussian 5x5 filter in x direction
inline Halide::Func downsampleG5X(Halide::Func f)
{
	Halide::Func downx("downx");
	Halide::Var x, y;

	downx(x, y, Halide::_) = gaussian5Coeffs[0] * f(2 * x, y, Halide::_)
		+ gaussian5Coeffs[1] * (f(2 * x - 1, y, Halide::_) + f(2 * x + 1, y, Halide::_))
		+ gaussian5Coeffs[2] * (f(2 * x - 2, y, Halide::_) + f(2 * x + 2, y, Halide::_));

	return downx;
}

// Downsample with a Gaussian 5x5 filter in y direction
inline Halide::Func downsampleG5Y(Halide::Func f)
{
	Halide::Func downy("downy");
	Halide::Var x, y;

	downy(x, y, Halide::_) = gaussian5Coeffs[0] * f(x, 2 * y, Halide::_)
		+ gaussian5Coeffs[1] * (f(x, 2 * y - 1, Halide::_) + f(x, 2 * y + 1, Halide::_))
		+ gaussian5Coeffs[2] * (f(x, 2 * y - 2, Halide::_) + f(x, 2 * y + 2, Halide::_));

	return downy;
}

// Upsample using bilinear interpolation in x direction
inline Halide::Func upsampleX(Halide::Func f)
{
	Halide::Func upx("upx");
	Halide::Var x, y;

	upx(x, y, Halide::_) = 0.25f * f((x / 2) - 1 + 2 * (x % 2), y, Halide::_) + 0.75f * f(x / 2, y, Halide::_);

	return upx;
}

// Upsample using bilinear interpolation in y direction
inline Halide::Func upsampleY(Halide::Func f)
{
	Halide::Func upy("upy");
	Halide::Var x, y;

	upy(x, y, Halide::_) = 0.25f * f(x, (y / 2) - 1 + 2 * (y % 2), Halide::_) + 0.75f * f(x, y / 2, Halide::_);

	return upy;
}

// Upsample using bilinear interpolation
inline Halide::Func upsample(Halide::Func f)
{
	Halide::Func upx("upx"), upy("upy");
	Halide::Var x, y;

	upx(x, y, Halide::_) = 0.25f * f((x / 2) - 1 + 2 * (x % 2), y, Halide::_) + 0.75f * f(x / 2, y, Halide::_);
	upy(x, y, Halide::_) = 0.25f * upx(x, (y / 2) - 1 + 2 * (y % 2), Halide::_) + 0.75f * upx(x, y / 2, Halide::_);

	return upy;
}

// Upsample using Gaussian upsampling in x direction
inline Halide::Func upsampleG5X(Halide::Func f)
{
	Halide::Func upx("upx");
	Halide::Var x, y;

	Halide::Expr c0 = ((x + 1) % 2) * gaussian5Coeffs[0];
	Halide::Expr c1 = (x % 2) * gaussian5Coeffs[1];
	Halide::Expr c2 = ((x + 1) % 2) * gaussian5Coeffs[2];

	upx(x, y, Halide::_) = (c0 * f(x / 2, y, Halide::_)
		+ c1 * (f(x / 2, y, Halide::_) + f(x / 2 + 1, y, Halide::_))
		+ c2 * (f(x / 2 - 1, y, Halide::_) + f(x / 2 + 1, y, Halide::_))) / (c0 + 2 * c1 + 2 * c2);

	return upx;
}

// Upsample using Gaussian upsampling in y direction
inline Halide::Func upsampleG5Y(Halide::Func f)
{
	Halide::Func upy("upy");
	Halide::Var x, y;

	Halide::Expr c0 = ((y + 1) % 2) * gaussian5Coeffs[0];
	Halide::Expr c1 = (y % 2) * gaussian5Coeffs[1];
	Halide::Expr c2 = ((y + 1) % 2) * gaussian5Coeffs[2];
	upy(x, y, Halide::_) = (c0 * f(x, y / 2, Halide::_)
		+ c1 * (f(x, y / 2, Halide::_) + f(x, y / 2 + 1, Halide::_))
		+ c2 * (f(x, y / 2 - 1, Halide::_) + f(x, y / 2 + 1, Halide::_))) / (c0 + 2 * c1 + 2 * c2);

	return upy;
}

inline Halide::Func clipToEdges(Halide::ImageParam im)
{
	Halide::Var x, y;
	return lambda(x, y, Halide::_, im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1), Halide::_));
}

inline Halide::Func clipToEdges(Halide::Func f, Halide::Expr width, Halide::Expr height)
{
	Halide::Var x, y;
	return lambda(x, y, Halide::_, f(clamp(x, 0, width - 1), clamp(y, 0, height - 1), Halide::_));
}

// Extern function to copy data to an external circular buffer of images.
// p is the offset in the third dimension to copy to (for circular buffers). Set to 0 if unused.
extern "C" __declspec(dllexport) int copyFloat32(int bufferType, int p, buffer_t* copyTo, buffer_t* in, buffer_t* out);

std::vector<Halide::Func> makeFuncArray(int pyramidLevels, std::string name);

// Creates a function which copies the output of a function to a circular buffer of images.
// pParam is the index in the circular buffer.
Halide::Func copyToCircularBuffer(Halide::Func input, Halide::ImageParam buffer, Halide::Expr bufferType, Halide::Param<int> pParam, std::string name);

std::vector<Halide::Func> copyPyramidToCircularBuffer(int pyramidLevels, const std::vector<Halide::Func>& input, const std::vector<Halide::ImageParam>& buffer, Halide::Expr bufferType, Halide::Param<int> pParam, std::string name);

Halide::Func gaussianBlurX(Halide::Func in, float sigma);
Halide::Func gaussianBlurY(Halide::Func in, float sigma);
