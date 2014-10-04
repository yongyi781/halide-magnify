#include "stdafx.h"
#include "Util.h"

using namespace Halide;

int copyFloat32(int p, buffer_t* copyTo, buffer_t* in, buffer_t* out)
{
	if (in->host == nullptr || out->host == nullptr)
	{
		if (in->host == nullptr)
		{
			for (int i = 0; i < 2; i++)
			{
				in->min[i] = out->min[i];
				in->extent[i] = out->extent[i];
			}
		}
	}
	else
	{
#if TRACE
		printf("[Copy] p: %d out: [%d, %d] x [%d, %d], in: [%d, %d] x [%d, %d]\n", p,
			out->min[0], out->min[0] + out->extent[0] - 1, out->min[1], out->min[1] + out->extent[1] - 1,
			in->min[0], in->min[0] + in->extent[0] - 1, in->min[1], in->min[1] + in->extent[1] - 1);
#endif
		float* src = (float*)in->host;
		float* dst = (float*)out->host;
		float* dstCopy = (float*)copyTo->host + p * copyTo->stride[2];
		for (int y = out->min[1]; y < out->min[1] + out->extent[1]; y++)
		{
			float* srcLine = src + (y - in->min[1]) * in->stride[1];
			float* dstLine = dst + (y - out->min[1]) * out->stride[1];
			float* copyLine = dstCopy + y * copyTo->stride[1];
			memcpy(dstLine, srcLine + out->min[0] - in->min[0], sizeof(float) * out->extent[0]);
			memcpy(copyLine + out->min[0], srcLine + out->min[0] - in->min[0], sizeof(float) * out->extent[0]);
		}
	}
	return 0;
}

std::vector<Halide::Func> makeFuncArray(int pyramidLevels, std::string name)
{
	std::vector<Halide::Func> f(pyramidLevels);
	for (int j = 0; j < pyramidLevels; j++)
	{
		f[j] = Halide::Func(name + std::to_string(j));
	}
	return f;
}

Func copyToCircularBuffer(Func input, const Image<float>& buffer, Param<int> pParam, std::string name)
{
	Func f(name);
	Param<buffer_t*> copyToParam;
	copyToParam.set(buffer.raw_buffer());
	f.define_extern("copyFloat32", { pParam, copyToParam, input }, Float(32), 2);
	return f;
}

std::vector<Func> copyPyramidToCircularBuffer(int pyramidLevels, const std::vector<Func>& input, const std::vector<Image<float>>& buffer, Param<int> pParam, std::string name)
{
	std::vector<Func> fPyr(pyramidLevels);
	for (int j = 0; j < pyramidLevels; j++)
		fPyr[j] = copyToCircularBuffer(input[j], buffer[j], pParam, name + "_" + std::to_string(j));
	return fPyr;
}

Image<float> transpose(const Image<float>& im)
{
	Image<float> transposed(im.height(), im.width());
	for (int yi = 0; yi < im.height(); ++yi)
		for (int xi = 0; xi < im.width(); ++xi)
			transposed(yi, xi) = im(xi, yi);
	return transposed;
}

Image<float> horiGaussKernel(float sigma)
{
	Image<float> out(6 * (int)sigma + 1, 1);
	int center = 3 * (int)sigma;
	out(center) = 1.f;
	float total = 1.f;
	for (int xi = 1; xi <= center; ++xi)
		total += 2 * (out(center + xi) = out(center - xi) = exp(-xi * xi / (2.f * sigma * sigma)));
	for (int i = 0; i < out.width(); ++i)
		out(i) /= total;
	return out;
}

Func convolve(Func in, Image<float> kernel)
{
	Func f;
	Var x, y, c;
	RDom r(kernel);
	if (in.dimensions() >= 3)
		f(x, y, c) = sum(kernel(r.x, r.y) * in(x + r.x - kernel.width() / 2, y + r.y - kernel.height() / 2, c));
	else
		f(x, y) = sum(kernel(r.x, r.y) * in(x + r.x - kernel.width() / 2, y + r.y - kernel.height() / 2));
	return f;
}

Func gaussianBlur(Func in, float sigma, bool use2DKernel)
{
	Image<float> kernel = horiGaussKernel(sigma);
	Func f = convolve(in, kernel);
	if (!use2DKernel)
		f.compute_root();
	return convolve(f, transpose(kernel));
}
