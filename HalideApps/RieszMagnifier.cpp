#include "stdafx.h"
#include "RieszMagnifier.h"
#include "Util.h"

using namespace Halide;

const float TINY = 1e-14f;

RieszMagnifier::RieszMagnifier(int channels, Halide::Type type, int pyramidLevels)
	: channels(channels), pyramidLevels(pyramidLevels), bandSigma(std::vector<float>(pyramidLevels))
	, a1(Param<float>("a1")), a2(Param<float>("a2")), b0(Param<float>("b0")), b1(Param<float>("b1")), b2(Param<float>("b2"))
	, alpha(Param<float>("alpha")), pParam(Param<int>("pParam")), output(Func("output"))
{
	if (channels != 1 && channels != 3)
		throw std::invalid_argument("Channels must be either 1 or 3.");
	if (type != UInt(8) && type != Float(32))
		throw std::invalid_argument("Only UInt(8) and Float(32) types are supported.");

	// Initialize pyramid buffer params
	for (int j = 0; j < pyramidLevels; j++)
		historyBuffer.push_back(ImageParam(Float(32), 4, "historyBuffer" + std::to_string(j)));

	// Initialize spatial regularization sigmas
	computeBandSigmas();

	input = ImageParam(type, channels == 3 ? 3 : 2, "input");
	Func floatInput("floatInput");
	if (type == UInt(8))
		floatInput(_) = cast<float>(input(_)) / 255.0f;
	else
		floatInput(_) = input(_);
	Func grey("grey");
	Func cb("cb"), cr("cr");
	if (channels == 3)
	{
		grey(x, y) = 0.299f * floatInput(x, y, 0) + 0.587f * floatInput(x, y, 1) + 0.114f * floatInput(x, y, 2);

		// CbCr - 0.5
		cb(x, y) = -0.168736f * floatInput(x, y, 0) - 0.331264f * floatInput(x, y, 1) + 0.5f * floatInput(x, y, 2);
		cr(x, y) = 0.5f * floatInput(x, y, 0) - 0.418688f * floatInput(x, y, 1) - 0.081312f * floatInput(x, y, 2);
	}
	else
	{
		grey(x, y) = floatInput(x, y);
	}

	// Gaussian pyramid
	gPyramidDownX = makeFuncArray(pyramidLevels + 1, "gPyramidDownX");
	gPyramid = makeFuncArray(pyramidLevels + 1, "gPyramid");
	gPyramid[0](x, y) = grey(x, y);
	for (int j = 1; j <= pyramidLevels; j++)
	{
		gPyramidDownX[j](x, y) = downsampleG5X(clipToEdges(gPyramid[j - 1], scaleSize(input.width(), j - 1), scaleSize(input.height(), j - 1)))(x, y);
		gPyramid[j](x, y) = downsampleG5Y(gPyramidDownX[j])(x, y);
	}

	// Laplacian pyramid
	lPyramidUpX = makeFuncArray(pyramidLevels, "lPyramidUpX");
	lPyramid = makeFuncArray(pyramidLevels + 1, "lPyramid");
	lPyramid[pyramidLevels](x, y) = gPyramid[pyramidLevels](x, y);
	for (int j = pyramidLevels - 1; j >= 0; j--)
	{
		lPyramidUpX[j](x, y) = upsampleG5X(clipToEdges(gPyramid[j + 1], scaleSize(input.width(), j + 1), scaleSize(input.height(), j + 1)))(x, y);
		lPyramid[j](x, y) = gPyramid[j](x, y) - upsampleG5Y(lPyramidUpX[j])(x, y);
	}
	lPyramidCopy = copyPyramidToCircularBuffer(pyramidLevels, lPyramid, historyBuffer, 0, pParam, "lPyramidCopy");

	clampedPyramidBuffer = makeFuncArray(pyramidLevels, "clampedPyramidBuffer");
	for (int j = 0; j < pyramidLevels; j++)
	{
		clampedPyramidBuffer[j](x, y, p) = clipToEdges(historyBuffer[j])(x, y, 0, p);
	}

	// R1 pyramid
	r1Pyramid = makeFuncArray(pyramidLevels, "r1Pyramid");
	r1Prev = makeFuncArray(pyramidLevels, "r1Prev");
	for (int j = 0; j < pyramidLevels; j++)
	{
		Func clamped = clipToEdges(lPyramid[j], scaleSize(input.width(), j), scaleSize(input.height(), j));
		r1Pyramid[j](x, y) = -0.6f * clamped(x - 1, y) + 0.6f * clamped(x + 1, y);
		r1Prev[j](x, y) = -0.6f * clampedPyramidBuffer[j](x - 1, y, (pParam + 1) % 2) + 0.6f * clampedPyramidBuffer[j](x + 1, y, (pParam + 1) % 2);
	}

	// R2 pyramid
	r2Pyramid = makeFuncArray(pyramidLevels, "r2Pyramid");
	r2Prev = makeFuncArray(pyramidLevels, "r2Prev");
	for (int j = 0; j < pyramidLevels; j++)
	{
		Func clamped = clipToEdges(lPyramid[j], scaleSize(input.width(), j), scaleSize(input.height(), j));
		r2Pyramid[j](x, y) = -0.6f * clamped(x, y - 1) + 0.6f * clamped(x, y + 1);
		r2Prev[j](x, y) = -0.6f * clampedPyramidBuffer[j](x, y - 1, (pParam + 1) % 2) + 0.6f * clampedPyramidBuffer[j](x, y + 1, (pParam + 1) % 2);
	}

	// quaternionic phase difference as a tuple
	productReal = makeFuncArray(pyramidLevels, "productReal");
	productI = makeFuncArray(pyramidLevels, "productI");
	productJ = makeFuncArray(pyramidLevels, "productJ");
	ijAmplitude = makeFuncArray(pyramidLevels, "ijAmplitude");
	amplitude = makeFuncArray(pyramidLevels, "amplitude");
	phi = makeFuncArray(pyramidLevels, "phi");
	qPhaseDiffC = makeFuncArray(pyramidLevels, "qPhaseDiffC");
	qPhaseDiffS = makeFuncArray(pyramidLevels, "qPhaseDiffS");
	for (int j = 0; j < pyramidLevels; j++)
	{
		// q x q_prev*
		productReal[j](x, y) = lPyramidCopy[j](x, y) * historyBuffer[j](x, y, 0, (pParam + 1) % 2)
			+ r1Pyramid[j](x, y) * r1Prev[j](x, y)
			+ r2Pyramid[j](x, y) * r2Prev[j](x, y);
		productI[j](x, y) = r1Pyramid[j](x, y) * historyBuffer[j](x, y, 0, (pParam + 1) % 2)
			- r1Prev[j](x, y) * lPyramid[j](x, y);
		productJ[j](x, y) = r2Pyramid[j](x, y) * historyBuffer[j](x, y, 0, (pParam + 1) % 2)
			- r2Prev[j](x, y) * lPyramid[j](x, y);

		ijAmplitude[j](x, y) = hypot(productI[j](x, y), productJ[j](x, y)) + TINY;
		amplitude[j](x, y) = hypot(ijAmplitude[j](x, y), productReal[j](x, y)) + TINY;

		// cos(phi) = q x q_prev^-1 = q x q_prev* / ||q * q_prev||
		phi[j](x, y) = acos(productReal[j](x, y) / amplitude[j](x, y)) / ijAmplitude[j](x, y);

		qPhaseDiffC[j](x, y) = productI[j](x, y) * phi[j](x, y);
		qPhaseDiffS[j](x, y) = productJ[j](x, y) * phi[j](x, y);
	}

	// Cumulative sums
	phaseC = makeFuncArray(pyramidLevels, "phaseC");
	phaseS = makeFuncArray(pyramidLevels, "phaseS");
	for (int j = 0; j < pyramidLevels; j++)
	{
		phaseC[j](x, y) = qPhaseDiffC[j](x, y) + historyBuffer[j](x, y, 1, (pParam + 1) % 2);
		phaseS[j](x, y) = qPhaseDiffS[j](x, y) + historyBuffer[j](x, y, 2, (pParam + 1) % 2);
	}

	phaseCCopy = copyPyramidToCircularBuffer(pyramidLevels, phaseC, historyBuffer, 1, pParam, "phaseCCopy");
	phaseSCopy = copyPyramidToCircularBuffer(pyramidLevels, phaseS, historyBuffer, 2, pParam, "phaseSCopy");

	changeC = makeFuncArray(pyramidLevels, "changeC");
	lowpass1C = makeFuncArray(pyramidLevels, "lowpass1C");
	lowpass2C = makeFuncArray(pyramidLevels, "lowpass2C");
	changeS = makeFuncArray(pyramidLevels, "changeS");
	lowpass1S = makeFuncArray(pyramidLevels, "lowpass1S");
	lowpass2S = makeFuncArray(pyramidLevels, "lowpass2S");
	for (int j = 0; j < pyramidLevels; j++)
	{
		// Linear filter. Order of evaluation here is important.
		changeC[j](x, y) = b0 * phaseCCopy[j](x, y) + historyBuffer[j](x, y, 3, (pParam + 1) % 2);
		lowpass1C[j](x, y) = b1 * phaseCCopy[j](x, y) + historyBuffer[j](x, y, 4, (pParam + 1) % 2) - a1 * changeC[j](x, y);
		lowpass2C[j](x, y) = b2 * phaseCCopy[j](x, y) - a2 * changeC[j](x, y);

		changeS[j](x, y) = b0 * phaseSCopy[j](x, y) + historyBuffer[j](x, y, 5, (pParam + 1) % 2);
		lowpass1S[j](x, y) = b1 * phaseSCopy[j](x, y) + historyBuffer[j](x, y, 6, (pParam + 1) % 2) - a1 * changeS[j](x, y);
		lowpass2S[j](x, y) = b2 * phaseSCopy[j](x, y) - a2 * changeS[j](x, y);
	}
	lowpass1CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1C, historyBuffer, 3, pParam, "lowpass1CCopy");
	lowpass2CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2C, historyBuffer, 4, pParam, "lowpass2CCopy");
	lowpass1SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1S, historyBuffer, 5, pParam, "lowpass1SCopy");
	lowpass2SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2S, historyBuffer, 6, pParam, "lowpass2SCopy");

	changeCTuple = makeFuncArray(pyramidLevels, "changeCTuple");
	changeSTuple = makeFuncArray(pyramidLevels, "changeSTuple");
	changeC2 = makeFuncArray(pyramidLevels, "changeC2");
	changeS2 = makeFuncArray(pyramidLevels, "changeS2");
	for (int j = 0; j < pyramidLevels; j++)
	{
		changeCTuple[j](x, y) = { changeC[j](x, y), lowpass1CCopy[j](x, y), lowpass2CCopy[j](x, y) };
		changeSTuple[j](x, y) = { changeS[j](x, y), lowpass1SCopy[j](x, y), lowpass2SCopy[j](x, y) };
		changeC2[j](x, y) = changeCTuple[j](x, y)[0];
		changeS2[j](x, y) = changeSTuple[j](x, y)[0];
	}

	amp = makeFuncArray(pyramidLevels, "amp");
	changeCAmp = makeFuncArray(pyramidLevels, "changeCAmp");
	changeCRegX = makeFuncArray(pyramidLevels, "changeCRegX");
	changeCReg = makeFuncArray(pyramidLevels, "changeCReg");
	changeSAmp = makeFuncArray(pyramidLevels, "changeSAmp");
	changeSRegX = makeFuncArray(pyramidLevels, "changeSRegX");
	changeSReg = makeFuncArray(pyramidLevels, "changeSReg");
	ampRegX = makeFuncArray(pyramidLevels, "ampRegX");
	ampReg = makeFuncArray(pyramidLevels, "ampReg");
	magC = makeFuncArray(pyramidLevels, "magC");
	pair = makeFuncArray(pyramidLevels, "pair");
	outLPyramid = makeFuncArray(pyramidLevels, "outLPyramid");
	for (int j = 0; j < pyramidLevels; j++)
	{
		float sigma = bandSigma[j];

		amp[j](x, y) = sqrt(lPyramid[j](x, y) * lPyramid[j](x, y)
			+ r1Pyramid[j](x, y) * r1Pyramid[j](x, y)
			+ r2Pyramid[j](x, y) * r2Pyramid[j](x, y)) + TINY;

		ampRegX[j](x, y) = gaussianBlurX(clipToEdges(amp[j], scaleSize(input.width(), j), scaleSize(input.height(), j)), sigma)(x, y);
		ampReg[j](x, y) = gaussianBlurY(ampRegX[j], sigma)(x, y);

		changeCAmp[j](x, y) = changeC2[j](x, y) * amp[j](x, y);
		changeCRegX[j](x, y) = gaussianBlurX(clipToEdges(changeCAmp[j], scaleSize(input.width(), j), scaleSize(input.height(), j)), sigma)(x, y);
		changeCReg[j](x, y) = gaussianBlurY(changeCRegX[j], sigma)(x, y) / ampReg[j](x, y);

		changeSAmp[j](x, y) = changeS2[j](x, y) * amp[j](x, y);
		changeSRegX[j](x, y) = gaussianBlurX(clipToEdges(changeSAmp[j], scaleSize(input.width(), j), scaleSize(input.height(), j)), sigma)(x, y);
		changeSReg[j](x, y) = gaussianBlurY(changeSRegX[j], sigma)(x, y) / ampReg[j](x, y);

		Expr creg = sigma == 0.0f ? changeC2[j](x, y) : changeCReg[j](x, y);
		Expr sreg = sigma == 0.0f ? changeS2[j](x, y) : changeSReg[j](x, y);
		magC[j](x, y) = hypot(creg, sreg) + TINY;

		pair[j](x, y) = (r1Pyramid[j](x, y) * creg + r2Pyramid[j](x, y) * sreg) / magC[j](x, y);
		outLPyramid[j](x, y) = lPyramid[j](x, y) * cos(alpha * magC[j](x, y)) - pair[j](x, y) * sin(alpha * magC[j](x, y));
	}

	outGPyramidUpX = makeFuncArray(pyramidLevels, "outGPyramidUpX");
	outGPyramid = makeFuncArray(pyramidLevels + 1, "outGPyramid");
	outGPyramid[pyramidLevels](x, y) = lPyramid[pyramidLevels](x, y);
	for (int j = pyramidLevels - 1; j >= 0; j--)
	{
		outGPyramidUpX[j](x, y) = upsampleG5X(clipToEdges(outGPyramid[j + 1], scaleSize(input.width(), j + 1), scaleSize(input.height(), j + 1)))(x, y);
		outGPyramid[j](x, y) = outLPyramid[j](x, y) + upsampleG5Y(outGPyramidUpX[j])(x, y);
	}

	if (channels == 1)
	{
		floatOutput(x, y) = clamp(outGPyramid[0](x, y), 0.0f, 1.0f);
		output(x, y) = type == UInt(8) ? cast<uint8_t>(floatOutput(x, y) * 255.0f) : floatOutput(x, y);
	}
	else
	{
		// YCrCb -> RGB
		floatOutput(x, y, c) = clamp(select(
			c == 0, outGPyramid[0](x, y) + 1.402f * cr(x, y),
			c == 1, outGPyramid[0](x, y) - 0.34414f * cb(x, y) - 0.71414f * cr(x, y),
			outGPyramid[0](x, y) + 1.772f * cb(x, y)), 0.0f, 1.0f);
		output(x, y, c) = type == UInt(8) ? cast<uint8_t>(floatOutput(x, y, c) * 255.0f) : floatOutput(x, y, c);
	}
}

void RieszMagnifier::schedule(bool tile, Halide::Target target)
{
	if (target.arch == Target::Arch::X86)
	{
		scheduleX86(tile);
	}
	else if (target.arch == Target::Arch::ARM)
	{
		scheduleARM(tile);
	}
}

void RieszMagnifier::scheduleX86(bool tile)
{
	const int VECTOR_SIZE = 8;

	// Schedule
	if (channels == 3)
		output.reorder(c, x, y).bound(c, 0, channels).unroll(c);
	output.parallel(y, 4).vectorize(x, 4);

	if (tile)
	{
		output.tile(x, y, xi, yi, 80, 20);
	}

	for (int j = 0; j < pyramidLevels; j++)
	{
		if (tile && j <= 0)
		{
			outGPyramid[j].compute_at(output, x);
			outGPyramidUpX[j].compute_at(output, x);

			ampReg[j].compute_at(output, x);
			ampRegX[j].compute_at(output, x);
			changeCReg[j].compute_at(output, x);
			changeCRegX[j].compute_at(output, x);
			changeSReg[j].compute_at(output, x);
			changeSRegX[j].compute_at(output, x);
			amp[j].compute_at(output, x);

			changeCTuple[j].compute_at(output, x);
			changeSTuple[j].compute_at(output, x);

			lowpass1CCopy[j].compute_at(output, x);
			lowpass2CCopy[j].compute_at(output, x);
			lowpass1SCopy[j].compute_at(output, x);
			lowpass2SCopy[j].compute_at(output, x);
			lowpass1C[j].compute_at(output, x);
			lowpass2C[j].compute_at(output, x);
			lowpass1S[j].compute_at(output, x);
			lowpass2S[j].compute_at(output, x);

			phaseCCopy[j].compute_at(output, x);
			phaseSCopy[j].compute_at(output, x);
			phaseC[j].compute_at(output, x);
			phaseS[j].compute_at(output, x);
			phi[j].compute_at(output, x);

			r1Pyramid[j].compute_at(output, x);
			r2Pyramid[j].compute_at(output, x);

			lPyramidCopy[j].compute_at(output, x);
			lPyramid[j].compute_at(output, x);
			lPyramidUpX[j].compute_at(output, x);
		}
		else
		{
			outGPyramid[j].compute_root();
			outGPyramidUpX[j].compute_root();

			ampReg[j].compute_root().split(y, y, yi, 16);
			ampRegX[j].compute_at(ampReg[j], y);
			changeCReg[j].compute_root().split(y, y, yi, 16);
			changeCRegX[j].compute_at(changeCReg[j], y);
			changeSReg[j].compute_root().split(y, y, yi, 16);
			changeSRegX[j].compute_at(changeSReg[j], y);
			amp[j].compute_root();

			changeCTuple[j].compute_root();
			changeSTuple[j].compute_root();

			lowpass1CCopy[j].compute_root();
			lowpass2CCopy[j].compute_root();
			lowpass1SCopy[j].compute_root();
			lowpass2SCopy[j].compute_root();
			lowpass1C[j].compute_root();
			lowpass2C[j].compute_root();
			lowpass1S[j].compute_root();
			lowpass2S[j].compute_root();

			phaseCCopy[j].compute_root();
			phaseSCopy[j].compute_root();
			phaseC[j].compute_root();
			phaseS[j].compute_root();
			phi[j].compute_root();

			r1Pyramid[j].compute_root();
			r2Pyramid[j].compute_root();

			lPyramidCopy[j].compute_root();
			lPyramid[j].compute_root().split(y, y, yi, 8);
			lPyramidUpX[j].compute_at(lPyramid[j], y);
		}

		if (j > 0)
		{
			gPyramid[j].compute_root().split(y, y, yi, 8);
			gPyramidDownX[j].compute_at(gPyramid[j], y);
		}

		if (j <= 4)
		{
			outGPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			outGPyramidUpX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			ampReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			ampRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeCReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeCRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			amp[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			changeCTuple[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSTuple[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			lowpass1C[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass2C[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass1S[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass2S[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			phaseC[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			phaseS[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			phi[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			r1Pyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			r2Pyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			lPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lPyramidUpX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			if (j > 0)
			{
				gPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
				gPyramidDownX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			}
		}
	}

	// The final level
	gPyramid[pyramidLevels].compute_root().parallel(y, 4).vectorize(x, VECTOR_SIZE);
}

void RieszMagnifier::scheduleARM(bool tile)
{
	const int VECTOR_SIZE = 4;

	// Schedule
	if (channels == 3)
		output.reorder(c, x, y).bound(c, 0, channels).unroll(c);
	output.parallel(y, 4).vectorize(x, 4);

	if (tile)
	{
		output.tile(x, y, xi, yi, 80, 20);
	}

	for (int j = 0; j < pyramidLevels; j++)
	{
		if (tile)
		{
			outGPyramid[j].compute_at(output, x);
			outGPyramidUpX[j].compute_at(output, x);

			ampReg[j].compute_at(output, x);
			ampRegX[j].compute_at(output, x);
			changeCReg[j].compute_at(output, x);
			changeCRegX[j].compute_at(output, x);
			changeSReg[j].compute_at(output, x);
			changeSRegX[j].compute_at(output, x);
			amp[j].compute_at(output, x);

			changeCTuple[j].compute_at(output, x);
			changeSTuple[j].compute_at(output, x);

			lowpass1CCopy[j].compute_at(output, x);
			lowpass2CCopy[j].compute_at(output, x);
			lowpass1SCopy[j].compute_at(output, x);
			lowpass2SCopy[j].compute_at(output, x);
			lowpass1C[j].compute_at(output, x);
			lowpass2C[j].compute_at(output, x);
			lowpass1S[j].compute_at(output, x);
			lowpass2S[j].compute_at(output, x);

			phaseCCopy[j].compute_at(output, x);
			phaseSCopy[j].compute_at(output, x);
			phaseC[j].compute_at(output, x);
			phaseS[j].compute_at(output, x);
			phi[j].compute_at(output, x);

			r1Pyramid[j].compute_at(output, x);
			r2Pyramid[j].compute_at(output, x);

			lPyramidCopy[j].compute_at(output, x);
			lPyramid[j].compute_at(output, x);
			lPyramidUpX[j].compute_at(output, x);
		}
		else
		{
			outGPyramid[j].compute_root();
			outGPyramidUpX[j].compute_root();

			ampReg[j].compute_root().split(y, y, yi, 16);
			ampRegX[j].compute_at(ampReg[j], y);
			changeCReg[j].compute_root().split(y, y, yi, 16);
			changeCRegX[j].compute_at(changeCReg[j], y);
			changeSReg[j].compute_root().split(y, y, yi, 16);
			changeSRegX[j].compute_at(changeSReg[j], y);
			amp[j].compute_root();

			changeCTuple[j].compute_root();
			changeSTuple[j].compute_root();

			lowpass1CCopy[j].compute_root();
			lowpass2CCopy[j].compute_root();
			lowpass1SCopy[j].compute_root();
			lowpass2SCopy[j].compute_root();
			lowpass1C[j].compute_root();
			lowpass2C[j].compute_root();
			lowpass1S[j].compute_root();
			lowpass2S[j].compute_root();

			phaseCCopy[j].compute_root();
			phaseSCopy[j].compute_root();
			phaseC[j].compute_root();
			phaseS[j].compute_root();
			phi[j].compute_root();

			r1Pyramid[j].compute_root();
			r2Pyramid[j].compute_root();

			lPyramidCopy[j].compute_root();
			lPyramid[j].compute_root().split(y, y, yi, 8);
			lPyramidUpX[j].compute_at(lPyramid[j], y);
		}

		if (j > 0)
		{
			gPyramid[j].compute_root().split(y, y, yi, 8);
			gPyramidDownX[j].compute_at(gPyramid[j], y);
		}

		if (j <= 4)
		{
			outGPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			outGPyramidUpX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			ampReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			ampRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeCReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeCRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			amp[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			changeCTuple[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSTuple[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			lowpass1C[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass2C[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass1S[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass2S[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			phaseC[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			phaseS[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			phi[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			r1Pyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			r2Pyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			lPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lPyramidUpX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			if (j > 0)
			{
				gPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
				gPyramidDownX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			}
		}
	}

	// The final level
	gPyramid[pyramidLevels].compute_root().parallel(y, 4).vectorize(x, VECTOR_SIZE);
}

void RieszMagnifier::compileJIT(bool tile, Target target)
{
	// Schedule
	schedule(tile, target);

	// Compile JIT
	std::cout << "Compiling JIT for target " << target.to_string() << "...";
	double t = currentTime();
	output.compile_jit();
	std::cout << "done! Elapsed: " << (currentTime() - t) / 1000 << " s" << std::endl;
}

void RieszMagnifier::compileToFile(std::string filenamePrefix, bool tile, Target target)
{
	// Schedule
	schedule(tile, target);

	std::vector<Argument> arguments{ input, a1, a2, b0, b1, b2, alpha, pParam };
	for (int j = 0; j < pyramidLevels; j++)
		arguments.push_back(historyBuffer[j]);

	std::cout << "Compiling to '" << filenamePrefix << "' for target " << target.to_string() << "...";
	double t = currentTime();
	output.compile_to_file(filenamePrefix, arguments, target);
	std::cout << "done! Elapsed: " << (currentTime() - t) / 1000 << " s" << std::endl;
}

void RieszMagnifier::bindJIT(float a1, float a2, float b0, float b1, float b2, float alpha, std::vector<Halide::Image<float>> historyBuffer)
{
	this->a1.set(a1);
	this->a2.set(a2);
	this->b0.set(b0);
	this->b1.set(b1);
	this->b2.set(b2);
	this->alpha.set(alpha);
	for (int j = 0; j < pyramidLevels; j++)
		this->historyBuffer[j].set(historyBuffer[j]);
}

void RieszMagnifier::process(Buffer frame, Buffer out)
{
	pParam.set(frameCounter % CIRCBUFFER_SIZE);
	input.set(frame);
	output.realize(out);

	frameCounter++;
}

void RieszMagnifier::computeBandSigmas()
{
	for (int j = 0; j < pyramidLevels; j++)
	{
		bandSigma[j] = 3;
	}
}
