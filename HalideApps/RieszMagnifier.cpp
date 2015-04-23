#include "stdafx.h"
#include "RieszMagnifier.h"
#include "Util.h"

using namespace Halide;

const float TINY = 1e-24f;

// Convert multiplication by float to int16 operation.
// Precondition: |expr * factor| < 32768
Expr mult16(float factor, Expr expr)
{
	int16_t m = (int16_t)(factor * 255);
	return expr * m / 255;
}

// Convert multiplication by float to int16 operation.
// Precondition: |expr * factor| < 32768
Expr mult16(Param<float> factor, Expr expr)
{
	Expr m = cast<int16_t>(factor * 255);
	return expr * m / 255;
}

// Convert multiplication by exprs to int16 operation.
// Precondition: |expr1 * expr2| < 32768
Expr multExpr16(Expr expr1, Expr expr2)
{
	return expr1 * expr2 / 255;
}

RieszMagnifier::RieszMagnifier(int channels, Halide::Type type, int pyramidLevels)
	: channels(channels), pyramidLevels(pyramidLevels), bandSigma(std::vector<float>(pyramidLevels))
{
	if (channels != 1 && channels != 3)
		throw std::invalid_argument("Channels must be either 1 or 3.");
	if (type != UInt(8) && type != Float(32))
		throw std::invalid_argument("Only UInt(8) and Float(32) types are supported.");

	// Initialize pyramid buffer params
	for (int j = 0; j < pyramidLevels; j++)
		historyBuffer.push_back(ImageParam(Int(16), 4, "historyBuffer" + std::to_string(j)));

	// Initialize spatial regularization sigmas
	computeBandSigmas();

	input = ImageParam(type, channels == 3 ? 3 : 2, "input");
	Func input16("input16");
	if (type != UInt(16))
		input16(_) = cast<int16_t>(input(_));
	else
		input16(_) = input(_);
	Func grey("grey");
	Func cb("cb"), cr("cr");
	if (channels == 3)
	{
		grey(x, y) = mult16(0.299f, input16(x, y, 0)) + mult16(0.587f, input16(x, y, 1)) + mult16(0.114f, input16(x, y, 2));
		// CbCr - 0.5
		cb(x, y) = 128 + mult16(0.564f, input16(x, y, 2) - grey(x, y));
		cr(x, y) = 128 + mult16(0.713f, input16(x, y, 0) - grey(x, y));
	}
	else
	{
		grey(x, y) = input16(x, y);
	}

	// Gaussian pyramid
	gPyramidDownX = makeFuncArray(pyramidLevels + 1, "gPyramidDownX");
	gPyramid = makeFuncArray(pyramidLevels + 1, "gPyramid");
	gPyramid[0](x, y) = grey(x, y);
	for (int j = 1; j <= pyramidLevels; j++)
	{
		gPyramidDownX[j](x, y) = downsample5X(clipToEdges(gPyramid[j - 1], scaleSize(input.width(), j - 1), scaleSize(input.height(), j - 1)))(x, y);
		gPyramid[j](x, y) = downsample5Y(gPyramidDownX[j])(x, y);
	}

	// Laplacian pyramid
	lPyramidUpX = makeFuncArray(pyramidLevels, "lPyramidUpX");
	lPyramid = makeFuncArray(pyramidLevels + 1, "lPyramid");
	lPyramid[pyramidLevels](x, y) = gPyramid[pyramidLevels](x, y);
	for (int j = pyramidLevels - 1; j >= 0; j--)
	{
		lPyramidUpX[j](x, y) = upsample5X(clipToEdges(gPyramid[j + 1], scaleSize(input.width(), j + 1), scaleSize(input.height(), j + 1)))(x, y);
		lPyramid[j](x, y) = gPyramid[j](x, y) - upsample5Y(lPyramidUpX[j])(x, y);
	}
	lPyramidCopy = copyPyramidToCircularBuffer(pyramidLevels, lPyramid, historyBuffer, 0, pParam, "lPyramidCopy", Int(16));

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
		r1Pyramid[j](x, y) = mult16(-0.6f, clamped(x - 1, y)) + mult16(0.6f, clamped(x + 1, y));
		r1Prev[j](x, y) = mult16(-0.6f, clampedPyramidBuffer[j](x - 1, y, (pParam + 1) % 2)) + mult16(0.6f, clampedPyramidBuffer[j](x + 1, y, (pParam + 1) % 2));
	}

	// R2 pyramid
	r2Pyramid = makeFuncArray(pyramidLevels, "r2Pyramid");
	r2Prev = makeFuncArray(pyramidLevels, "r2Prev");
	for (int j = 0; j < pyramidLevels; j++)
	{
		Func clamped = clipToEdges(lPyramid[j], scaleSize(input.width(), j), scaleSize(input.height(), j));
		r2Pyramid[j](x, y) = mult16(-0.6f, clamped(x, y - 1)) + mult16(0.6f, clamped(x, y + 1));
		r2Prev[j](x, y) = mult16(-0.6f, clampedPyramidBuffer[j](x, y - 1, (pParam + 1) % 2)) + mult16(0.6f, clampedPyramidBuffer[j](x, y + 1, (pParam + 1) % 2));
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
		productReal[j](x, y) = multExpr16(lPyramidCopy[j](x, y), historyBuffer[j](x, y, 0, (pParam + 1) % 2))
			+ multExpr16(r1Pyramid[j](x, y), r1Prev[j](x, y))
			+ multExpr16(r2Pyramid[j](x, y), r2Prev[j](x, y));
		productI[j](x, y) = multExpr16(r1Pyramid[j](x, y), historyBuffer[j](x, y, 0, (pParam + 1) % 2))
			- multExpr16(r1Prev[j](x, y), lPyramid[j](x, y));
		productJ[j](x, y) = multExpr16(r2Pyramid[j](x, y), historyBuffer[j](x, y, 0, (pParam + 1) % 2))
			- multExpr16(r2Prev[j](x, y), lPyramid[j](x, y));

		ijAmplitude[j](x, y) = hypot(cast<float>(productI[j](x, y)) / 255.0f, cast<float>(productJ[j](x, y)) / 255.0f) + TINY;
		amplitude[j](x, y) = hypot(ijAmplitude[j](x, y), cast<float>(productReal[j](x, y)) / 255.0f) + TINY;

		// cos(phi) = q x q_prev^-1 = q x q_prev* / ||q * q_prev||
		phi[j](x, y) = acos(cast<float>(productReal[j](x, y)) / 255.0f / amplitude[j](x, y)) / ijAmplitude[j](x, y);

		qPhaseDiffC[j](x, y) = cast<int16_t>(productI[j](x, y) * phi[j](x, y));
		qPhaseDiffS[j](x, y) = cast<int16_t>(productJ[j](x, y) * phi[j](x, y));
	}

	// Cumulative sums
	phaseC = makeFuncArray(pyramidLevels, "phaseC");
	phaseS = makeFuncArray(pyramidLevels, "phaseS");
	for (int j = 0; j < pyramidLevels; j++)
	{
		phaseC[j](x, y) = historyBuffer[j](x, y, 1, (pParam + 1) % 2) + qPhaseDiffC[j](x, y);
		phaseS[j](x, y) = historyBuffer[j](x, y, 2, (pParam + 1) % 2) + qPhaseDiffS[j](x, y);
	}

	phaseCCopy = copyPyramidToCircularBuffer(pyramidLevels, phaseC, historyBuffer, 1, pParam, "phaseCCopy", Int(16));
	phaseSCopy = copyPyramidToCircularBuffer(pyramidLevels, phaseS, historyBuffer, 2, pParam, "phaseSCopy", Int(16));

	changeC = makeFuncArray(pyramidLevels, "changeC");
	lowpass1C = makeFuncArray(pyramidLevels, "lowpass1C");
	lowpass2C = makeFuncArray(pyramidLevels, "lowpass2C");
	changeS = makeFuncArray(pyramidLevels, "changeS");
	lowpass1S = makeFuncArray(pyramidLevels, "lowpass1S");
	lowpass2S = makeFuncArray(pyramidLevels, "lowpass2S");
	for (int j = 0; j < pyramidLevels; j++)
	{
		// Linear filter. Order of evaluation here is important.
		changeC[j](x, y) = mult16(b0, phaseCCopy[j](x, y)) + historyBuffer[j](x, y, 3, (pParam + 1) % 2);
		lowpass1C[j](x, y) = mult16(b1, phaseCCopy[j](x, y)) + historyBuffer[j](x, y, 4, (pParam + 1) % 2) - mult16(a1, changeC[j](x, y));
		lowpass2C[j](x, y) = mult16(b2, phaseCCopy[j](x, y)) - mult16(a2, changeC[j](x, y));

		changeS[j](x, y) = mult16(b0, phaseSCopy[j](x, y)) + historyBuffer[j](x, y, 5, (pParam + 1) % 2);
		lowpass1S[j](x, y) = mult16(b1, phaseSCopy[j](x, y)) + historyBuffer[j](x, y, 6, (pParam + 1) % 2) - mult16(a1, changeS[j](x, y));
		lowpass2S[j](x, y) = mult16(b2, phaseSCopy[j](x, y)) - mult16(a2, changeS[j](x, y));
	}
	lowpass1CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1C, historyBuffer, 3, pParam, "lowpass1CCopy", Int(16));
	lowpass2CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2C, historyBuffer, 4, pParam, "lowpass2CCopy", Int(16));
	lowpass1SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1S, historyBuffer, 5, pParam, "lowpass1SCopy", Int(16));
	lowpass2SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2S, historyBuffer, 6, pParam, "lowpass2SCopy", Int(16));

	changeC2 = makeFuncArray(pyramidLevels, "changeC2");
	changeS2 = makeFuncArray(pyramidLevels, "changeS2");
	for (int j = 0; j < pyramidLevels; j++)
	{
		// The two TINY's are to force computation of the lowpass**Copy's.
		changeC2[j](x, y) = changeC[j](x, y) + lowpass1CCopy[j](x, y) / 32767 + lowpass2CCopy[j](x, y) / 32767;
		changeS2[j](x, y) = changeS[j](x, y) + lowpass1SCopy[j](x, y) / 32767 + lowpass2SCopy[j](x, y) / 32767;
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
		output16(x, y) = clamp(outGPyramid[0](x, y), 0.0f, 1.0f);
		output(x, y) = type != UInt(16) ? cast<uint8_t>(output16(x, y)) : output16(x, y);
	}
	else
	{
		// YCrCb -> RGB
		//output16(x, y, c) = clamp(select(
		//	c == 0, outGPyramid[0](x, y) + 1.402f * cr(x, y),
		//	c == 1, outGPyramid[0](x, y) - 0.34414f * cb(x, y) - 0.71414f * cr(x, y),
		//	outGPyramid[0](x, y) + 1.772f * cb(x, y)), 0.0f, 1.0f);
		//output16(x, y, c) = clamp(select(
		//	c == 0, grey(x, y) + mult16(1.403f, cr(x, y) - 128),
		//	c == 1, grey(x, y) - mult16(0.714f, cr(x, y) - 128) - mult16(0.344f, cb(x, y) - 128),
		//	grey(x, y) + mult16(1.773f, cb(x, y) - 128)), 0, 255);
		output16(x, y, c) = select(changeC2[0](x, y) > 16383, 255, 0);
		output(x, y, c) = type != UInt(16) ? cast<uint8_t>(output16(x, y, c)) : output16(x, y, c);
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

inline void innerScheduleX86(Func f, Var x, Var y, bool parallel = false)
{
	f.vectorize(x, 8);
	if (parallel)
		f.parallel(y, 4);
}

void RieszMagnifier::scheduleX86(bool tile)
{
	const int VECTOR_SIZE = 8;

	// Schedule
	if (channels == 3)
		output.reorder(c, x, y).bound(c, 0, channels).unroll(c);

	output.vectorize(x, VECTOR_SIZE);
	if (tile)
	{
		output.tile(x, y, xi, yi, 20, 80);
	}
	output.parallel(x);

	for (int j = 0; j < pyramidLevels; j++)
	{
		bool computeAt = tile && j <= 1;
		if (computeAt)
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

			changeCAmp[j].compute_at(output, x);
			changeSAmp[j].compute_at(output, x);

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

			lPyramidCopy[j].compute_at(output, x);
			lPyramid[j].compute_at(output, x);
			lPyramidUpX[j].compute_at(output, x);
		}
		else
		{
			outGPyramid[j].compute_root();
			outGPyramidUpX[j].compute_root();

			ampReg[j].compute_root();
			ampRegX[j].compute_root();
			changeCReg[j].compute_root();
			changeCRegX[j].compute_root();
			changeSReg[j].compute_root();
			changeSRegX[j].compute_root();
			amp[j].compute_root();

			changeCAmp[j].compute_root();
			changeSAmp[j].compute_root();

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

			lPyramidCopy[j].compute_root();
			lPyramid[j].compute_root();
			lPyramidUpX[j].compute_root();
		}

		if (j > 0)
		{
			gPyramid[j].compute_root();
			gPyramidDownX[j].compute_root();
		}

		if (j <= 4)
		{
			// If computeAt, don't parallelize since it's not necessary.
			innerScheduleX86(outGPyramid[j], x, y, !computeAt);
			innerScheduleX86(outGPyramidUpX[j], x, y, !computeAt);

			innerScheduleX86(ampReg[j], x, y, !computeAt);
			innerScheduleX86(ampRegX[j], x, y, !computeAt);
			innerScheduleX86(changeCReg[j], x, y, !computeAt);
			innerScheduleX86(changeCRegX[j], x, y, !computeAt);
			innerScheduleX86(changeSReg[j], x, y, !computeAt);
			innerScheduleX86(changeSRegX[j], x, y, !computeAt);
			innerScheduleX86(amp[j], x, y, !computeAt);

			innerScheduleX86(changeCAmp[j], x, y, !computeAt);
			innerScheduleX86(changeSAmp[j], x, y, !computeAt);

			innerScheduleX86(lowpass1C[j], x, y, !computeAt);
			innerScheduleX86(lowpass2C[j], x, y, !computeAt);
			innerScheduleX86(lowpass1S[j], x, y, !computeAt);
			innerScheduleX86(lowpass2S[j], x, y, !computeAt);

			innerScheduleX86(phaseC[j], x, y, !computeAt);
			innerScheduleX86(phaseS[j], x, y, !computeAt);
			innerScheduleX86(phi[j], x, y, !computeAt);

			innerScheduleX86(lPyramid[j], x, y, !computeAt);
			innerScheduleX86(lPyramidUpX[j], x, y, !computeAt);
			if (j > 0)
			{
				innerScheduleX86(gPyramid[j], x, y, true);
				innerScheduleX86(gPyramidDownX[j], x, y, true);
			}
		}
	}

	// The final level
	gPyramid[pyramidLevels].compute_root();
	innerScheduleX86(gPyramid[pyramidLevels], x, y, true);
}

inline void innerScheduleARM(Func f, Var x, Var y, bool parallel = false)
{
	f.vectorize(x, 4);
	if (parallel)
		f.parallel(y, 4);
}

void RieszMagnifier::scheduleARM(bool tile)
{
	const int VECTOR_SIZE = 4;

	// Schedule
	if (channels == 3)
		output.reorder(c, x, y).bound(c, 0, channels).unroll(c);

	output.vectorize(x, VECTOR_SIZE);
	if (tile)
	{
		output.tile(x, y, xi, yi, 40, 80);
	}
	output.parallel(x);

	for (int j = 0; j < pyramidLevels; j++)
	{
		bool computeAt = tile && j <= 1;
		if (computeAt)
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

			changeCAmp[j].compute_at(output, x);
			changeSAmp[j].compute_at(output, x);

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

			lPyramidCopy[j].compute_at(output, x);
			lPyramid[j].compute_at(output, x);
			lPyramidUpX[j].compute_at(output, x);
		}
		else
		{
			outGPyramid[j].compute_root();
			outGPyramidUpX[j].compute_root();

			ampReg[j].compute_root();
			ampRegX[j].compute_root();
			changeCReg[j].compute_root();
			changeCRegX[j].compute_root();
			changeSReg[j].compute_root();
			changeSRegX[j].compute_root();
			amp[j].compute_root();

			changeCAmp[j].compute_root();
			changeSAmp[j].compute_root();

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

			lPyramidCopy[j].compute_root();
			lPyramid[j].compute_root();
			lPyramidUpX[j].compute_root();
		}

		if (j > 0)
		{
			gPyramid[j].compute_root();
			gPyramidDownX[j].compute_root();
		}

		if (j <= 4)
		{
			// If computeAt, don't parallelize since it's not necessary.
			innerScheduleARM(outGPyramid[j], x, y, !computeAt);
			innerScheduleARM(outGPyramidUpX[j], x, y, !computeAt);

			innerScheduleARM(ampReg[j], x, y, !computeAt);
			innerScheduleARM(ampRegX[j], x, y, !computeAt);
			innerScheduleARM(changeCReg[j], x, y, !computeAt);
			innerScheduleARM(changeCRegX[j], x, y, !computeAt);
			innerScheduleARM(changeSReg[j], x, y, !computeAt);
			innerScheduleARM(changeSRegX[j], x, y, !computeAt);
			innerScheduleARM(amp[j], x, y, !computeAt);

			innerScheduleARM(changeCAmp[j], x, y, !computeAt);
			innerScheduleARM(changeSAmp[j], x, y, !computeAt);

			innerScheduleARM(lowpass1C[j], x, y, !computeAt);
			innerScheduleARM(lowpass2C[j], x, y, !computeAt);
			innerScheduleARM(lowpass1S[j], x, y, !computeAt);
			innerScheduleARM(lowpass2S[j], x, y, !computeAt);

			innerScheduleARM(phaseC[j], x, y, !computeAt);
			innerScheduleARM(phaseS[j], x, y, !computeAt);
			innerScheduleARM(phi[j], x, y, !computeAt);

			innerScheduleARM(lPyramid[j], x, y, !computeAt);
			innerScheduleARM(lPyramidUpX[j], x, y, !computeAt);
			if (j > 0)
			{
				innerScheduleARM(gPyramid[j], x, y, true);
				innerScheduleARM(gPyramidDownX[j], x, y, true);
			}
		}
	}

	// The final level
	gPyramid[pyramidLevels].compute_root();
	innerScheduleARM(gPyramid[pyramidLevels], x, y, true);
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

void RieszMagnifier::bindJIT(float a1, float a2, float b0, float b1, float b2, float alpha,
	std::vector<Halide::Image<int16_t>> historyBuffer)
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
