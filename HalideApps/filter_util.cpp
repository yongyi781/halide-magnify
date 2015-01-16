#include "stdafx.h"
#include "filter_util.h"

static const double M_EPSILON = 1.0e-12;

/*
* helper functions
*/
bool _sort_predicate(std::complex<double> d1, std::complex<double> d2)
{
	if (std::real(d1) < std::real(d2)) {
		return true;
	}
	else if (std::real(d2) < std::real(d1)) {
		return false;
	}
	else { // real(d1) == real(d2)
		if (std::imag(d1) < std::imag(d2)) {
			return true;
		}
		else { // imag(d2) \leq imag(d1)
			return false;
		}
	}
}

bool _has_pos_imag(const std::complex<double> in)
{
	return (std::imag(in) > 0);
}

bool _has_neg_imag(const std::complex<double> in)
{
	return (std::imag(in) < 0);
}

/*
* Slow way to compute polynomial coefficients
*/
std::vector<std::complex<double>> filter_util::poly(std::vector<std::complex<double>> roots)
{
	std::vector<std::complex<double> > output(roots.size() + 1, 0.0);
	std::vector<std::complex<double> > coeffs(roots.size() + 1, 0.0);
	unsigned int currsize;
	std::complex<double> w;

	// in-place sort of roots from smallest to largest to help
	// preserve precision
	std::sort(roots.begin(), roots.end(), _sort_predicate);
	coeffs[0] = 1.0;
	currsize = 1;
	for (unsigned int k = 0; k < roots.size(); k++) {
		w = -roots[k];
		for (unsigned int j = currsize; j > 0; j--) {
			coeffs[j] = coeffs[j] * w + coeffs[j - 1];
		}
		coeffs[0] = coeffs[0] * w;
		currsize++;
	}

	// find number of positive roots
	std::vector<std::complex<double> > pos_roots = roots;
	std::vector<std::complex<double> >::iterator pos_end = std::remove_if(pos_roots.begin(), pos_roots.end(), _has_neg_imag);
	pos_roots.erase(pos_end, pos_roots.end());
	std::sort(pos_roots.begin(), pos_roots.end(), _sort_predicate);

	// find number of negative roots
	std::vector<std::complex<double> > neg_roots = roots;
	std::vector<std::complex<double> >::iterator neg_end;
	neg_end = std::remove_if(neg_roots.begin(), neg_roots.end(), _has_pos_imag);
	neg_roots.erase(neg_end, neg_roots.end());
	std::sort(neg_roots.begin(), neg_roots.end(), _sort_predicate);

	if (neg_roots.size() == pos_roots.size() &&
		std::equal(pos_roots.begin(), pos_roots.end(), neg_roots.begin())) {
		for (unsigned k = 0; k < coeffs.size(); k++) {
			output[k] = std::real(coeffs[k]);
		}
	}
	else {
		output = coeffs;
	}

	return output;
}

/*
* Return a digital filter from an analog filter using the bilinear transform.
* The bilinear transform substitutes (z-1)/(z+1) for s.
*/
void filter_util::bilinear(std::vector<std::complex<double> >& b,
	std::vector<std::complex<double> >& a,
	double fs)
{
	unsigned int D = (int)a.size() - 1;
	unsigned int N = (int)b.size() - 1;
	unsigned int M = std::max(N, D);
	unsigned int Np = M;
	unsigned int Dp = M;

	std::vector<std::complex<double> >bprime(Np + 1, 0.0);
	std::vector<std::complex<double> >aprime(Dp + 1, 0.0);
	std::complex<double> val = 0.0;

	// convert b
	for (unsigned j = 0; j < Np + 1; j++) {
		val = 0.0;
		for (unsigned i = 0; i < N + 1; i++) {
			for (unsigned k = 0; k < i + 1; k++) {
				for (unsigned l = 0; l < M - i + 1; l++) {
					if (k + l == j) {
						val = val + std::complex<double>(comb(i, k)) *
							std::complex<double>(comb(M - i, l)) *
							b[N - i] * pow(2 * fs, i)*pow(-1, k);
					}
				}
			}
		}
		bprime[j] = real(val);
	}

	// convert a
	for (unsigned j = 0; j < Dp + 1; j++) {
		val = 0.0;
		for (unsigned i = 0; i < D + 1; i++) {
			for (unsigned k = 0; k < i + 1; k++) {
				for (unsigned l = 0; l < M - i + 1; l++) {
					if (k + l == j) {
						val += std::complex<double>(comb(i, k)) *
							std::complex<double>(comb(M - i, l)) *
							a[D - i] * pow(2 * fs, i)*pow(-1, k);
					}
				}
			}
		}
		aprime[j] = real(val);
	}

	normalize(bprime, aprime);
	a = aprime;
	b = bprime;
}

/*
* Create a low-pass filter with cutoff frequency w_0 from input transfer function
* Transfer function is assumed to have real coefficients only
*/
void filter_util::lp2lp(std::vector<std::complex<double> >& b,
	std::vector<std::complex<double> >& a,
	double wo)
{
	std::vector<double> pwo;
	int d = (int)a.size();
	int n = (int)b.size();
	int M = int(std::max(double(d), double(n)));
	unsigned int start1 = int(std::max(double(n - d), 0.0));
	unsigned int start2 = int(std::max(double(d - n), 0.0));

	// std::cout << std::endl;
	// std::cout << d << ',' << n << ',' << M << std::endl;
	// std::cout << start1 << ',' << start2 << std::endl;

	for (int k = M - 1; k > -1; k--) {
		pwo.push_back(pow(wo, double(k)));
	}
	// std::cout << "pwo primed" << std::endl;


	for (unsigned int k = start2; k < pwo.size() && k - start2 < b.size(); k++) {
		b[k - start2] = b[k - start2] *
			std::complex<double>(pwo[start1]) / std::complex<double>(pwo[k]);
	}
	for (unsigned int k = start1; k < pwo.size() && k - start1 < a.size(); k++) {
		a[k - start1] = a[k - start1] *
			std::complex<double>(pwo[start1]) / std::complex<double>(pwo[k]);
	}
	normalize(b, a);
}

/*
* Compute zeros, poles and gain given filter order (assuming
* normalized Butterworth form of transfer function)
*/
void filter_util::compute_buttap(unsigned N,
	std::vector<std::complex<double> >& zeros,
	std::vector<std::complex<double> >& poles,
	double& gain)
{
	std::complex<double> j = std::complex<double>(0, 1.0);
	unsigned int k;
	for (k = 1; k < N + 1; k++) {
		std::complex<double> temp = exp(j*(2.0*k - 1) / (2.0*N)*M_PI)*j;
		poles.push_back(temp);
	}
	gain = 1.0; // gain of 1
	// zeros is unaffected, should be empty
	zeros.clear();
}

/*
* Return polynomial transfer function representation from zeros and poles
* Transfer function coefficients will always be real.
*/
void filter_util::zpk2tf(std::vector<std::complex<double> > zeros,
	std::vector<std::complex<double> > poles,
	double gain,
	std::vector<std::complex<double> >& a,
	std::vector<std::complex<double> >& b)
{
	// we only handle single dimensional arrays
	a = poly(poles);
	b = poly(zeros);
	for (unsigned int k = 0; k < b.size(); k++) {
		b[k] = b[k] * gain;
	}
}

/*
* Returns n choose k
*/
unsigned filter_util::comb(unsigned n, unsigned k)
{
	if (k > n) return 0;
	if (k * 2 > n) k = n - k;
	if (k == 0) return 1;

	unsigned result = n;
	for (unsigned i = 2; i <= k; ++i) {
		result *= (n - i + 1);
		result /= i;
	}
	return result;
}

/*
* Normalize polynomial representation of a transfer function
* Input coefficients expected to be real numbers.
* Output coefficients will also be real numbers.
*/
void filter_util::normalize(std::vector<std::complex<double> >& b,
	std::vector<std::complex<double> >& a)
{
	std::complex<double> leading_coeff;

	// remove leading zeros, avoid dividing by 0
	while (a.front() == 0.0 && a.size() > 1) {
		a.erase(a.begin());
	}
	leading_coeff = a.front();
	for (unsigned int k = 0; k < a.size(); k++) {
		a[k] = a[k] / leading_coeff;
	}
	for (unsigned int k = 0; k < b.size(); k++) {
		b[k] = b[k] / leading_coeff;
	}
}

void filter_util::butter(unsigned int N,
	std::vector<double> Wn,
	std::vector<double>& out_a,
	std::vector<double>& out_b,
	std::string btype,
	bool analog,
	std::string ftype)
{

	std::vector<double> warped;
	std::vector<std::complex<double> > a, b;

	std::vector<std::complex<double> > zeros, poles;
	double gain;

	double bw, wo; // low or high pass bandwidth, cutoff frequency
	double fs = 2.0;

	// lowercase the string parameters
	// transform(btype.begin(), btype.end(), btype.begin(), ::tolower);
	// transform(ftype.begin(), ftype.end(), ftype.begin(), ::tolower);

	if (btype.compare("lowpass") != 0) {
		throw std::string("only lowpass band type supported");
	}

	if (ftype.compare("butter") != 0) {
		throw std::string("only butterworth filter type supported.");
	}

	// pre-warp frequencies for digital filter design
	if (!analog) {
		for (unsigned int k = 0; k < Wn.size(); k++) {
			warped.push_back(2 * fs*tan(M_PI*Wn[k] / fs));
		}
	}
	else {
		warped = Wn;
	}

	// convert to low-pass prototype
	if (btype.compare("lowpass") == 0 ||
		btype.compare("highpass") == 0) {
		wo = warped[0];
	}
	else {
		bw = warped[1] - warped[0];
		wo = sqrt(warped[0] * warped[1]);
	}

	// get analog lowpass prototype (supporting butterworth only at the moment)
	if (ftype.compare("butter") == 0) {
		filter_util::compute_buttap(N, zeros, poles, gain);
	}
	else {
		throw "Exception";
	}

	// get transfer function B and A coefficients
	filter_util::zpk2tf(zeros, poles, gain, a, b);

	// transform to lowpass, bandpass, highpass or bandstop
	if (btype.compare("lowpass") == 0) {
		filter_util::lp2lp(b, a, wo);
	}
	else if (btype.compare("highpass") == 0) {
		throw "Exception";
	}
	else if (btype.compare("bandpass") == 0) {
		throw "Exception";
	}
	else { // bandstop
		throw "Exception";
	}

	// find discrete equivalent
	if (!analog) {
		filter_util::bilinear(b, a, fs);
	}

	// assign output
	out_a.clear();
	for (unsigned k = 0; k < a.size(); k++) {
		out_a.push_back(std::real(a[k]));
		//        std::cerr << "a[" << k << "]: " << std::real(a[k]) << " ";
	}
	out_b.clear();
	for (unsigned k = 0; k < b.size(); k++) {
		out_b.push_back(std::real(b[k]));
		//        std::cerr << "b[" << k << "]: " << std::real(b[k]) << " ";
	}

}

void filter_util::butterBP(unsigned int N,
	std::vector<double> Wn,
	std::vector<double>& out_a,
	std::vector<double>& out_b)
{
	std::vector<double> warped;

	double bw, wo; // low or high pass bandwidth, cutoff frequency
	double fs = 2.0;

	// pre-warp frequencies for digital filter design
	for (unsigned int k = 0; k < Wn.size(); k++) {
		warped.push_back(2 * fs*tan(M_PI*Wn[k] / fs));
	}

	bw = warped[1] - warped[0];
	wo = sqrt(warped[0] * warped[1]);

	//Transform prototype filter to desired one
	std::vector<double> A;
	A.push_back(-bw);
	A.push_back(wo);
	A.push_back(-wo);
	A.push_back(0);

	//Bilienar transform back to digital on state space rep.
	double t = 1 / fs;
	std::vector<double> t1, t2;
	t1.push_back(A[0] * t / 2 + 1);
	t1.push_back(A[1] * t / 2);
	t1.push_back(A[2] * t / 2);
	t1.push_back(A[3] * t / 2 + 1);

	t2.push_back(-A[0] * t / 2 + 1);
	t2.push_back(-A[1] * t / 2);
	t2.push_back(-A[2] * t / 2);
	t2.push_back(-A[3] * t / 2 + 1);

	double D = 1.0 / (t2[0] * t2[3] - t2[1] * t2[2]);
	A.clear();
	A.push_back(D*(t2[3] * t1[0] - t2[1] * t1[2]));
	A.push_back(D*(t2[3] * t1[1] - t2[1] * t1[3]));
	A.push_back(D*(t2[0] * t1[2] - t2[2] * t1[0]));
	A.push_back(D*(t2[0] * t1[3] - t1[1] * t2[2]));

	//Convert to coefficients
	out_a.clear();
	out_a.push_back(1.0);
	out_a.push_back(-A[0] - A[3]);
	out_a.push_back(A[0] * A[3] - A[1] * A[2]);

	//Numerator
	std::vector<std::complex<double> > kern;
	kern.push_back(std::complex<double>(1));
	kern.push_back(std::complex<double>(cos(wo), sin(wo)));
	kern.push_back(std::complex<double>(cos(2 * wo), sin(2 * wo)));
	std::complex<double> Top = kern[0] * out_a[0] + kern[1] * out_a[1] + kern[2] * out_a[2];
	std::complex<double> Bot = kern[0] - kern[2];
	double Mai = std::real(Top / Bot);
	out_b.clear();
	out_b.push_back(Mai);
	out_b.push_back(0);
	out_b.push_back(-Mai);
}

void filter_util::computeFilter(double fps, double freqCenter, double freqWidth, std::vector<double>& filterA, std::vector<double>& filterB)
{
	double lowCutoff = freqCenter - freqWidth / 2;
	double highCutoff = freqCenter + freqWidth / 2;

	if (lowCutoff < 0 || highCutoff < 0)
		throw std::invalid_argument("freqCenter - freqWidth / 2 and freqCenter + freqWidth / 2 should be positive.");

	// TODO: Should recompute the fps periodically, not assume fixed value
	filter_util::butterBP(1, { lowCutoff / (fps / 2.0), highCutoff / (fps / 2.0) }, filterA, filterB);
	filterA[1] /= filterA[0];
	filterA[2] /= filterA[0];
}
