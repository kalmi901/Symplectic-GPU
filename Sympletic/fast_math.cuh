#ifndef FAST_MATH_CUH
#define FAST_MATH_CUH

#include <cuda_runtime.h>
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif


// https://stackoverflow.com/questions/9939322/fast-1-x-division-reciprocal
// for positive numbers
__inline__ __host__ __device__ double f_div_estimate(double x) {
	union {
		double dbl;
#ifdef __cplusplus
		std::uint_least64_t ull;
#else
		uint_least64_t ull;
#endif
	} u;
	u.dbl = x;
	u.ull = (0xbfcdd6a18f6a6f52ULL - u.ull) >> 1;
	// pow( x, -0.5 )
	u.dbl *= u.dbl;             // pow( pow(x,-0.5), 2 ) = pow( x, -1 ) = 1.0 / x
	return u.dbl;
}

__inline__ __host__ __device__ double f_div(double x) {
	union {
		double dbl;
#ifdef __cplusplus
		std::uint_least64_t ull;
#else
		uint_least64_t ull;
#endif
	} u{ u.dbl = x };
	//u.dbl = x;
	u.ull = (0xbfcdd6a18f6a6f52ULL - u.ull) >> 1;
	// pow( x, -0.5 )
	u.dbl *= u.dbl;             // pow( pow(x,-0.5), 2 ) = pow( x, -1 ) = 1.0 / x
	u.dbl = 2 * u.dbl - u.dbl * u.dbl * x;		// 1st newton iter
	u.dbl = 2 * u.dbl - u.dbl * u.dbl * x;		// 2nd newton iter
	u.dbl = 2 * u.dbl - u.dbl * u.dbl * x;		// 3th newton iter
	return u.dbl;
}

__inline__ __host__ __device__ float f_div_estimate(float x) {
	union {
		float single;
#ifdef __cplusplus
		std::uint_least32_t uint;
#else
		uint_least32_t uint;
#endif 
	} u;
	u.single = x;
	u.uint = (0xbe6eb3beU - u.uint) >> 1;
	// pow( x, -0.5 )
	u.single *= u.single;       // pow( pow(x,-0.5), 2 ) = pow( x, -1 ) = 1.0 / x
	return u.single;
}

__inline__ __host__ __device__ float f_div(float x) {
	union {
		float single;
#ifdef __cplusplus
		std::uint_least32_t uint;
#else
		uint_least32_t uint;
#endif 
	} u{ u.single = x };
	//u.single = x;
	u.uint = (0xbe6eb3beU - u.uint) >> 1;
	// pow( x, -0.5 )
	u.single *= u.single;       // pow( pow(x,-0.5), 2 ) = pow( x, -1 ) = 1.0 / x
	u.single = 2 * u.single - u.single * u.single * x;		// 1st newton iter
	u.single = 2 * u.single - u.single * u.single * x;		// 2nd newton iter
	u.single = 2 * u.single - u.single * u.single * x;		// 3th newton iter
	return u.single;
}

// K., Huang & Y., Chen
// fast division

__inline__ __host__ __device__ double f_div2(double x) {
	union {
		double dbl;
#ifdef __cplusplus
		std::uint_least64_t ull;
#else
		uint_least64_t ull;
#endif
	} u{ u.dbl = x };
	u.ull = 0x7FDE9F73AABB2400 - u.ull;

	// Iterating steps
	double delta{ 1.0 - u.dbl * x };
	u.dbl += u.dbl * delta;			// 1st iteration
	
	delta = delta * delta;
	u.dbl += u.dbl * delta;			// 2dn iteration

	delta = delta * delta;
	u.dbl += u.dbl * delta;			// 3rd iteration

	return u.dbl;
}

__inline__ __host__ __device__ float f_div2(float x) {
	union {
		float single;
#ifdef __cplusplus
		std::uint_least32_t uint;
#else
		uint_least32_t uint;
#endif 
	} u{ u.single = x };
	u.uint = 0x7EF8D4FD - u.uint;

	// Iterating steps
	float delta{ 1.f - u.single * x };
	u.single += u.single * delta;		// 1st iteration

	delta = delta * delta;
	u.single += u.single * delta;		// 2nd iteration

	delta = delta * delta;
	u.single += u.single * delta;		// 3rd iteration

	return u.single;
}

#endif