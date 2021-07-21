#ifndef AD_H
#define AD_H

#include <cuda_runtime.h>

// Struct to represent Dual numbers
template <size_t NUMVAR, class Precision>
struct Dual
{
public:
	// store real and dual values
	Precision			 real;
	Precision dual[NUMVAR] {};

	// constructor
	__host__ __device__ Dual() {}

	__host__ __device__ Dual(Precision value, int i)
		: real(value)	{dual[i] = (Precision)1.0;	}

	__host__ __device__ Dual(Precision value)
		:real(value)	{}
};


//----------------- Math Operators ----------------------------

// ADD --------------------------------------------------------
// Dual + Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator + (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real + b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] + b.dual[i];
	}
	return c;
}

// Dual + Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator + (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	//Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	Dual<NUMVAR, Precision> c{ a };
	c.real += (Precision)b;
	/*c.real = a.real + (Precision)b;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i];
	}*/
	return c;
}

// Number + Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator + (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	return b + a;
}

// Substract ----------------------------------------------------------
// Dual - Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator - (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real - b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] - b.dual[i];
	}
	return c;
}

// Dual - Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator - (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	//Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	Dual<NUMVAR, Precision> c{ a };
	c.real -= (Precision)b;
	/*c.real = a.real - (Precision)b;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i];
	}*/
	return c;
}

// Number - Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator - (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = (Precision)a - b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = -b.dual[i];
	}
	return c;
}

// Multiply -----------------------------------------------------------
// Dual * Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator * (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real * b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.real * b.dual[i] + a.dual[i] * b.real;
	}
	return c;
}

// Dual * Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator * (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real * (Precision)b;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * (Precision)b;
	}
	return c;
}

// Number * Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator * (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	return b * a;
}

// Divide -----------------------------------------------
// Dual / Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator / (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real / b.real;
	Precision rD = (Precision)1.0 / (b.real * b.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = (a.dual[i] * b.real - a.real * b.dual[i]) * rD;
	}
	return c;
}

// Dual / Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator / (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	Precision rb = (Precision)1.0 / b;
	c.real = a.real * rb;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * rb;
	}
	return c;
}

// Number / Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator / (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)a);
	//return c / b;
	//return Dual<NUMVAR, Precision>((Precision)a) / b;
	c.real = (Precision)a / b.real;
	Precision rD = (Precision)1.0 / (b.real * b.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = -a * b.dual[i] * rD;
	}
	return c;
}

// ------------------ Functions -------------------------

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> sqrt(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	Precision Temp = sqrt(a.real);
	c.real = Temp;
	Temp = 1.0 / Temp;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = 0.5 * a.dual[i] * Temp;
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> pow(const Dual<NUMVAR, Precision> &a, const Precision &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = pow(a.real, b);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = b * a.dual[i] * pow(a.real, b - 1.0);
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> pow(const Precision &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = pow(a, b.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{	// Avoid log(0)
		c.dual[i] = a == 0 ? 0 : b.dual[i] * c.real * log(a);
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> pow(const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = pow(a.real, b.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{	// Avoid log(0)
		c.dual[i] = a.real == 0 ? 0 : b.real * a.dual[i] * pow(a.real, b.real - 1.0) + b.dual[i] * c.real * log(a.real);
	}
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> log(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = log(a.real);
	Precision Temp = (Precision)1.0 / a.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Temp;
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> log10(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = log10(a.real);
	Precision Temp = (Precision)1.0 / (a.real * log(10.0));
	{
		for (size_t i = 0; i < NUMVAR; i++)
		{
			c.dual[i] * a.dual[i] * Temp;
		}
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> log2(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = log10(a.real);
	Precision Temp = (Precision)1.0 / (a.real * log(2.0));
	{
		for (size_t i = 0; i < NUMVAR; i++)
		{
			c.dual[i] * a.dual[i] * Temp;
		}
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> exp(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = exp(a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * c.real;
	}
	return c;
}

// ---------------------- Trigonometric functions ----------------------

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> sin(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = sin(a.real);
	Precision Cos = cos(a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Cos;
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> asin(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = asin = (a.real);
	Precision Temp = (Precision)1.0 / sqrt(1 - a.real*a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Temp;
	}
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> cos(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = cos(a.real);
	Precision Sin = sin(a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = -a.dual[i] * Sin;
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> acos(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = acos(a.real);
	Precision Temp = -(Precision)1.0 / sqrt(1 - a.real*a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Temp;
	}
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> tan(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = tan(a.real);
	Precision Temp = (Precision)1.0/(cos(a.real) * cos(a.real));
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Temp;
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> atan(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = atan(a.real);
	Precision Temp = (Precision)1.0 / (1.0 + a.real * a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Temp;
	}
	return c;
}

#endif