/*
 * Copyright 2011, Blender Foundation.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef __UTIL_TYPES_H__
#define __UTIL_TYPES_H__

#ifndef __KERNEL_OPENCL__

#include <assert.h>
#include <stdlib.h>

#endif /* ndef __KERNEL_OPENCL__ */

/* Qualifiers for kernel code shared by CPU and GPU */

#ifndef __KERNEL_GPU__

#define __device static inline
#define __device_noinline static
#define __global
#define __local
#define __shared
#define __constant

#if defined(_WIN32) && !defined(FREE_WINDOWS)
#define __device_inline static inline __forceinline
#define __align(...) __declspec(align(__VA_ARGS__))
#define __may_alias
#else
#define __device_inline static inline __attribute__((always_inline))
#ifndef FREE_WINDOWS64
#define __forceinline inline __attribute__((always_inline))
#endif	/* ndef FREE_WINDOWS64 */
#define __align(...) __attribute__((aligned(__VA_ARGS__)))
#define __may_alias __attribute__((__may_alias__))
#endif  /* defined(_WIN32) && !defined(FREE_WINDOWS) */

#endif

/* Bitness */

#if defined(__ppc64__) || defined(__PPC64__) || defined(__x86_64__) || defined(__ia64__) || defined(_M_X64)
#define __KERNEL_64_BIT__
#endif

/* SIMD Types */

#ifndef __KERNEL_GPU__

#ifndef __KERNEL_SSE_DISABLED__

/* SSE2 is always available on x86_64 CPUs, so auto enable */
#if defined(__x86_64__) && !defined(__KERNEL_SSE2__)
#define __KERNEL_SSE2__
#endif

#ifndef __KERNEL_SSE__
#define __KERNEL_SSE__
#endif

/* newer versions of SSE imply older versions */
#ifdef __KERNEL_SSE4__
#define __KERNEL_SSSE3__
#endif

#ifdef __KERNEL_SSSE3__
#define __KERNEL_SSE3__
#endif

#ifdef __KERNEL_SSE3__
#define __KERNEL_SSE2__
#endif

#ifdef __KERNEL_SSE2__
#ifndef __KERNEL_SSE__
#define __KERNEL_SSE__
#endif
#endif

#endif	/* ndef __KERNEL_SSE_DISABLED__ */

/* SSE intrinsics headers */
#ifndef FREE_WINDOWS64

#ifdef __KERNEL_SSE2__
#include <xmmintrin.h> /* SSE 1 */
#include <emmintrin.h> /* SSE 2 */
#endif

#ifdef __KERNEL_SSE3__
#include <pmmintrin.h> /* SSE 3 */
#endif

#ifdef __KERNEL_SSSE3__
#include <tmmintrin.h> /* SSSE 3 */
#endif

#ifdef __KERNEL_SSE4__
#include <smmintrin.h> /* SSE 4 */
#endif

#else	/* ndef FREE_WINDOWS64 */

/* MinGW64 has conflicting declarations for these SSE headers in <windows.h>.
 * Since we can't avoid including <windows.h>, better only include that */
#include <windows.h>

#endif	/* ndef FREE_WINDOWS64 */

/* int8_t, uint16_t, and friends */
#ifndef _WIN32
#include <stdint.h>
#endif

CCL_NAMESPACE_BEGIN

/* Types
 *
 * Define simpler unsigned type names, and integer with defined number of bits.
 * Also vector types, named to be compatible with OpenCL builtin types, while
 * working for CUDA and C++ too. */

/* Shorter Unsigned Names */

#ifndef __KERNEL_OPENCL__

typedef unsigned char uchar;
typedef unsigned int uint;

#endif

#ifndef __KERNEL_GPU__

/* Fixed Bits Types */

#ifdef _WIN32

typedef signed char int8_t;
typedef unsigned char uint8_t;

typedef signed short int16_t;
typedef unsigned short uint16_t;

typedef signed int int32_t;
typedef unsigned int uint32_t;

typedef long long int64_t;
typedef unsigned long long uint64_t;

#ifdef __KERNEL_64_BIT__
typedef int64_t ssize_t;
#else
typedef int32_t ssize_t;
#endif

#endif

#if defined(__KERNEL_SSE__)
#define SSE_ALIGN __align(16)
#else
#define SSE_ALIGN
#endif

/* Generic Memory Pointer */

typedef uint64_t device_ptr;

/* Vector Types */

struct __align(2) uchar2 {
	uchar x, y;

	typedef uchar value_type;

	__forceinline uchar2() {}
	__forceinline explicit uchar2(uchar n) : x(n), y(n) {}
	__forceinline uchar2(uchar x, uchar y) : x(x), y(y) {}
	__forceinline uchar operator[](int i) const { return *(&x + i); }
	__forceinline uchar& operator[](int i) { return *(&x + i); }
	//__forceinline operator bool() { return x && y; }
};

struct __align(4) uchar3 {
	uchar x, y, z;

	typedef uchar value_type;

	__forceinline uchar3() {}
	__forceinline explicit uchar3(uchar n) : x(n), y(n), z(n) {}
	__forceinline uchar3(uchar x, uchar y, uchar z) : x(x), y(y), z(z) {}
	__forceinline uchar operator[](int i) const { return *(&x + i); }
	__forceinline uchar& operator[](int i) { return *(&x + i); }
	//__forceinline operator bool() { return x && y && z; }
};

struct __align(4) uchar4 {
	uchar x, y, z, w;

	typedef uchar value_type;

	__forceinline uchar4() {}
	__forceinline explicit uchar4(uchar n) : x(n), y(n), z(n), w(n) {}
	__forceinline uchar4(const uchar3 &a, uchar w) : x(a.x), y(a.y), z(a.z), w(w) {}
	__forceinline uchar4(uchar x, uchar y, uchar z, uchar w) : x(x), y(y), z(z), w(w) {}
	__forceinline uchar operator[](int i) const { return *(&x + i); }
	__forceinline uchar& operator[](int i) { return *(&x + i); }
	//__forceinline operator bool() const { return x && y && z && w; }
};

struct __align(8) int2 {
	int x, y;

	typedef int value_type;

	__forceinline int2() {}
	__forceinline explicit int2(int n) : x(n), y(n) {}
	__forceinline int2(int x, int y) : x(x), y(y) {}
	__forceinline int operator[](int i) const { return *(&x + i); }
	__forceinline int& operator[](int i) { return *(&x + i); }
	//__forceinline operator bool() const { return x && y; }
};

struct SSE_ALIGN int3 {
#ifdef __KERNEL_SSE__
	union {
		__m128i m128;
		struct { int x, y, z, w_unused; };
	};
#else
	int x, y, z, w_unused;
#endif

	typedef int value_type;

	__forceinline int3() {}

#ifdef __KERNEL_SSE__
	__forceinline int3(int x, int y, int z)
	{
		m128 = _mm_set_epi32(0, z, y, x);
	}
	__forceinline explicit int3(int n)
	{
		m128 = _mm_set1_epi32(n);
	}
	__forceinline int3(const __m128i a) : m128(a) {}
	__forceinline operator const __m128i&(void) const { return m128; }
	__forceinline operator __m128i&(void) { return m128; }


	//__forceinline operator bool() const
	//{
	//	return _mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) == 0;
	//}
#else
	__forceinline explicit int3(int n) : x(n), y(n), z(n) {}
	__forceinline int3(int x, int y, int z) : x(x), y(y), z(z) {}
	//__forceinline operator bool() const { return x && y && z; }
#endif

	__forceinline int operator[](int i) const { return *(&x + i); }
	__forceinline int& operator[](int i) { return *(&x + i); }
};

struct SSE_ALIGN int4 {
#ifdef __KERNEL_SSE__
	union {
		__m128i m128;
		struct { int x, y, z, w; };
	};
#else
	int x, y, z, w;
#endif

	typedef int value_type;

	__forceinline int4() {}

#ifdef __KERNEL_SSE__
	__forceinline int4(int x, int y, int z, int w)
	{
		m128 = _mm_set_epi32(w, z, y, x);
	}
	__forceinline explicit int4(int n)
	{
		m128 = _mm_set1_epi32(n);
	}
	__forceinline explicit int4(const int3 &a, uint w)
	{
		/* mask off a.w */
		__m128i tmp = _mm_cvtsi32_si128(0xFFFFFFFF);
		tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(0, 1, 1, 1));
		m128 = _mm_andnot_si128(tmp, a);

		/* insert w */
		tmp = _mm_cvtsi32_si128(w);
		tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(0, 1, 1, 1));
		m128 = _mm_or_si128(m128, tmp);
	}
	__forceinline int4(__m128i a) : m128(a) {}
	__forceinline operator const __m128i&(void) const { return m128; }
	__forceinline operator __m128i&(void) { return m128; }

	//__forceinline operator bool() const
	//{
	//	return _mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) == 0;
	//}
#else
	__forceinline explicit int4(int n) : x(n), y(n), z(n), w(n) {}
	__forceinline int4(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {}
	__forceinline int4(const int3 &a, int w) : x(a.x), y(a.y), z(a.z), w(w) {}
	//__forceinline operator bool() const { return x && y && z && w; }
#endif

#if defined(__GNUC__) && defined(__KERNEL_SSE__) && ((__GNUC__*10000+__GNUC_MINOR__*100+__GNUC_PATCHLEVEL__) >= 40800) && 0
	__forceinline int operator[](int i) const { return m128[i]; }
	//__forceinline int& operator[](int i) { return m128[i]; }
#else
	__forceinline int operator[](int i) const { return *(&x + i); }
	__forceinline int& operator[](int i) { return *(&x + i); }
#endif
};

struct __align(8) uint2 {
	uint x, y;

	typedef uint value_type;

	__forceinline uint2() {}
	__forceinline explicit uint2(uint n) : x(n), y(n) {}
	__forceinline uint2(uint x, uint y) : x(x), y(y) {}
	__forceinline uint operator[](uint i) const { return *(&x + i); }
	__forceinline uint& operator[](uint i) { return *(&x + i); }
	//__forceinline operator bool() const { return x && y; }
};

struct SSE_ALIGN uint3 {
#ifdef __KERNEL_SSE__
	union {
        __m128i m128;
		struct { uint x, y, z, w; };
	};
#else
	struct { uint x, y, z, w; };
#endif

	typedef uint value_type;

	__forceinline uint3() {}

#ifdef __KERNEL_SSE__
	__forceinline uint3(uint x, uint y, uint z)
	{
		m128 = _mm_set_epi32(0, (int)z, (int)y, (int)x);
	}
	__forceinline explicit uint3(uint n)
	{
		m128 = _mm_set1_epi32((int)n);
	}
	__forceinline uint3(const __m128i a) : m128(a) {}
    __forceinline operator const __m128i&(void) const { return m128; }
    __forceinline operator __m128i&(void) { return m128; }

	//__forceinline operator bool() const
	//{
	//	return (_mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) & 0xFFF) == 0;
	//}
#else
	__forceinline explicit uint3(uint n) : x(n), y(n), z(n) {}
	__forceinline uint3(uint x, uint y, uint z) : x(x), y(y), z(z) {}
	//__forceinline operator bool() const { return x && y && z; }
#endif

	__forceinline uint operator[](int i) const { return *(&x + i); }
	__forceinline uint& operator[](int i) { return *(&x + i); }
};

struct SSE_ALIGN uint4 {
#ifdef __KERNEL_SSE__
    union {
        __m128i m128;
		struct { uint x, y, z, w; };
    };
#else
	uint x, y, z, w;
#endif

	typedef uint value_type;

	__forceinline uint4() {}

#ifdef __KERNEL_SSE__
	__forceinline uint4(uint x, uint y, uint z, uint w)
	{
		m128 = _mm_set_epi32((int)w, (int)z, (int)y, (int)x);
	}
	__forceinline explicit uint4(uint n)
	{
		m128 = _mm_set1_epi32((int)n);
	}
	__forceinline explicit uint4(const uint3 &a, uint w)
	{
		/* mask off a.w */
		__m128i tmp = _mm_cvtsi32_si128(0xFFFFFFFF);
		tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(0, 1, 1, 1));
		m128 = _mm_andnot_si128(tmp, a);

		/* insert w */
		tmp = _mm_cvtsi32_si128(w);
		tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(0, 1, 1, 1));
		m128 = _mm_or_si128(m128, tmp);
	}
	__forceinline uint4(const __m128i a) : m128(a) {}
    __forceinline operator const __m128i&(void) const { return m128; }
    __forceinline operator __m128i&(void) { return m128; }

	//__forceinline operator bool() const
	//{
	//	return _mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) == 0;
	//}
#else
	__forceinline explicit uint4(uint n) : x(n), y(n), z(n), w(n) {}
	__forceinline uint4(uint x, uint y, uint z, uint w) : x(x), y(y), z(z), w(w) {}
	__forceinline uint4(const uint3 &a, uint w) : x(a.x), y(a.y), z(a.z), w(w) {}
	//__forceinline operator bool() const { return x && y && z && w; }
#endif

	__forceinline uint operator[](int i) const { return *(&x + i); }
	__forceinline uint& operator[](int i) { return *(&x + i); }
};

struct __align(8) float2 {
	float x, y;

	typedef float value_type;

	__forceinline float2() {}
	__forceinline explicit float2(float n) : x(n), y(n) {}
	__forceinline float2(float x, float y) : x(x), y(y) {}
	__forceinline float operator[](int i) const { return *(&x + i); }
	__forceinline float& operator[](int i) { return *(&x + i); }
	//__forceinline operator bool() const { return x && y; }
};

struct SSE_ALIGN float3 {
#ifdef __KERNEL_SSE__
	union {
		__m128 m128;
		struct { float x, y, z, w; };
	};
#else
	float x, y, z, w_unused;
#endif

	typedef float value_type;

	__forceinline float3() {}

#ifdef __KERNEL_SSE__
	__forceinline float3(float x, float y, float z)
	{
		m128 = _mm_set_ps(0.0f, z, y, x);
	}
	__forceinline explicit float3(float n)
	{
		m128 = _mm_set1_ps(n);
	}
	__forceinline float3(const __m128 a) : m128(a) {}
	__forceinline operator const __m128(void) const { return m128; }
	__forceinline operator __m128(void) { return m128; }
	__forceinline __m128 operator=(const __m128 r) { return m128 = r; }
#else
	__forceinline explicit float3(float n) : x(n), y(n), z(n) {}
	__forceinline float3(float x, float y, float z) : x(x), y(y), z(z) {}
	//__forceinline operator bool() const { return x && y && z; }
#endif

	__forceinline float operator[](int i) const { return *(&x + i); }
	__forceinline float& operator[](int i) { return *(&x + i); }
};

struct __align(16) float4 {
#ifdef __KERNEL_SSE__
	union __align(16) {
		__m128 m128;
		struct { float x, y, z, w; };
	};
#else
	float x, y, z, w;
#endif

	typedef float value_type;

	__forceinline float4() {}

#ifdef __KERNEL_SSE__
	__forceinline float4(float x, float y, float z, float w)
	{
		m128 = _mm_set_ps(w, z, y, x);
	}
	__forceinline explicit float4(float n)
	{
		m128 = _mm_set1_ps(n);
	}
	__forceinline explicit float4(const float3 &a, float w)
	{
#if defined __KERNEL_SSE4__
		m128 = _mm_blend_ps(a, _mm_set1_ps(w), 1 << 3);
#else
		/* mask off a.w */
		__m128 tmp = _mm_castsi128_ps(_mm_cvtsi32_si128(0xFFFFFFFF));
		tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0, 1, 1, 1));
		m128 = _mm_andnot_ps(tmp, a);

		/* insert w */
		tmp = _mm_set_ss(w);
		tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0, 1, 1, 1));
		m128 = _mm_or_ps(m128, tmp);
#endif
	}
	__forceinline explicit float4(const float3 &a)
	{
#ifdef __KERNEL_SSE4__
		m128 = _mm_blend_ps(a, _mm_setzero_ps(), 1 << 3);
#else
		m128 = float4(a, 0.0f);
#endif
	}
	__forceinline float4(const __m128 a) : m128(a) {}
	__forceinline operator const __m128&(void) const { return m128; }
	__forceinline operator __m128&(void) { return m128; }
#else
	__forceinline explicit float4(float n) : x(n), y(n), z(n), w(n) {}
	__forceinline float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	__forceinline float4(const float3 &a, float w) : x(a.x), y(a.y), z(a.z), w(w) {}
	//__forceinline operator bool() const { return x && y && z && w; }
#endif

	__forceinline float operator[](int i) const { return *(&x + i); }
	__forceinline float& operator[](int i) { return *(&x + i); }
};

#endif

#ifndef __KERNEL_GPU__

/* Vector Type Constructors
 * 
 * OpenCL does not support C++ class, so we use these instead. */

/* uchar constructors */

__device_inline uchar2 make_uchar2_1(uchar n)
{
	return uchar2(n, n);
}

__device_inline uchar2 make_uchar2(uchar x, uchar y)
{
	return uchar2(x, y);
}

__device_inline uchar3 make_uchar3_1(uchar n)
{
	return uchar3(n);
}

__device_inline uchar3 make_uchar3(uchar x, uchar y, uchar z)
{
	return uchar3(x, y, z);
}

__device_inline uchar4 make_uchar4_1(uchar n)
{
	return uchar4(n);
}

__device_inline uchar4 make_uchar4(uchar x, uchar y, uchar z, uchar w)
{
	return uchar4(x, y, z, w);
}

__device_inline uchar4 make_uchar4(const uchar3 &a, uchar w)
{
	return uchar4(a, w);
}

/* uint constructors */

__device_inline uint2 make_uint2_1(uint n)
{
	return uint2(n);
}

__device_inline uint2 make_uint2(uint x, uint y)
{
	return uint2(x, y);
}

__device_inline uint3 make_uint3_1(uint n)
{
	return uint3(n);
}

__device_inline uint3 make_uint3(uint x, uint y, uint z)
{
	return uint3(x, y, z);
}

__device_inline uint4 make_uint4_1(uint n)
{
	return uint4(n);
}

__device_inline uint4 make_uint4(uint x, uint y, uint z, uint w)
{
	return uint4(x, y, z, w);
}

__device_inline uint4 make_uint4_31(const uint3 &a, uint w)
{
	return uint4(a, w);
}

/* int constructors */

__device_inline int2 make_int2_1(int n)
{
	return int2(n);
}

__device_inline int2 make_int2(int x, int y)
{
	return int2(x, y);
}

__device_inline int3 make_int3_1(int n)
{
	return int3(n);
}

__device_inline int3 make_int3(int x, int y, int z)
{
	return int3(x, y, z);
}

__device_inline int4 make_int4_1(int n)
{
	return int4(n);
}

__device_inline int4 make_int4(int x, int y, int z, int w)
{
	return int4(x, y, z, w);
}

__device_inline int4 make_int4_31(const int3 &a, int w)
{
	return int4(a, w);
}

/* float constructors */

__device_inline float2 make_float2_1(float n)
{
	return float2(n);
}

__device_inline float2 make_float2(float x, float y)
{
	return float2(x, y);
}

#if defined(__GNUC__) && defined(__KERNEL_SSE__) && 1

//#define make_float3_1(n) float3((__m128){(n)})
//#define make_float3(x,y,z) float3((__m128){0.0f,(z),(y),(x)})

__device_inline float3 make_float3_1(float n)
{
	return float3((__m128)(__v4sf){(n),(n),(n),(n)});
}

__device_inline float3 make_float3(float x, float y, float z)
{
	return float3((__m128)(__v4sf){(x), (y), (z), 0.0f});//, (z), (y), (x)});
}

#else

__device_inline float3 make_float3_1(float n)
{
	return float3(n);
}

__device_inline float3 make_float3(float x, float y, float z)
{
	return float3(x, y, z);
}

#endif

#if defined(__GNUC__) && defined(__KERNEL_SSE2__) && 1

//#define make_float4_1(n) float4((__m128){(n)})
//#define make_float4_31(f3, s) float4((f3), (s))
//#define make_float4(x,y,z,w) float4((__m128){(w),(z),(y),(x)})

__device_inline float4 make_float4_1(float n)
{
	return float4((__m128)(__v4sf){(n),(n),(n),(n)});
}

__device_inline float4 make_float4(float x, float y, float z, float w)
{
	return float4((__m128){(x), (y), (z), (w)});
}

__device_inline float4 make_float4_31(const float3 &f3, float w)
{
	//return __builtin_shuffle(f3.m128, (__m128){(w),(w),(w),(w)}, (__v4si){0, 1, 2, 4});
	return (__m128){f3.m128[0], f3.m128[1], f3.m128[2], w};
}

#else
__device_inline float4 make_float4_1(float n)
{
	return float4(n);
}

__device_inline float4 make_float4(float x, float y, float z, float w)
{
	return float4(x, y, z, w);
}

__device_inline float4 make_float4_31(const float3 &a, float w)
{
	return float4(a.x, a.y, a.z, w);
}

__device_inline float4 make_float4(const float3 &a)
{
	return float4(a.x, a.y, a.z, 1.0f);
}
#endif

/* conversions */

__device_inline float4 convert_float4(const int4 &a)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtepi32_ps(a);
#else
	return float4(a.x, a.y, a.z, a.w);
#endif
}

__device_inline float4 convert_float4(const uchar4 &a)
{
#ifdef __KERNEL_SSE__
	__m128i t0;
	__m128i zero = _mm_setzero_si128();

	/* load as integer and get it into SSE register */
	t0 = _mm_cvtsi32_si128(*(int*)&a);

	/* zero extend 8-bit values to 32-bit */
	t0 = _mm_unpacklo_epi8(t0, zero);
	t0 = _mm_unpacklo_epi16(t0, zero);

	/* convert to float */
	return _mm_cvtepi32_ps(t0);
#else
	return float4(a.x, a.y, a.z, a.w);
#endif
}

__device_inline int4 convert_int4(const float4 &a)
{
#ifdef __KERNEL_SSE__
	return _mm_cvttps_epi32(a);
#else
	return int4(a.x, a.y, a.z, a.w);
#endif
}

/* reinterpret casts */

__device_inline float4 as_float4(const int4 &a)
{
#ifdef __KERNEL_SSE__
	return _mm_castsi128_ps(a);
#else
	union { float f; int i; } t;
	float4 r;
	t.i = a.x;
	r.x = t.f;

	t.i = a.y;
	r.y = t.f;

	t.i = a.z;
	r.z = t.f;

	t.i = a.w;
	r.w = t.f;
	return r;
#endif
}

__device_inline int4 as_int4(const float4 &a)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(a);
#else
	union { float f; int i; } t;
	int4 r;

	t.f = a.x;
	r.x = t.i;

	t.f = a.y;
	r.y = t.i;

	t.f = a.z;
	r.z = t.i;

	t.f = a.w;
	r.w = t.i;

	return r;
#endif
}

/*  */

__device_inline int align_up(int offset, int alignment)
{
	return (offset + alignment - 1) & ~(alignment - 1);
}

/* result.N = (mask.N == 0xFFFFFFFF ? b.N : a.N).
 * mask members must be either 0x00000000 or 0xFFFFFFFF */

/* non-vector, for making branchless ternary */
__device_inline int mask_select(bool cond, int true_val, int false_val)
{
	/* mask is 0xFFFFFFFF for true, 0x00000000 for false */
	int mask = -(int)cond;
	return (true_val & mask) | (false_val & ~mask);
}

/* select uchar2 */
__device_inline uchar2 mask_select(const uchar2& mask, const uchar2& true_val, const uchar2& false_val)
{
	uchar2 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	return r;
}

/* select uchar3 */
__device_inline uchar3 mask_select(const uchar3& mask, const uchar3& true_val, const uchar3& false_val)
{
	uchar3 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	return r;
}

/* select uchar4 */
__device_inline uchar4 mask_select(const uchar4& mask, const uchar4& true_val, const uchar4& false_val)
{
	uchar4 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	r.w = mask.w ? true_val.w : false_val.w;
	return r;
}

/* select uint2 */
__device_inline uint2 mask_select(const uint2& mask, const uint2& true_val, const uint2& false_val)
{
	uint2 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	return r;
}

/* select uint3 */
__device_inline uint3 mask_select(const uint3& mask, const uint3& true_val, const uint3& false_val)
{
#ifdef __KERNEL_SSE__
	__m128i true_parts = _mm_and_si128(mask, true_val);
	__m128i false_parts = _mm_andnot_si128(mask, false_val);
	return _mm_or_si128(true_parts, false_parts);
#else
	uint3 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	return r;
#endif
}

/* select uint4 */
__device_inline uint4 mask_select(const uint4& mask, const uint4& true_val, const uint4& false_val)
{
#ifdef __KERNEL_SSE__
	__m128i true_parts = _mm_and_si128(mask, true_val);
	__m128i false_parts = _mm_andnot_si128(mask, false_val);
	return _mm_or_si128(true_parts, false_parts);
#else
	uint4 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	r.w = mask.w ? true_val.w : false_val.w;
	return r;
#endif
}

/* select int2 */
__device_inline int2 mask_select(const int2& mask, const int2& true_val, const int2& false_val)
{
	int2 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	return r;
}

/* select int3 */
__device_inline int3 mask_select(const int3& mask, const int3& true_val, const int3& false_val)
{
#ifdef __KERNEL_SSE__
	__m128i false_vec = _mm_andnot_si128(mask, false_val);
	__m128i true_vec = _mm_and_si128(true_val, mask);
	return int3(_mm_or_si128(false_vec, true_vec));
#else
	int3 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	return r;
#endif
}

/* select int4 */
__device_inline int4 mask_select(const int4& mask, const int4& true_val, const int4& false_val)
{
#ifdef __KERNEL_SSE__
	__m128i false_vec = _mm_andnot_si128(mask, false_val);
	__m128i true_vec = _mm_and_si128(mask, true_val);
	return int4(_mm_or_si128(false_vec, true_vec));
#else
	int4 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	r.w = mask.w ? true_val.w : false_val.w;
	return r;
#endif
}

/* select float2 */
__device_inline float2 mask_select(const float2& mask, const float2& true_val, const float2& false_val)
{
	float2 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	return r;
}

/* select float3 */
__device_inline float3 mask_select(const int3& mask, const float3& true_val, const float3& false_val)
{
#ifdef __KERNEL_SSE__
	__m128 ps_mask = _mm_castsi128_ps(mask);
	__m128 false_vec = _mm_andnot_ps(ps_mask, false_val);
	__m128 true_vec = _mm_and_ps(ps_mask, true_val);
	return float3(_mm_or_ps(false_vec, true_vec));
#else
	float3 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	return r;
#endif
}

/* select float4 */
__device_inline float4 mask_select(int4 mask, float4 true_val, float4 false_val)
{
#ifdef __KERNEL_SSE__
	__m128 ps_mask = _mm_castsi128_ps(mask);
	__m128 false_vec = _mm_andnot_ps(ps_mask, false_val);
	__m128 true_vec = _mm_and_ps(ps_mask, true_val);
	return float4(_mm_or_ps(false_vec, true_vec));
#else
	float4 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	r.w = mask.w ? true_val.w : false_val.w;
	return r;
#endif
}

__device_inline float fast_rcp(float a)
{
#ifndef __KERNEL_SSE__
	return 1.0f / a;
#else
	__m128 ta = _mm_set_ss(a);
	__m128 r = _mm_rcp_ss(ta);

	// extra precision
	//r = _mm_sub_ss(_mm_add_ss(r, r), _mm_mul_ss(_mm_mul_ss(r, r), ta));

	return _mm_cvtss_f32(r);
#endif
}

/* return vector of reciprocal of all members of a */
__device_inline float3 fast_rcp(float3 a)
{
#if defined __KERNEL_SSE__
	/* compute approximate reciprocal for whole vector */
	__m128 r = _mm_rcp_ps(a);

	//r = _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), a));

	return r;
#else
	return float3(1.0f / a.x, 1.0f / a.y, 1.0f / a.z);
#endif
}

#endif

/* SSE shuffle utility functions */

#if defined __KERNEL_SSSE3__

/* faster version for SSSE3 */
typedef __m128i shuffle_swap_t;

__device_inline const shuffle_swap_t shuffle_swap_identity(void)
{
	return _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

__device_inline const shuffle_swap_t shuffle_swap_swap(void)
{
	return _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
}

__device_inline const __m128 shuffle_swap(const __m128& a, const shuffle_swap_t& shuf)
{
	return _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(a), shuf));
}

#elif defined __KERNEL_SSE2__

/* somewhat slower version for SSE2 */
typedef int shuffle_swap_t;

__device_inline const shuffle_swap_t shuffle_swap_identity(void)
{
	return 0;
}

__device_inline const shuffle_swap_t shuffle_swap_swap(void)
{
	return 1;
}

__device_inline const __m128 shuffle_swap(const __m128& a, shuffle_swap_t shuf)
{
	/* shuffle value must be a constant, so we need to branch */
	if(shuf)
		return _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 0, 3, 2));
	return a;
}

template<size_t i0, size_t i1, size_t i2, size_t i3>
__device_inline const __m128 shuffle(const __m128& a, const __m128& b)
{
	return _mm_shuffle_ps(a, b, _MM_SHUFFLE(i3, i2, i1, i0));
}

template<size_t i0, size_t i1, size_t i2, size_t i3>
__device_inline const __m128 shuffle(const __m128& b)
{
	return _mm_shuffle_ps(b, b, _MM_SHUFFLE(i3, i2, i1, i0));
}

template<size_t i0, size_t i1, size_t i2, size_t i3>
__device_inline const __m128i shuffle(const __m128i& b)
{
	return _mm_shuffle_epi32(b, _MM_SHUFFLE(i3, i2, i1, i0));
}

template<size_t i0, size_t i1, size_t i2>
__device_inline const __m128i shuffle(const __m128i& b)
{
	return _mm_shuffle_epi32(b, _MM_SHUFFLE(3, i2, i1, i0));
}

#else

#endif
#endif

#ifndef __KERNEL_OPENCL__

/* uchar extract */

template<int src>
__forceinline uchar extract(const uchar2& b)
{
	assert(src >= 0 && src < 2);
	return b[src];
}

template<int src>
__forceinline uchar extract(const uchar3& b)
{
	assert(src >= 0 && src < 3);
	return b[src];
}

template<int src>
__forceinline uchar extract(const uchar4& b)
{
	assert(src >= 0 && src < 4);
	return b[src];
}

/* uint extract */

template<int src>
__forceinline uint extract(const uint2& b)
{
	assert(src >= 0 && src < 2);
	return b[src];
}

template<int src>
__forceinline uint extract(const uint3& b)
{
	assert(src >= 0 && src < 3);
#ifdef __KERNEL_SSE__
	return (uint)_mm_cvtsi128_si32(_mm_shuffle_epi32(b, _MM_SHUFFLE(3, 2, 1, src)));
#else
	return b[src];
#endif
}

template<>
__forceinline uint extract<0>(const uint3& b)
{
#ifdef __KERNEL_SSE__
	return (uint)_mm_cvtsi128_si32(b);
#else
	return b.x;
#endif
}

template<int src>
__forceinline uint extract(const uint4& b)
{
	assert(src >= 0 && src < 4);
#ifdef __KERNEL_SSE__
	return (uint)_mm_cvtsi128_si32(_mm_shuffle_epi32(b, _MM_SHUFFLE(3, 2, 1, src)));
#else
	return b[src];
#endif
}

template<>
__forceinline uint extract<0>(const uint4& b)
{
#ifdef __KERNEL_SSE__
	return (uint)_mm_cvtsi128_si32(b);
#else
	return b.x;
#endif
}

/* int extract */

template<int src>
__forceinline int extract(const int2& b)
{
	assert(src >= 0 && src < 2);
	return b[src];
}

template<int src>
__forceinline int extract(const int3& b)
{
	assert(src >= 0 && src < 3);
#ifdef __KERNEL_SSE__
	return _mm_cvtsi128_si32(_mm_shuffle_epi32(b, _MM_SHUFFLE(3, 2, 1, src)));
#else
	return b[src];
#endif
}

template<>
__forceinline int extract<0>(const int3& b)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtsi128_si32(b);
#else
	return b.x;
#endif
}

template<int src>
__forceinline int extract(const int4& b)
{
	assert(src >= 0 && src < 4);
#ifdef __KERNEL_SSE__
	return _mm_cvtsi128_si32(_mm_shuffle_epi32(b, _MM_SHUFFLE(3, 2, 1, src)));
#else
	return b[src];
#endif
}

/* float extract */

template<int src>
__forceinline float extract(const float2& b)
{
	assert(src >= 0 && src < 2);
	return b[src];
}

template<int src>
__forceinline float extract(const float3& b)
{
	assert(src >= 0 && src < 3);
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(_mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 2, 1, src)));
#else
	return b[src];
#endif
}

/* specialization for element 0, which doesn't require shuffle */
template<>
__forceinline float extract<0>(const float3& b)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(b);
#else
	return b.x;
#endif
}

/* float4 extract */

template<int src>
__forceinline float extract(const float4& b)
{
	assert(src >= 0 && src < 4);
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(_mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 2, 1, src)));
#else
	return b[src];
#endif
}

/* specialization for element 0, which doesn't require shuffle */
template<>
__forceinline float extract<0>(const float4& b)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(b);
#else
	return b.x;
#endif
}

/* uchar2 insert */

template<int elem>
__forceinline uchar2 insert(const uchar2 a, uchar b)
{
	assert(elem >= 0 && elem < 2);
	uchar2 r(a);
	r[elem] = b;
	return r;
}

/* uchar3 insert */

template<int elem>
__forceinline uchar3 insert(const uchar3 a, uchar b)
{
	assert(elem >= 0 && elem < 3);
	uchar3 r(a);
	r[elem] = b;
	return r;
}

/* uchar4 insert */

template<int elem>
__forceinline uchar4 insert(const uchar4 a, uchar b)
{
	assert(elem >= 0 && elem < 4);
	uchar4 r(a);
	r[elem] = b;
	return r;
}

/* uint2 insert */

template<int elem>
__forceinline uint2 insert(const uint2 a, uint b)
{
	assert(elem >= 0 && elem < 2);
	uint2 r(a);
	r[elem] = b;
	return r;
}

/* uint3 insert */

template<int elem>
__forceinline uint3 insert(const uint3 a, uint b)
{
	assert(elem >= 0 && elem < 3);
	uint3 r(a);
	r[elem] = b;
	return r;
}

/* uint4 insert */

template<int elem>
__forceinline uint4 insert(const uint4 a, uint b)
{
	assert(elem >= 0 && elem < 4);
	uint4 r(a);
	r[elem] = b;
	return r;
}

/* int2 insert */

template<int elem>
__forceinline int2 insert(const int2 a, int b)
{
	assert(elem >= 0 && elem < 2);
	int2 r(a);
	r[elem] = b;
	return r;
}

/* int3 insert */

template<int elem>
__forceinline int3 insert(const int3 a, int b)
{
	assert(elem >= 0 && elem < 3);
#ifdef __KERNEL_SSE4__
	return _mm_insert_epi32(a, b, elem);
#elif defined __KERNEL_SSE__
	/* build register with value in desired element */
	__m128i value = _mm_cvtsi32_si128(b);
	value = _mm_shuffle_epi32(value, _MM_SHUFFLE(
			1,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128i mask = _mm_cvtsi32_si128(0xFFFFFFFF);
	mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(
			1,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* clear target element */
	__m128i t = _mm_andnot_si128(mask, a);

	/* put value into register */
	t = _mm_or_si128(t, value);

	return t;
#else
	int3 t(a);
	t[elem] = b;
	return t;
#endif
}

/* int4 insert */

template<int elem>
__forceinline int4 insert(const int4 a, int b)
{
	assert(elem >= 0 && elem < 4);
#ifdef __KERNEL_SSE4__
	return _mm_insert_epi32(a, b, elem);
#elif defined __KERNEL_SSE__
	/* build register with value in desired element */
	__m128i value = _mm_cvtsi32_si128(b);
	value = _mm_shuffle_epi32(value, _MM_SHUFFLE(
			elem != 3 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128i mask = _mm_cvtsi32_si128(0xFFFFFFFF);
	mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(
			elem != 3 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* clear target element */
	__m128i t = _mm_andnot_si128(mask, a);

	/* put value into register */
	t = _mm_or_si128(t, value);

	return t;
#else
	int4 t(a);
	t[elem] = b;
	return t;
#endif
}

/* float2 insert */

template<int elem>
__forceinline float2 insert(const float2 a, float b)
{
	assert(elem >= 0 && elem < 2);
	float2 r(a);
	r[elem] = b;
	return r;
}

/* float3 insert */

template<int elem>
__forceinline float3 insert(const float3 a, float b)
{
	assert(elem >= 0 && elem < 4);
#ifdef __KERNEL_SSE4__
	return _mm_insert_ps(a, _mm_set_ss(b), elem << 4);
#elif defined __KERNEL_SSE__
	/* build register with value in desired element */
	__m128 value = _mm_set_ss(b);
	value = _mm_shuffle_ps(value, value, _MM_SHUFFLE(
			1,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128 mask = _mm_castsi128_ps(_mm_cvtsi32_si128(0xFFFFFFFF));
	mask = _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(
			1,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* clear target element */
	__m128 t = _mm_andnot_ps(mask, a);

	/* put value into register */
	return _mm_or_ps(t, value);
#else
	float3 t(a);
	t[elem] = b;
	return t;
#endif
}

template<int elem>
__forceinline float4 insert(const float4 a, float b)
{
	assert(elem >= 0 && elem < 4);
#ifdef __KERNEL_SSE4__
	return _mm_insert_ps(a, _mm_set_ss(b), elem << 4);
#elif defined __KERNEL_SSE__
	/* build register with value in desired element */
	__m128 value = _mm_set_ss(b);
	value = _mm_shuffle_ps(value, value, _MM_SHUFFLE(
			elem != 3 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128 mask = _mm_castsi128_ps(_mm_cvtsi32_si128(0xFFFFFFFF));
	mask = _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(
			elem != 3 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 0 ? 1 : 0));

	/* clear target element */
	__m128 t = _mm_andnot_ps(mask, a);

	/* put value into register */
	return _mm_or_ps(t, value);
#else
	float4 t(a);
	t[elem] = b;
	return t;
#endif
}
#endif

#ifndef __KERNEL_GPU__

void *malloc_aligned(size_t size, size_t alignment);

void free_aligned(void *ptr);

#endif

/* single element extraction */

#ifdef __KERNEL_OPENCL__
#define S_x(v)	v.x
#define S_y(v)	v.y
#define S_z(v)	v.z
#define S_w(v)	v.w
#else
#define S_x(v)	extract<0>(v)
#define S_y(v)	extract<1>(v)
#define S_z(v)	extract<2>(v)
#define S_w(v)	extract<3>(v)
#endif

/* 2 element swizzle */

#ifdef __KERNEL_OPENCL__
#define S_xx(v) v.xx
#define S_xy(v) v
#define S_yx(v) v.yx
#define S_yy(v) v.yy
#else
#define S_xx(v) shuffle<0, 0>(v)
#define S_xy(v)               v
#define S_yx(v) shuffle<1, 0>(v)
#define S_yy(v) shuffle<1, 1>(v)
#endif

/* 3 element swizzle */

#ifdef __KERNEL_OPENCL__

#define S_xxx(v)  v.xxx
#define S_xxy(v)  v.xxy
#define S_xxz(v)  v.xxz
#define S_xyx(v)  v.xyx
#define S_xyy(v)  v.xyy
#define S_xyz(v)  v
#define S_xzx(v)  v.xzx
#define S_xzy(v)  v.xzy
#define S_xzz(v)  v.xzz
#define S_yxx(v)  v.yxx
#define S_yxy(v)  v.yxy
#define S_yxz(v)  v.yxz
#define S_yyx(v)  v.yyx
#define S_yyy(v)  v.yyy
#define S_yyz(v)  v.yyz
#define S_yzx(v)  v.yzx
#define S_yzy(v)  v.yzy
#define S_yzz(v)  v.yzz
#define S_zxx(v)  v.zxx
#define S_zxy(v)  v.zxy
#define S_zxz(v)  v.zxz
#define S_zyx(v)  v.zyx
#define S_zyy(v)  v.zyy
#define S_zyz(v)  v.zyz
#define S_zzx(v)  v.zzx
#define S_zzy(v)  v.zzy
#define S_zzz(v)  v.zzz

#else

#define S_xxx(v)  shuffle<0, 0, 0>(v)
#define S_xxy(v)  shuffle<0, 0, 1>(v)
#define S_xxz(v)  shuffle<0, 0, 2>(v)
#define S_xyx(v)  shuffle<0, 1, 0>(v)
#define S_xyy(v)  shuffle<0, 1, 1>(v)
#define S_xyz(v)                   v
#define S_xzx(v)  shuffle<0, 2, 0>(v)
#define S_xzy(v)  shuffle<0, 2, 1>(v)
#define S_xzz(v)  shuffle<0, 2, 2>(v)
#define S_yxx(v)  shuffle<1, 0, 0>(v)
#define S_yxy(v)  shuffle<1, 0, 1>(v)
#define S_yxz(v)  shuffle<1, 0, 2>(v)
#define S_yyx(v)  shuffle<1, 1, 0>(v)
#define S_yyy(v)  shuffle<1, 1, 1>(v)
#define S_yyz(v)  shuffle<1, 1, 2>(v)
#define S_yzx(v)  shuffle<1, 2, 0>(v)
#define S_yzy(v)  shuffle<1, 2, 1>(v)
#define S_yzz(v)  shuffle<1, 2, 2>(v)
#define S_zxx(v)  shuffle<2, 0, 0>(v)
#define S_zxy(v)  shuffle<2, 0, 1>(v)
#define S_zxz(v)  shuffle<2, 0, 2>(v)
#define S_zyx(v)  shuffle<2, 1, 0>(v)
#define S_zyy(v)  shuffle<2, 1, 1>(v)
#define S_zyz(v)  shuffle<2, 1, 2>(v)
#define S_zzx(v)  shuffle<2, 2, 0>(v)
#define S_zzy(v)  shuffle<2, 2, 1>(v)
#define S_zzz(v)  shuffle<2, 2, 2>(v)

#endif

/* 4 element swizzle */

#ifdef __KERNEL_OPENCL__

#define S_xxxx(v)  v.xxxx
#define S_xxxy(v)  v.xxxy
#define S_xxxz(v)  v.xxxz
#define S_xxxw(v)  v.xxxw
#define S_xxyx(v)  v.xxyx
#define S_xxyy(v)  v.xxyy
#define S_xxyz(v)  v.xxyz
#define S_xxyw(v)  v.xxyw
#define S_xxzx(v)  v.xxzx
#define S_xxzy(v)  v.xxzy
#define S_xxzz(v)  v.xxzz
#define S_xxzw(v)  v.xxzw
#define S_xxwx(v)  v.xxwx
#define S_xxwy(v)  v.xxwy
#define S_xxwz(v)  v.xxwz
#define S_xxww(v)  v.xxww
#define S_xyxx(v)  v.xyxx
#define S_xyxy(v)  v.xyxy
#define S_xyxz(v)  v.xyxz
#define S_xyxw(v)  v.xyxw
#define S_xyyx(v)  v.xyyx
#define S_xyyy(v)  v.xyyy
#define S_xyyz(v)  v.xyyz
#define S_xyyw(v)  v.xyyw
#define S_xyzx(v)  v.xyzx
#define S_xyzy(v)  v.xyzy
#define S_xyzz(v)  v.xyzz
#define S_xyzw(v)  v
#define S_xywx(v)  v.xywx
#define S_xywy(v)  v.xywy
#define S_xywz(v)  v.xywz
#define S_xyww(v)  v.xyww
#define S_xzxx(v)  v.xzxx
#define S_xzxy(v)  v.xzxy
#define S_xzxz(v)  v.xzxz
#define S_xzxw(v)  v.xzxw
#define S_xzyx(v)  v.xzyx
#define S_xzyy(v)  v.xzyy
#define S_xzyz(v)  v.xzyz
#define S_xzyw(v)  v.xzyw
#define S_xzzx(v)  v.xzzx
#define S_xzzy(v)  v.xzzy
#define S_xzzz(v)  v.xzzz
#define S_xzzw(v)  v.xzzw
#define S_xzwx(v)  v.xzwx
#define S_xzwy(v)  v.xzwy
#define S_xzwz(v)  v.xzwz
#define S_xzww(v)  v.xzww
#define S_xwxx(v)  v.xwxx
#define S_xwxy(v)  v.xwxy
#define S_xwxz(v)  v.xwxz
#define S_xwxw(v)  v.xwxw
#define S_xwyx(v)  v.xwyx
#define S_xwyy(v)  v.xwyy
#define S_xwyz(v)  v.xwyz
#define S_xwyw(v)  v.xwyw
#define S_xwzx(v)  v.xwzx
#define S_xwzy(v)  v.xwzy
#define S_xwzz(v)  v.xwzz
#define S_xwzw(v)  v.xwzw
#define S_xwwx(v)  v.xwwx
#define S_xwwy(v)  v.xwwy
#define S_xwwz(v)  v.xwwz
#define S_xwww(v)  v.xwww
#define S_yxxx(v)  v.yxxx
#define S_yxxy(v)  v.yxxy
#define S_yxxz(v)  v.yxxz
#define S_yxxw(v)  v.yxxw
#define S_yxyx(v)  v.yxyx
#define S_yxyy(v)  v.yxyy
#define S_yxyz(v)  v.yxyz
#define S_yxyw(v)  v.yxyw
#define S_yxzx(v)  v.yxzx
#define S_yxzy(v)  v.yxzy
#define S_yxzz(v)  v.yxzz
#define S_yxzw(v)  v.yxzw
#define S_yxwx(v)  v.yxwx
#define S_yxwy(v)  v.yxwy
#define S_yxwz(v)  v.yxwz
#define S_yxww(v)  v.yxww
#define S_yyxx(v)  v.yyxx
#define S_yyxy(v)  v.yyxy
#define S_yyxz(v)  v.yyxz
#define S_yyxw(v)  v.yyxw
#define S_yyyx(v)  v.yyyx
#define S_yyyy(v)  v.yyyy
#define S_yyyz(v)  v.yyyz
#define S_yyyw(v)  v.yyyw
#define S_yyzx(v)  v.yyzx
#define S_yyzy(v)  v.yyzy
#define S_yyzz(v)  v.yyzz
#define S_yyzw(v)  v.yyzw
#define S_yywx(v)  v.yywx
#define S_yywy(v)  v.yywy
#define S_yywz(v)  v.yywz
#define S_yyww(v)  v.yyww
#define S_yzxx(v)  v.yzxx
#define S_yzxy(v)  v.yzxy
#define S_yzxz(v)  v.yzxz
#define S_yzxw(v)  v.yzxw
#define S_yzyx(v)  v.yzyx
#define S_yzyy(v)  v.yzyy
#define S_yzyz(v)  v.yzyz
#define S_yzyw(v)  v.yzyw
#define S_yzzx(v)  v.yzzx
#define S_yzzy(v)  v.yzzy
#define S_yzzz(v)  v.yzzz
#define S_yzzw(v)  v.yzzw
#define S_yzwx(v)  v.yzwx
#define S_yzwy(v)  v.yzwy
#define S_yzwz(v)  v.yzwz
#define S_yzww(v)  v.yzww
#define S_ywxx(v)  v.ywxx
#define S_ywxy(v)  v.ywxy
#define S_ywxz(v)  v.ywxz
#define S_ywxw(v)  v.ywxw
#define S_ywyx(v)  v.ywyx
#define S_ywyy(v)  v.ywyy
#define S_ywyz(v)  v.ywyz
#define S_ywyw(v)  v.ywyw
#define S_ywzx(v)  v.ywzx
#define S_ywzy(v)  v.ywzy
#define S_ywzz(v)  v.ywzz
#define S_ywzw(v)  v.ywzw
#define S_ywwx(v)  v.ywwx
#define S_ywwy(v)  v.ywwy
#define S_ywwz(v)  v.ywwz
#define S_ywww(v)  v.ywww
#define S_zxxx(v)  v.zxxx
#define S_zxxy(v)  v.zxxy
#define S_zxxz(v)  v.zxxz
#define S_zxxw(v)  v.zxxw
#define S_zxyx(v)  v.zxyx
#define S_zxyy(v)  v.zxyy
#define S_zxyz(v)  v.zxyz
#define S_zxyw(v)  v.zxyw
#define S_zxzx(v)  v.zxzx
#define S_zxzy(v)  v.zxzy
#define S_zxzz(v)  v.zxzz
#define S_zxzw(v)  v.zxzw
#define S_zxwx(v)  v.zxwx
#define S_zxwy(v)  v.zxwy
#define S_zxwz(v)  v.zxwz
#define S_zxww(v)  v.zxww
#define S_zyxx(v)  v.zyxx
#define S_zyxy(v)  v.zyxy
#define S_zyxz(v)  v.zyxz
#define S_zyxw(v)  v.zyxw
#define S_zyyx(v)  v.zyyx
#define S_zyyy(v)  v.zyyy
#define S_zyyz(v)  v.zyyz
#define S_zyyw(v)  v.zyyw
#define S_zyzx(v)  v.zyzx
#define S_zyzy(v)  v.zyzy
#define S_zyzz(v)  v.zyzz
#define S_zyzw(v)  v.zyzw
#define S_zywx(v)  v.zywx
#define S_zywy(v)  v.zywy
#define S_zywz(v)  v.zywz
#define S_zyww(v)  v.zyww
#define S_zzxx(v)  v.zzxx
#define S_zzxy(v)  v.zzxy
#define S_zzxz(v)  v.zzxz
#define S_zzxw(v)  v.zzxw
#define S_zzyx(v)  v.zzyx
#define S_zzyy(v)  v.zzyy
#define S_zzyz(v)  v.zzyz
#define S_zzyw(v)  v.zzyw
#define S_zzzx(v)  v.zzzx
#define S_zzzy(v)  v.zzzy
#define S_zzzz(v)  v.zzzz
#define S_zzzw(v)  v.zzzw
#define S_zzwx(v)  v.zzwx
#define S_zzwy(v)  v.zzwy
#define S_zzwz(v)  v.zzwz
#define S_zzww(v)  v.zzww
#define S_zwxx(v)  v.zwxx
#define S_zwxy(v)  v.zwxy
#define S_zwxz(v)  v.zwxz
#define S_zwxw(v)  v.zwxw
#define S_zwyx(v)  v.zwyx
#define S_zwyy(v)  v.zwyy
#define S_zwyz(v)  v.zwyz
#define S_zwyw(v)  v.zwyw
#define S_zwzx(v)  v.zwzx
#define S_zwzy(v)  v.zwzy
#define S_zwzz(v)  v.zwzz
#define S_zwzw(v)  v.zwzw
#define S_zwwx(v)  v.zwwx
#define S_zwwy(v)  v.zwwy
#define S_zwwz(v)  v.zwwz
#define S_zwww(v)  v.zwww
#define S_wxxx(v)  v.wxxx
#define S_wxxy(v)  v.wxxy
#define S_wxxz(v)  v.wxxz
#define S_wxxw(v)  v.wxxw
#define S_wxyx(v)  v.wxyx
#define S_wxyy(v)  v.wxyy
#define S_wxyz(v)  v.wxyz
#define S_wxyw(v)  v.wxyw
#define S_wxzx(v)  v.wxzx
#define S_wxzy(v)  v.wxzy
#define S_wxzz(v)  v.wxzz
#define S_wxzw(v)  v.wxzw
#define S_wxwx(v)  v.wxwx
#define S_wxwy(v)  v.wxwy
#define S_wxwz(v)  v.wxwz
#define S_wxww(v)  v.wxww
#define S_wyxx(v)  v.wyxx
#define S_wyxy(v)  v.wyxy
#define S_wyxz(v)  v.wyxz
#define S_wyxw(v)  v.wyxw
#define S_wyyx(v)  v.wyyx
#define S_wyyy(v)  v.wyyy
#define S_wyyz(v)  v.wyyz
#define S_wyyw(v)  v.wyyw
#define S_wyzx(v)  v.wyzx
#define S_wyzy(v)  v.wyzy
#define S_wyzz(v)  v.wyzz
#define S_wyzw(v)  v.wyzw
#define S_wywx(v)  v.wywx
#define S_wywy(v)  v.wywy
#define S_wywz(v)  v.wywz
#define S_wyww(v)  v.wyww
#define S_wzxx(v)  v.wzxx
#define S_wzxy(v)  v.wzxy
#define S_wzxz(v)  v.wzxz
#define S_wzxw(v)  v.wzxw
#define S_wzyx(v)  v.wzyx
#define S_wzyy(v)  v.wzyy
#define S_wzyz(v)  v.wzyz
#define S_wzyw(v)  v.wzyw
#define S_wzzx(v)  v.wzzx
#define S_wzzy(v)  v.wzzy
#define S_wzzz(v)  v.wzzz
#define S_wzzw(v)  v.wzzw
#define S_wzwx(v)  v.wzwx
#define S_wzwy(v)  v.wzwy
#define S_wzwz(v)  v.wzwz
#define S_wzww(v)  v.wzww
#define S_wwxx(v)  v.wwxx
#define S_wwxy(v)  v.wwxy
#define S_wwxz(v)  v.wwxz
#define S_wwxw(v)  v.wwxw
#define S_wwyx(v)  v.wwyx
#define S_wwyy(v)  v.wwyy
#define S_wwyz(v)  v.wwyz
#define S_wwyw(v)  v.wwyw
#define S_wwzx(v)  v.wwzx
#define S_wwzy(v)  v.wwzy
#define S_wwzz(v)  v.wwzz
#define S_wwzw(v)  v.wwzw
#define S_wwwx(v)  v.wwwx
#define S_wwwy(v)  v.wwwy
#define S_wwwz(v)  v.wwwz
#define S_wwww(v)  v.wwww

#else

#define S_xxxx(v)  shuffle<0,0,0,0>(v)
#define S_xxxy(v)  shuffle<0,0,0,1>(v)
#define S_xxxz(v)  shuffle<0,0,0,2>(v)
#define S_xxxw(v)  shuffle<0,0,0,3>(v)
#define S_xxyx(v)  shuffle<0,0,1,0>(v)
#define S_xxyy(v)  shuffle<0,0,1,1>(v)
#define S_xxyz(v)  shuffle<0,0,1,2>(v)
#define S_xxyw(v)  shuffle<0,0,1,3>(v)
#define S_xxzx(v)  shuffle<0,0,2,0>(v)
#define S_xxzy(v)  shuffle<0,0,2,1>(v)
#define S_xxzz(v)  shuffle<0,0,2,2>(v)
#define S_xxzw(v)  shuffle<0,0,2,3>(v)
#define S_xxwx(v)  shuffle<0,0,3,0>(v)
#define S_xxwy(v)  shuffle<0,0,3,1>(v)
#define S_xxwz(v)  shuffle<0,0,3,2>(v)
#define S_xxww(v)  shuffle<0,0,3,3>(v)
#define S_xyxx(v)  shuffle<0,1,0,0>(v)
#define S_xyxy(v)  shuffle<0,1,0,1>(v)
#define S_xyxz(v)  shuffle<0,1,0,2>(v)
#define S_xyxw(v)  shuffle<0,1,0,3>(v)
#define S_xyyx(v)  shuffle<0,1,1,0>(v)
#define S_xyyy(v)  shuffle<0,1,1,1>(v)
#define S_xyyz(v)  shuffle<0,1,1,2>(v)
#define S_xyyw(v)  shuffle<0,1,1,3>(v)
#define S_xyzx(v)  shuffle<0,1,2,0>(v)
#define S_xyzy(v)  shuffle<0,1,2,1>(v)
#define S_xyzz(v)  shuffle<0,1,2,2>(v)
#define S_xyzw(v)					v
#define S_xywx(v)  shuffle<0,1,3,0>(v)
#define S_xywy(v)  shuffle<0,1,3,1>(v)
#define S_xywz(v)  shuffle<0,1,3,2>(v)
#define S_xyww(v)  shuffle<0,1,3,3>(v)
#define S_xzxx(v)  shuffle<0,2,0,0>(v)
#define S_xzxy(v)  shuffle<0,2,0,1>(v)
#define S_xzxz(v)  shuffle<0,2,0,2>(v)
#define S_xzxw(v)  shuffle<0,2,0,3>(v)
#define S_xzyx(v)  shuffle<0,2,1,0>(v)
#define S_xzyy(v)  shuffle<0,2,1,1>(v)
#define S_xzyz(v)  shuffle<0,2,1,2>(v)
#define S_xzyw(v)  shuffle<0,2,1,3>(v)
#define S_xzzx(v)  shuffle<0,2,2,0>(v)
#define S_xzzy(v)  shuffle<0,2,2,1>(v)
#define S_xzzz(v)  shuffle<0,2,2,2>(v)
#define S_xzzw(v)  shuffle<0,2,2,3>(v)
#define S_xzwx(v)  shuffle<0,2,3,0>(v)
#define S_xzwy(v)  shuffle<0,2,3,1>(v)
#define S_xzwz(v)  shuffle<0,2,3,2>(v)
#define S_xzww(v)  shuffle<0,2,3,3>(v)
#define S_xwxx(v)  shuffle<0,3,0,0>(v)
#define S_xwxy(v)  shuffle<0,3,0,1>(v)
#define S_xwxz(v)  shuffle<0,3,0,2>(v)
#define S_xwxw(v)  shuffle<0,3,0,3>(v)
#define S_xwyx(v)  shuffle<0,3,1,0>(v)
#define S_xwyy(v)  shuffle<0,3,1,1>(v)
#define S_xwyz(v)  shuffle<0,3,1,2>(v)
#define S_xwyw(v)  shuffle<0,3,1,3>(v)
#define S_xwzx(v)  shuffle<0,3,2,0>(v)
#define S_xwzy(v)  shuffle<0,3,2,1>(v)
#define S_xwzz(v)  shuffle<0,3,2,2>(v)
#define S_xwzw(v)  shuffle<0,3,2,3>(v)
#define S_xwwx(v)  shuffle<0,3,3,0>(v)
#define S_xwwy(v)  shuffle<0,3,3,1>(v)
#define S_xwwz(v)  shuffle<0,3,3,2>(v)
#define S_xwww(v)  shuffle<0,3,3,3>(v)
#define S_yxxx(v)  shuffle<1,0,0,0>(v)
#define S_yxxy(v)  shuffle<1,0,0,1>(v)
#define S_yxxz(v)  shuffle<1,0,0,2>(v)
#define S_yxxw(v)  shuffle<1,0,0,3>(v)
#define S_yxyx(v)  shuffle<1,0,1,0>(v)
#define S_yxyy(v)  shuffle<1,0,1,1>(v)
#define S_yxyz(v)  shuffle<1,0,1,2>(v)
#define S_yxyw(v)  shuffle<1,0,1,3>(v)
#define S_yxzx(v)  shuffle<1,0,2,0>(v)
#define S_yxzy(v)  shuffle<1,0,2,1>(v)
#define S_yxzz(v)  shuffle<1,0,2,2>(v)
#define S_yxzw(v)  shuffle<1,0,2,3>(v)
#define S_yxwx(v)  shuffle<1,0,3,0>(v)
#define S_yxwy(v)  shuffle<1,0,3,1>(v)
#define S_yxwz(v)  shuffle<1,0,3,2>(v)
#define S_yxww(v)  shuffle<1,0,3,3>(v)
#define S_yyxx(v)  shuffle<1,1,0,0>(v)
#define S_yyxy(v)  shuffle<1,1,0,1>(v)
#define S_yyxz(v)  shuffle<1,1,0,2>(v)
#define S_yyxw(v)  shuffle<1,1,0,3>(v)
#define S_yyyx(v)  shuffle<1,1,1,0>(v)
#define S_yyyy(v)  shuffle<1,1,1,1>(v)
#define S_yyyz(v)  shuffle<1,1,1,2>(v)
#define S_yyyw(v)  shuffle<1,1,1,3>(v)
#define S_yyzx(v)  shuffle<1,1,2,0>(v)
#define S_yyzy(v)  shuffle<1,1,2,1>(v)
#define S_yyzz(v)  shuffle<1,1,2,2>(v)
#define S_yyzw(v)  shuffle<1,1,2,3>(v)
#define S_yywx(v)  shuffle<1,1,3,0>(v)
#define S_yywy(v)  shuffle<1,1,3,1>(v)
#define S_yywz(v)  shuffle<1,1,3,2>(v)
#define S_yyww(v)  shuffle<1,1,3,3>(v)
#define S_yzxx(v)  shuffle<1,2,0,0>(v)
#define S_yzxy(v)  shuffle<1,2,0,1>(v)
#define S_yzxz(v)  shuffle<1,2,0,2>(v)
#define S_yzxw(v)  shuffle<1,2,0,3>(v)
#define S_yzyx(v)  shuffle<1,2,1,0>(v)
#define S_yzyy(v)  shuffle<1,2,1,1>(v)
#define S_yzyz(v)  shuffle<1,2,1,2>(v)
#define S_yzyw(v)  shuffle<1,2,1,3>(v)
#define S_yzzx(v)  shuffle<1,2,2,0>(v)
#define S_yzzy(v)  shuffle<1,2,2,1>(v)
#define S_yzzz(v)  shuffle<1,2,2,2>(v)
#define S_yzzw(v)  shuffle<1,2,2,3>(v)
#define S_yzwx(v)  shuffle<1,2,3,0>(v)
#define S_yzwy(v)  shuffle<1,2,3,1>(v)
#define S_yzwz(v)  shuffle<1,2,3,2>(v)
#define S_yzww(v)  shuffle<1,2,3,3>(v)
#define S_ywxx(v)  shuffle<1,3,0,0>(v)
#define S_ywxy(v)  shuffle<1,3,0,1>(v)
#define S_ywxz(v)  shuffle<1,3,0,2>(v)
#define S_ywxw(v)  shuffle<1,3,0,3>(v)
#define S_ywyx(v)  shuffle<1,3,1,0>(v)
#define S_ywyy(v)  shuffle<1,3,1,1>(v)
#define S_ywyz(v)  shuffle<1,3,1,2>(v)
#define S_ywyw(v)  shuffle<1,3,1,3>(v)
#define S_ywzx(v)  shuffle<1,3,2,0>(v)
#define S_ywzy(v)  shuffle<1,3,2,1>(v)
#define S_ywzz(v)  shuffle<1,3,2,2>(v)
#define S_ywzw(v)  shuffle<1,3,2,3>(v)
#define S_ywwx(v)  shuffle<1,3,3,0>(v)
#define S_ywwy(v)  shuffle<1,3,3,1>(v)
#define S_ywwz(v)  shuffle<1,3,3,2>(v)
#define S_ywww(v)  shuffle<1,3,3,3>(v)
#define S_zxxx(v)  shuffle<2,0,0,0>(v)
#define S_zxxy(v)  shuffle<2,0,0,1>(v)
#define S_zxxz(v)  shuffle<2,0,0,2>(v)
#define S_zxxw(v)  shuffle<2,0,0,3>(v)
#define S_zxyx(v)  shuffle<2,0,1,0>(v)
#define S_zxyy(v)  shuffle<2,0,1,1>(v)
#define S_zxyz(v)  shuffle<2,0,1,2>(v)
#define S_zxyw(v)  shuffle<2,0,1,3>(v)
#define S_zxzx(v)  shuffle<2,0,2,0>(v)
#define S_zxzy(v)  shuffle<2,0,2,1>(v)
#define S_zxzz(v)  shuffle<2,0,2,2>(v)
#define S_zxzw(v)  shuffle<2,0,2,3>(v)
#define S_zxwx(v)  shuffle<2,0,3,0>(v)
#define S_zxwy(v)  shuffle<2,0,3,1>(v)
#define S_zxwz(v)  shuffle<2,0,3,2>(v)
#define S_zxww(v)  shuffle<2,0,3,3>(v)
#define S_zyxx(v)  shuffle<2,1,0,0>(v)
#define S_zyxy(v)  shuffle<2,1,0,1>(v)
#define S_zyxz(v)  shuffle<2,1,0,2>(v)
#define S_zyxw(v)  shuffle<2,1,0,3>(v)
#define S_zyyx(v)  shuffle<2,1,1,0>(v)
#define S_zyyy(v)  shuffle<2,1,1,1>(v)
#define S_zyyz(v)  shuffle<2,1,1,2>(v)
#define S_zyyw(v)  shuffle<2,1,1,3>(v)
#define S_zyzx(v)  shuffle<2,1,2,0>(v)
#define S_zyzy(v)  shuffle<2,1,2,1>(v)
#define S_zyzz(v)  shuffle<2,1,2,2>(v)
#define S_zyzw(v)  shuffle<2,1,2,3>(v)
#define S_zywx(v)  shuffle<2,1,3,0>(v)
#define S_zywy(v)  shuffle<2,1,3,1>(v)
#define S_zywz(v)  shuffle<2,1,3,2>(v)
#define S_zyww(v)  shuffle<2,1,3,3>(v)
#define S_zzxx(v)  shuffle<2,2,0,0>(v)
#define S_zzxy(v)  shuffle<2,2,0,1>(v)
#define S_zzxz(v)  shuffle<2,2,0,2>(v)
#define S_zzxw(v)  shuffle<2,2,0,3>(v)
#define S_zzyx(v)  shuffle<2,2,1,0>(v)
#define S_zzyy(v)  shuffle<2,2,1,1>(v)
#define S_zzyz(v)  shuffle<2,2,1,2>(v)
#define S_zzyw(v)  shuffle<2,2,1,3>(v)
#define S_zzzx(v)  shuffle<2,2,2,0>(v)
#define S_zzzy(v)  shuffle<2,2,2,1>(v)
#define S_zzzz(v)  shuffle<2,2,2,2>(v)
#define S_zzzw(v)  shuffle<2,2,2,3>(v)
#define S_zzwx(v)  shuffle<2,2,3,0>(v)
#define S_zzwy(v)  shuffle<2,2,3,1>(v)
#define S_zzwz(v)  shuffle<2,2,3,2>(v)
#define S_zzww(v)  shuffle<2,2,3,3>(v)
#define S_zwxx(v)  shuffle<2,3,0,0>(v)
#define S_zwxy(v)  shuffle<2,3,0,1>(v)
#define S_zwxz(v)  shuffle<2,3,0,2>(v)
#define S_zwxw(v)  shuffle<2,3,0,3>(v)
#define S_zwyx(v)  shuffle<2,3,1,0>(v)
#define S_zwyy(v)  shuffle<2,3,1,1>(v)
#define S_zwyz(v)  shuffle<2,3,1,2>(v)
#define S_zwyw(v)  shuffle<2,3,1,3>(v)
#define S_zwzx(v)  shuffle<2,3,2,0>(v)
#define S_zwzy(v)  shuffle<2,3,2,1>(v)
#define S_zwzz(v)  shuffle<2,3,2,2>(v)
#define S_zwzw(v)  shuffle<2,3,2,3>(v)
#define S_zwwx(v)  shuffle<2,3,3,0>(v)
#define S_zwwy(v)  shuffle<2,3,3,1>(v)
#define S_zwwz(v)  shuffle<2,3,3,2>(v)
#define S_zwww(v)  shuffle<2,3,3,3>(v)
#define S_wxxx(v)  shuffle<3,0,0,0>(v)
#define S_wxxy(v)  shuffle<3,0,0,1>(v)
#define S_wxxz(v)  shuffle<3,0,0,2>(v)
#define S_wxxw(v)  shuffle<3,0,0,3>(v)
#define S_wxyx(v)  shuffle<3,0,1,0>(v)
#define S_wxyy(v)  shuffle<3,0,1,1>(v)
#define S_wxyz(v)  shuffle<3,0,1,2>(v)
#define S_wxyw(v)  shuffle<3,0,1,3>(v)
#define S_wxzx(v)  shuffle<3,0,2,0>(v)
#define S_wxzy(v)  shuffle<3,0,2,1>(v)
#define S_wxzz(v)  shuffle<3,0,2,2>(v)
#define S_wxzw(v)  shuffle<3,0,2,3>(v)
#define S_wxwx(v)  shuffle<3,0,3,0>(v)
#define S_wxwy(v)  shuffle<3,0,3,1>(v)
#define S_wxwz(v)  shuffle<3,0,3,2>(v)
#define S_wxww(v)  shuffle<3,0,3,3>(v)
#define S_wyxx(v)  shuffle<3,1,0,0>(v)
#define S_wyxy(v)  shuffle<3,1,0,1>(v)
#define S_wyxz(v)  shuffle<3,1,0,2>(v)
#define S_wyxw(v)  shuffle<3,1,0,3>(v)
#define S_wyyx(v)  shuffle<3,1,1,0>(v)
#define S_wyyy(v)  shuffle<3,1,1,1>(v)
#define S_wyyz(v)  shuffle<3,1,1,2>(v)
#define S_wyyw(v)  shuffle<3,1,1,3>(v)
#define S_wyzx(v)  shuffle<3,1,2,0>(v)
#define S_wyzy(v)  shuffle<3,1,2,1>(v)
#define S_wyzz(v)  shuffle<3,1,2,2>(v)
#define S_wyzw(v)  shuffle<3,1,2,3>(v)
#define S_wywx(v)  shuffle<3,1,3,0>(v)
#define S_wywy(v)  shuffle<3,1,3,1>(v)
#define S_wywz(v)  shuffle<3,1,3,2>(v)
#define S_wyww(v)  shuffle<3,1,3,3>(v)
#define S_wzxx(v)  shuffle<3,2,0,0>(v)
#define S_wzxy(v)  shuffle<3,2,0,1>(v)
#define S_wzxz(v)  shuffle<3,2,0,2>(v)
#define S_wzxw(v)  shuffle<3,2,0,3>(v)
#define S_wzyx(v)  shuffle<3,2,1,0>(v)
#define S_wzyy(v)  shuffle<3,2,1,1>(v)
#define S_wzyz(v)  shuffle<3,2,1,2>(v)
#define S_wzyw(v)  shuffle<3,2,1,3>(v)
#define S_wzzx(v)  shuffle<3,2,2,0>(v)
#define S_wzzy(v)  shuffle<3,2,2,1>(v)
#define S_wzzz(v)  shuffle<3,2,2,2>(v)
#define S_wzzw(v)  shuffle<3,2,2,3>(v)
#define S_wzwx(v)  shuffle<3,2,3,0>(v)
#define S_wzwy(v)  shuffle<3,2,3,1>(v)
#define S_wzwz(v)  shuffle<3,2,3,2>(v)
#define S_wzww(v)  shuffle<3,2,3,3>(v)
#define S_wwxx(v)  shuffle<3,3,0,0>(v)
#define S_wwxy(v)  shuffle<3,3,0,1>(v)
#define S_wwxz(v)  shuffle<3,3,0,2>(v)
#define S_wwxw(v)  shuffle<3,3,0,3>(v)
#define S_wwyx(v)  shuffle<3,3,1,0>(v)
#define S_wwyy(v)  shuffle<3,3,1,1>(v)
#define S_wwyz(v)  shuffle<3,3,1,2>(v)
#define S_wwyw(v)  shuffle<3,3,1,3>(v)
#define S_wwzx(v)  shuffle<3,3,2,0>(v)
#define S_wwzy(v)  shuffle<3,3,2,1>(v)
#define S_wwzz(v)  shuffle<3,3,2,2>(v)
#define S_wwzw(v)  shuffle<3,3,2,3>(v)
#define S_wwwx(v)  shuffle<3,3,3,0>(v)
#define S_wwwy(v)  shuffle<3,3,3,1>(v)
#define S_wwwz(v)  shuffle<3,3,3,2>(v)
#define S_wwww(v)  shuffle<3,3,3,3>(v)

#endif

CCL_NAMESPACE_END

#endif /* __UTIL_TYPES_H__ */

