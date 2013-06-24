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

#endif

/* Qualifiers for kernel code shared by CPU and GPU */

#ifndef __KERNEL_GPU__

#define __device static inline
#define __device_noinline static
#define __global
#define __local
#define __shared
#define __constant

#if defined(_WIN32) && !defined(FREE_WINDOWS)
#define __device_inline static __forceinline
#define __align(...) __declspec(align(__VA_ARGS__))
#define __may_alias
#else
#define __device_inline static inline __attribute__((always_inline))
#ifndef FREE_WINDOWS64
#define __forceinline inline __attribute__((always_inline))
#endif
#define __align(...) __attribute__((aligned(__VA_ARGS__)))
#define __may_alias __attribute__((__may_alias__))
#endif

#endif

/* Bitness */

#if defined(__ppc64__) || defined(__PPC64__) || defined(__x86_64__) || defined(__ia64__) || defined(_M_X64)
#define __KERNEL_64_BIT__
#endif

/* SIMD Types */

#ifndef __KERNEL_GPU__

/* not enabled, globally applying it just gives slowdown,
 * but useful for testing. */
#ifndef __KERNEL_SSE_DISABLED__
#ifndef __KERNEL_SSE__
#define __KERNEL_SSE__
#endif
#endif

#ifdef __KERNEL_SSE__

#include <xmmintrin.h> /* SSE 1 */

#ifdef __KERNEL_SSE2__
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

//#ifndef __KERNEL_SSE2__
//#define __KERNEL_SSE2__
//#endif

//#ifndef __KERNEL_SSE3__
//#define __KERNEL_SSE3__
//#endif

//#ifndef __KERNEL_SSSE3__
//#define __KERNEL_SSSE3__
//#endif

//#ifndef __KERNEL_SSE4__
//#define __KERNEL_SSE4__
//#endif

#else

#if defined(__x86_64__) || defined(__KERNEL_SSSE3__)

/* MinGW64 has conflicting declarations for these SSE headers in <windows.h>.
 * Since we can't avoid including <windows.h>, better only include that */
#ifdef FREE_WINDOWS64
#include <windows.h>
#else
#include <xmmintrin.h> /* SSE 1 */
#include <emmintrin.h> /* SSE 2 */

#ifdef __KERNEL_SSE3__
#include <pmmintrin.h> /* SSE 3 */
#endif
#ifdef __KERNEL_SSSE3__
#include <tmmintrin.h> /* SSSE 3 */
#endif
#endif

#ifndef __KERNEL_SSE2__
#define __KERNEL_SSE2__
#endif

#endif

#endif

/* int8_t, uint16_t, and friends */
#ifndef _WIN32
#include <stdint.h>
#endif

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

#if defined __KERNEL_SSE__
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
	__forceinline operator bool() { return x && y; }
};

struct __align(4) uchar3 {
	uchar x, y, z;

	typedef uchar value_type;

	__forceinline uchar3() {}
	__forceinline explicit uchar3(uchar n) : x(n), y(n), z(n) {}
	__forceinline uchar3(uchar x, uchar y, uchar z) : x(x), y(y), z(z) {}
	__forceinline uchar operator[](int i) const { return *(&x + i); }
	__forceinline uchar& operator[](int i) { return *(&x + i); }
	__forceinline operator bool() { return x && y && z; }
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
	__forceinline operator bool() const { return x && y && z && w; }
};

struct __align(8) int2 {
	int x, y;

	typedef int value_type;

	__forceinline int2() {}
	__forceinline explicit int2(int n) : x(n), y(n) {}
	__forceinline int2(int x, int y) : x(x), y(y) {}
	__forceinline int operator[](int i) const { return *(&x + i); }
	__forceinline int& operator[](int i) { return *(&x + i); }
	__forceinline operator bool() const { return x && y; }
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
	__forceinline int3(__m128i a) : m128(a) {}
	__forceinline operator const __m128i&(void) const { return m128; }
	__forceinline operator __m128i&(void) { return m128; }


	__forceinline operator bool() const
	{
		return _mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) == 0;
	}
#else
	__forceinline explicit int3(int n) : x(n), y(n), z(n) {}
	__forceinline int3(int x, int y, int z) : x(x), y(y), z(z) {}
	__forceinline operator bool() const { return x && y && z; }
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

	__forceinline operator bool() const
	{
		return _mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) == 0;
	}
#else
	__forceinline explicit int4(int n) : x(n), y(n), z(n), w(n) {}
	__forceinline int4(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {}
	__forceinline int4(const int3 &a, int w) : x(a.x), y(a.y), z(a.z), w(w) {}
	__forceinline operator bool() const { return x && y && z && w; }
#endif

	__forceinline int operator[](int i) const { return *(&x + i); }
	__forceinline int& operator[](int i) { return *(&x + i); }
};

struct __align(8) uint2 {
	uint x, y;

	typedef uint value_type;

	__forceinline uint2() {}
	__forceinline explicit uint2(uint n) : x(n), y(n) {}
	__forceinline uint2(uint x, uint y) : x(x), y(y) {}
	__forceinline uint operator[](uint i) const { return *(&x + i); }
	__forceinline uint& operator[](uint i) { return *(&x + i); }
	__forceinline operator bool() const { return x && y; }
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

	__forceinline operator bool() const
	{
		return (_mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) & 0xFFF) == 0;
	}
#else
	__forceinline explicit uint3(uint n) : x(n), y(n), z(n) {}
	__forceinline uint3(uint x, uint y, uint z) : x(x), y(y), z(z) {}
	__forceinline operator bool() const { return x && y && z; }
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

	__forceinline operator bool() const
	{
		return _mm_movemask_epi8(_mm_cmpeq_epi32(m128, _mm_setzero_si128())) == 0;
	}
#else
	__forceinline explicit uint4(uint n) : x(n), y(n), z(n), w(n) {}
	__forceinline uint4(uint x, uint y, uint z, uint w) : x(x), y(y), z(z), w(w) {}
	__forceinline uint4(const uint3 &a, uint w) : x(a.x), y(a.y), z(a.z), w(w) {}
	__forceinline operator bool() const { return x && y && z && w; }
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
	__forceinline operator bool() const { return x && y; }
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
	__forceinline operator const __m128&(void) const { return m128; }
	__forceinline operator __m128&(void) { return m128; }
#else
	__forceinline explicit float3(float n) : x(n), y(n), z(n) {}
	__forceinline float3(float x, float y, float z) : x(x), y(y), z(z) {}
	__forceinline operator bool() const { return x && y && z; }
#endif

	__forceinline float operator[](int i) const { return *(&x + i); }
	__forceinline float& operator[](int i) { return *(&x + i); }
};

struct __align(16) float4 {
#ifdef __KERNEL_SSE__
	union {
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
#ifdef __KERNEL_SSE4__
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
	__forceinline operator bool() const { return x && y && z && w; }
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

__device_inline uchar2 make_uchar2(uchar n)
{
	return uchar2(n);
}

__device_inline uchar2 make_uchar2(uchar x, uchar y)
{
	return uchar2(x, y);
}

__device_inline uchar3 make_uchar3(uchar n)
{
	return uchar3(n);
}

__device_inline uchar3 make_uchar3(uchar x, uchar y, uchar z)
{
	return uchar3(x, y, z);
}

__device_inline uchar4 make_uchar4(uchar n)
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

__device_inline uint2 make_uint2(uint n)
{
	return uint2(n);
}

__device_inline uint2 make_uint2(uint x, uint y)
{
	return uint2(x, y);
}

__device_inline uint3 make_uint3(uint n)
{
	return uint3(n);
}

__device_inline uint3 make_uint3(uint x, uint y, uint z)
{
	return uint3(x, y, z);
}

__device_inline uint4 make_uint4(uint n)
{
	return uint4(n);
}

__device_inline uint4 make_uint4(uint x, uint y, uint z, uint w)
{
	return uint4(x, y, z, w);
}

__device_inline uint4 make_uint4(const uint3 &a, uint w)
{
	return uint4(a, w);
}

/* int constructors */

__device_inline int2 make_int2(int n)
{
	return int2(n);
}

__device_inline int2 make_int2(int x, int y)
{
	return int2(x, y);
}

__device_inline int3 make_int3(int n)
{
	return int3(n);
}

__device_inline int3 make_int3(int x, int y, int z)
{
	return int3(x, y, z);
}

__device_inline int4 make_int4(int n)
{
	return int4(n);
}

__device_inline int4 make_int4(int x, int y, int z, int w)
{
	return int4(x, y, z, w);
}

__device_inline int4 make_int4(const int3 &a, int w)
{
	return int4(a, w);
}

/* float constructors */

__device_inline float2 make_float2(float n)
{
	return float2(n);
}

__device_inline float2 make_float2(float x, float y)
{
	return float2(x, y);
}

__device_inline float3 make_float3(float n)
{
	return float3(n);
}

__device_inline float3 make_float3(float x, float y, float z)
{
	return float3(x, y, z);
}

__device_inline float4 make_float4(float n)
{
	return float4(n);
}

__device_inline float4 make_float4(float x, float y, float z, float w)
{
	return float4(x, y, z, w);
}

__device_inline float4 make_float4(const float3 &a, float w)
{
	return float4(a, w);
}

__device_inline float4 make_float4(const float3 &a)
{
	return float4(a);
}

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
	t0 = _mm_unpackhi_epi16(t0, zero);

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
	return float4(*(float4*)&a.x, *(float4*)&a.y, *(float4*)&a.z, *(float4*)&a.w);
#endif
}

__device_inline int4 as_int4(const float4 &a)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(a);
#else
	return int4(*(int4*)&a.x, *(int4*)&a.y, *(int4*)&a.z, *(int4*)&a.w);
#endif
}

/*  */

__device_inline int align_up(int offset, int alignment)
{
	return (offset + alignment - 1) & ~(alignment - 1);
}

/* result.N = (mask.N == 0xFFFFFFFF ? b.N : a.N).
 * mask members must be either 0x00000000 or 0xFFFFFFFF */

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
/* FIXME: SSE optimize */
__device_inline uint3 mask_select(const uint3& mask, const uint3& true_val, const uint3& false_val)
{
	uint3 r;
	r.x = mask.x ? true_val.x : false_val.x;
	r.y = mask.y ? true_val.y : false_val.y;
	r.z = mask.z ? true_val.z : false_val.z;
	return r;
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
	__m128 ta = _mm_set_ss(a);
	__m128 r = _mm_rcp_ss(ta);

	// extra precision
	r = _mm_sub_ss(_mm_add_ss(r, r), _mm_mul_ss(_mm_mul_ss(r, r), ta));

	return _mm_cvtss_f32(r);
}

/* return vector of reciprocal of all members of a */
__device_inline float3 fast_rcp(float3 a)
{
#if defined __KERNEL_SSE__
	/* Set high float to 1.0f (necessary? I'm thinking infinity case has penalty) */
	__m128 mask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0, 0, 0));
	__m128 highone = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
	a = _mm_andnot_ps(mask, a);
	a = _mm_or_ps(highone, a);

	/* compute approximate reciprocal for whole vector */
	__m128 r = _mm_rcp_ps(a);

	// extra precision
	return _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), a));
#else
	return float3(1.0f / a.x, 1.0f / a.y, 1.0f / a.z);
#endif
}

#endif

#ifdef __KERNEL_SSSE3__

/* SSE shuffle utility functions */

#ifdef __KERNEL_SSSE3__

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

#else

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
	else
		return _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 1, 0));
}

#endif

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
#endif

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
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128i mask = _mm_cvtsi32_si128(0xFFFFFFFF);
	mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

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
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128i mask = _mm_cvtsi32_si128(0xFFFFFFFF);
	mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

	/* clear target element */
	__m128i t = _mm_andnot_si128(a, mask);

	/* put value into register */
	t = _mm_or_si128(t, value);
	return t;
#else
	int3 t(a);
	t[elem] = b;
	return t;
#endif
}

template<int elem>
__forceinline float4 insert(const float4& a, float b)
{
	assert(elem >= 0 && elem < 4);
#ifdef __KERNEL_SSE4__
	return _mm_insert_ps(a, _mm_set_ss(b), elem << 4);
#elif defined __KERNEL_SSE__
	/* build register with value in desired element */
	__m128 value = _mm_set_ss(b);
	value = _mm_shuffle_ps(value, value, _MM_SHUFFLE(
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128 mask = _mm_castsi128_ps(_mm_cvtsi32_si128(0xFFFFFFFF));
	mask = _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

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

template<int elem>
__forceinline float3 &insert(float3& a, float b)
{
	assert(elem >= 0 && elem < 3);
#ifdef __KERNEL_SSE4__
	a = _mm_insert_ps(a, _mm_set_ss(b), elem << 4);
#elif defined __KERNEL_SSE__
	/* build register with value in desired element */
	__m128 value = _mm_set_ss(b);
	value = _mm_shuffle_ps(value, value, _MM_SHUFFLE(
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

	/* build mask with 0xFFFFFFFF in desired element */
	__m128 mask = _mm_castsi128_ps(_mm_cvtsi32_si128(0xFFFFFFFF));
	mask = _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(
			elem != 0 ? 1 : 0,
			elem != 1 ? 1 : 0,
			elem != 2 ? 1 : 0,
			elem != 3 ? 1 : 0));

	/* clear target element */
	a = _mm_andnot_ps(mask, a);

	/* put value into register */
	a = _mm_or_ps(a, value);
#else
	float3 t(a);
	a[elem] = b;
	return t;
#endif
}

#ifndef __KERNEL_GPU__

static inline void *malloc_aligned(size_t size, size_t alignment)
{
	void *data = (void*)malloc(size + sizeof(void*) + alignment - 1);

	union { void *ptr; size_t offset; } u;
	u.ptr = (char*)data + sizeof(void*);
	u.offset = (u.offset + alignment - 1) & ~(alignment - 1);
	*(((void**)u.ptr) - 1) = data;

	return u.ptr;
}

static inline void free_aligned(void *ptr)
{
	if(ptr) {
		void *data = *(((void**)ptr) - 1);
		free(data);
	}
}

#endif

CCL_NAMESPACE_END

#endif /* __UTIL_TYPES_H__ */

