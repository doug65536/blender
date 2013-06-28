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

#ifndef __UTIL_MATH_H__
#define __UTIL_MATH_H__

/* Math
 *
 * Basic math functions on scalar and vector types. This header is used by
 * both the kernel code when compiled as C++, and other C++ non-kernel code. */

#ifndef __KERNEL_OPENCL__

#ifdef _MSC_VER
#  define _USE_MATH_DEFINES
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>

#endif

#include "util_types.h"

CCL_NAMESPACE_BEGIN

/* Float Pi variations */

/* Division */
#ifndef M_PI_F
#define M_PI_F		(3.14159265358979323846264338327950288f) 		/* pi */
#endif
#ifndef M_PI_2_F
#define M_PI_2_F	(1.57079632679489661923132169163975144f) 		/* pi/2 */
#endif
#ifndef M_PI_4_F
#define M_PI_4_F	(0.785398163397448309615660845819875721f) 	/* pi/4 */
#endif
#ifndef M_1_PI_F
#define M_1_PI_F	(0.318309886183790671537767526745028724f) 	/* 1/pi */
#endif
#ifndef M_2_PI_F
#define M_2_PI_F	(0.636619772367581343075535053490057448f) 	/* 2/pi */
#endif

/* Multiplication */
#ifndef M_2PI_F
#define M_2PI_F		(6.283185307179586476925286766559005768f)		/* 2*pi */
#endif
#ifndef M_4PI_F
#define M_4PI_F		(12.56637061435917295385057353311801153f)		/* 4*pi */
#endif

/* Float sqrt variations */

#ifndef M_SQRT2_F
#define M_SQRT2_F	(1.41421356237309504880f) 					/* sqrt(2) */
#endif


/* Scalar */

#ifdef _WIN32

#ifndef __KERNEL_GPU__

#if(!defined(FREE_WINDOWS))
#define copysignf(x, y) ((float)_copysign(x, y))
#define hypotf(x, y) _hypotf(x, y)
#define isnan(x) _isnan(x)
#define isfinite(x) _finite(x)
#endif

#endif

#ifndef __KERNEL_OPENCL__

__device_inline float fmaxf(float a, float b)
{
	return (a >= b)? a: b;
}

__device_inline float fminf(float a, float b)
{
	return (a <= b)? a: b;
}

#endif

#endif

#ifndef __KERNEL_GPU__

__device_inline int max(int a, int b)
{
	return (a >= b)? a: b;
}

__device_inline uint max(uint a, uint b)
{
	return (a >= b)? a: b;
}

__device_inline int min(int a, int b)
{
	return (a <= b)? a: b;
}

__device_inline uint min(uint a, uint b)
{
	return (a <= b)? a: b;
}

__device_inline float max(float a, float b)
{
	return (a >= b)? a: b;
}

__device_inline float min(float a, float b)
{
	return (a <= b)? a: b;
}

__device_inline double max(double a, double b)
{
	return (a >= b)? a: b;
}

__device_inline double min(double a, double b)
{
	return (a <= b)? a: b;
}

#endif

__device_inline float min4(float a, float b, float c, float d)
{
	return min(min(a, b), min(c, d));
}

__device_inline float max4(float a, float b, float c, float d)
{
	return max(max(a, b), max(c, d));
}

#ifndef __KERNEL_OPENCL__

__device_inline uchar clamp(uchar a, uchar mn, uchar mx)
{
	return min(max(a, mn), mx);
}

__device_inline uint clamp(uint a, uint mn, uint mx)
{
	return min(max(a, mn), mx);
}

__device_inline int clamp(int a, int mn, int mx)
{
	return min(max(a, mn), mx);
}

__device_inline float clamp(float a, float mn, float mx)
{
	return min(max(a, mn), mx);
}

#endif

__device_inline int float_to_int(float f)
{
	/* fixed bug causing poor performance in msvc
	 * was using _mm_load_ss(&f), which always goes through memory */
#if defined(__KERNEL_SSE2__)
	return _mm_cvtt_ss2si(_mm_set_ss(f));
#else
	return (int)f;
#endif
}

__device_inline int floor_to_int(float f)
{
	return float_to_int(floorf(f));
}

__device_inline int ceil_to_int(float f)
{
	return float_to_int(ceilf(f));
}

__device_inline float signf(float f)
{
	return (f < 0.0f)? -1.0f: 1.0f;
}

__device_inline float nonzerof(float f, float eps)
{
	if(fabsf(f) < eps)
		return signf(f)*eps;
	else
		return f;
}

__device_inline float smoothstepf(float f)
{
	float ff = f*f;
	return (3.0f*ff - 2.0f*ff*f);
}

/* Float2 Vector */

#ifndef __KERNEL_OPENCL__

__device_inline bool is_zero(const float2 a)
{
	return (a.x == 0.0f && a.y == 0.0f);
}

#endif

#ifndef __KERNEL_OPENCL__

__device_inline float average(const float2 a)
{
	return (a.x + a.y)*(1.0f/2.0f);
}

#endif

#ifndef __KERNEL_OPENCL__

/* generate masks, where the element is 0xFFFFFFFF if the condition is true
 * for use with select(mask, true_vec, false_vec) */

#ifdef __KERNEL_SSE__
/* helper to invert the argument */
__forceinline __m128i sse_invert_epi32(const __m128i a)
{
	return _mm_xor_si128(a, _mm_cmpeq_epi32(a, a));
}

/* 32 bit integer multiply helper */

__forceinline __m128i sse_mul_32(const __m128i a, const __m128i b)
{
#if defined __KERNEL_SSE4__
	return _mm_mullo_epi32(a, b);
#else
	/* get y and w into separate registers */
	__m128i t0 = _mm_srli_si128(a, 4);
	__m128i t1 = _mm_srli_si128(b, 4);

	/* do x and z */
	__m128i t2 = _mm_mul_epu32(a, b);

	/* do y and w */
	__m128i t3 = _mm_mul_epu32(t0, t1);

	/* move z into second element */
	t2 = _mm_shuffle_epi32(t2, _MM_SHUFFLE (0,0,2,0));

	/* move w into second element */
	t3 = _mm_shuffle_epi32(t3, _MM_SHUFFLE (0,0,2,0));

	/* interleave results */
	return _mm_unpacklo_epi32(t2, t3);
#endif
}

/* helper macros for missing intrinsics */
#define _mm_cmple_epi32(a,b) sse_invert_epi32(_mm_cmpgt_epi32((a),(b)));
#define _mm_cmpne_epi32(a,b) sse_invert_epi32(_mm_cmpeq_epi32((a),(b)));
#define _mm_cmpge_epi32(a,b) sse_invert_epi32(_mm_cmplt_epi32((a),(b)));

/* unsigned 32 bit comparison helper */

/* note for SSE optimizations here:
 * there are no unsigned comparisons, so MIN_INT is added to both operands */

#ifdef __KERNEL_SSE__
__forceinline __m128i sse_unsigned_bias() { return _mm_set1_epi32(0x80000000); }
#endif

__forceinline __m128i _mm_cmplt_epu32(__m128i a, __m128i b)
{
	__m128i bias = sse_unsigned_bias();
	return _mm_cmplt_epi32(_mm_add_epi32(a, bias), _mm_add_epi32(b, bias));
}

__forceinline __m128i _mm_cmple_epu32(__m128i a, __m128i b)
{
	__m128i bias = sse_unsigned_bias();
	return _mm_cmple_epi32(_mm_add_epi32(a, bias), _mm_add_epi32(b, bias));
}

__forceinline __m128i _mm_cmpge_epu32(__m128i a, __m128i b)
{
	__m128i bias = sse_unsigned_bias();
	return _mm_cmpge_epi32(_mm_add_epi32(a, bias), _mm_add_epi32(b, bias));
}

__forceinline __m128i _mm_cmpgt_epu32(__m128i a, __m128i b)
{
	__m128i bias = sse_unsigned_bias();
	return _mm_cmpgt_epi32(_mm_add_epi32(a, bias), _mm_add_epi32(b, bias));
}

#endif

/* uchar2 comparisons */

__device_inline uchar2 operator<(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x < b.x, a.y < b.y);
}

__device_inline uchar2 operator<=(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x <= b.x, a.y <= b.y);
}

__device_inline uchar2 operator==(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x == b.x, a.y == b.y);
}

__device_inline uchar2 operator!=(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x != b.x, a.y != b.y);
}

__device_inline uchar2 operator>=(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x >= b.x, a.y >= b.y);
}

__device_inline uchar2 operator>(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x > b.x, a.y > b.y);
}

__device_inline bool is_equal(const uchar2 a, const uchar2 b)
{
	return a.x == b.x && a.y == b.y;
}

__device_inline bool is_notequal(const uchar2 a, const uchar2 b)
{
	return !is_equal(a, b);
}

/* uchar3 comparisons */

__device_inline uchar3 operator<(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x < b.x, a.y < b.y, a.z < b.z);
}

__device_inline uchar3 operator<=(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}

__device_inline uchar3 operator==(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x == b.x, a.y == b.y, a.z == b.z);
}

__device_inline uchar3 operator!=(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x != b.x, a.y != b.y, a.z != b.z);
}

__device_inline uchar3 operator>=(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

__device_inline uchar3 operator>(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x > b.x, a.y > b.y, a.z > b.z);
}

__device_inline bool is_equal(const uchar3 a, const uchar3 b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device_inline bool is_notequal(const uchar3 a, const uchar3 b)
{
	return !is_equal(a, b);
}

/* uchar4 comparisons */

__device_inline uchar4 operator<(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}

__device_inline uchar4 operator>(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}

__device_inline uchar4 operator>=(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
}

__device_inline uchar4 operator<=(const uchar4 a, uchar4& b)
{
	return make_uchar4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
}

__device_inline uchar4 operator==(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
}

__device_inline uchar4 operator!=(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
}

__device_inline bool is_equal(const uchar4 a, const uchar4 b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

__device_inline bool is_notequal(const uchar4 a, const uchar4 b)
{
	return !is_equal(a, b);
}

/* uint2 comparisons */

__device_inline uint2 operator<(const uint2 a, const uint2 b)
{
	return make_uint2(a.x < b.x, a.y < b.y);
}

__device_inline uint2 operator<=(const uint2 a, const uint2 b)
{
	return make_uint2(a.x <= b.x, a.y <= b.y);
}

__device_inline uint2 operator==(const uint2 a, const uint2 b)
{
	return make_uint2(a.x == b.x, a.y == b.y);
}

__device_inline uint2 operator!=(const uint2 a, const uint2 b)
{
	return make_uint2(a.x != b.x, a.y != b.y);
}

__device_inline uint2 operator>=(const uint2 a, const uint2 b)
{
	return make_uint2(a.x >= b.x, a.y >= b.y);
}

__device_inline uint2 operator>(const uint2 a, const uint2 b)
{
	return make_uint2(a.x > b.x, a.y > b.y);
}

__device_inline bool is_equal(const uint2 a, const uint2 b)
{
	return a.x == b.x && a.y == b.y;
}

__device_inline bool is_notequal(const uint2 a, const uint2 b)
{
	return !is_equal(a, b);
}

/* uint3 comparisons */

__device_inline uint3 operator<(const uint3 a, const uint3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmplt_epu32(a, b);
#else
	return make_uint3(a.x < b.x, a.y < b.y, a.z < b.z);
#endif
}

__device_inline uint3 operator<=(const uint3 a, const uint3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmple_epu32(a, b);
#else
	return make_uint3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
#endif
}

__device_inline uint3 operator==(const uint3 a, const uint3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpeq_epi32(a, b);
#else
	return make_uint3(a.x == b.x, a.y == b.y, a.z == b.z);
#endif
}

__device_inline uint3 operator!=(const uint3 a, const uint3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpne_epi32(a, b);
#else
	return make_uint3(a.x != b.x, a.y != b.y, a.z != b.z);
#endif
}

__device_inline uint3 operator>=(const uint3 a, const uint3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpge_epi32(a, b);
#else
	return make_uint3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
#endif
}

__device_inline uint3 operator>(const uint3 a, const uint3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpgt_epi32(a, b);
#else
	return make_uint3(a.x > b.x, a.y > b.y, a.z > b.z);
#endif
}

__device_inline bool is_equal(const uint3 a, const uint3 b)
{
#ifdef __KERNEL_SSE__
	return (_mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) & 0xFFF) == 0xFFFF;
#else
	return a.x == b.x && a.y == b.y && a.z == b.z;
#endif
}

__device_inline bool is_notequal(const uint3 a, const uint3 b)
{
	return !is_equal(a, b);
}

/* uint4 comparisons */

__device_inline uint4 operator<(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmplt_epu32(a, b);
#else
	return make_uint4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
#endif
}

__device_inline uint4 operator<=(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmple_epu32(a, b);
#else
	return make_uint4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
#endif
}

__device_inline uint4 operator==(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpeq_epi32(a, b);
#else
	return make_uint4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
#endif
}

__device_inline uint4 operator!=(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpne_epi32(a, b);
#else
	return make_uint4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
#endif
}

__device_inline uint4 operator>=(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpge_epu32(a, b);
#else
	return make_uint4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
#endif
}

__device_inline uint4 operator>(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpgt_epu32(a, b);
#else
	return make_uint4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
#endif
}

__device_inline bool is_equal(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) == 0xFFFF;
#else
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
#endif
}

__device_inline bool is_notequal(const uint4 a, const uint4 b)
{
	return !is_equal(a, b);
}

/* int2 comparisons */

__device_inline int2 operator<(const int2 a, const int2 b)
{
	return make_int2(a.x < b.x, a.y < b.y);
}

__device_inline int2 operator<=(const int2 a, const int2 b)
{
	return make_int2(a.x <= b.x, a.y <= b.y);
}

__device_inline int2 operator==(const int2 a, const int2 b)
{
	return make_int2(a.x == b.x, a.y == b.y);
}

__device_inline int2 operator!=(const int2 a, const int2 b)
{
	return make_int2(a.x != b.x, a.y != b.y);
}

__device_inline int2 operator>=(const int2 a, const int2 b)
{
	return make_int2(a.x >= b.x, a.y >= b.y);
}

__device_inline int2 operator>(const int2 a, const int2 b)
{
	return make_int2(a.x > b.x, a.y > b.y);
}

__device_inline bool is_equal(const int2 a, const int2 b)
{
	return a.x == b.x && a.y == b.y;
}

__device_inline bool is_notequal(const int2 a, const int2 b)
{
	return !is_equal(a, b);
}

/* int3 comparisons */

__device_inline int3 operator<(const int3 a, const int3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmplt_epi32(a, b);
#else
	return make_int3(a.x < b.x, a.y < b.y, a.z < b.z);
#endif
}

__device_inline int3 operator<=(const int3 a, const int3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmple_epi32(a, b);
#else
	return make_int3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
#endif
}

__device_inline int3 operator==(const int3 a, const int3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpeq_epi32(a, b);
#else
	return make_int3(a.x == b.x, a.y == b.y, a.z == b.z);
#endif
}

__device_inline int3 operator!=(const int3 a, const int3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpne_epi32(a, b);
#else
	return make_int3(a.x != b.x, a.y != b.y, a.z != b.z);
#endif
}

__device_inline int3 operator>=(const int3 a, const int3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpge_epi32(a, b);
#else
	return make_int3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
#endif
}

__device_inline int3 operator>(const int3 a, const int3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpgt_epi32(a, b);
#else
	return make_int3(a.x > b.x, a.y > b.y, a.z > b.z);
#endif
}

__device_inline bool is_equal(const int3 a, const int3 b)
{
#ifdef __KERNEL_SSE__
	return (_mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) & 0xFFF) == 0xFFFF;
#else
	return a.x == b.x && a.y == b.y && a.z == b.z;
#endif
}

__device_inline bool is_notequal(const int3 a, const int3 b)
{
	return !is_equal(a, b);
}

/* int4 comparisons */

__device_inline int4 operator<(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmplt_epi32(a, b);
#else
	return make_int4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
#endif
}

__device_inline int4 operator>(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpgt_epi32(a, b);
#else
	return make_int4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
#endif
}

__device_inline int4 operator>=(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpge_epi32(a, b);
#else
	return make_int4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
#endif
}

__device_inline int4 operator<=(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmple_epi32(a, b);
#else
	return make_int4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
#endif
}

__device_inline int4 operator==(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpeq_epi32(a, b);
#else
	return make_int4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
#endif
}

__device_inline int4 operator!=(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_cmpne_epi32(a, b);
#else
	return make_int4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
#endif
}

__device_inline bool is_equal(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) == 0xFFFF;
#else
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
#endif
}

__device_inline bool is_notequal(const int4 a, const int4 b)
{
	return !is_equal(a, b);
}

/* float2 comparisons */

__device_inline int2 operator<(const float2 a, const float2 b)
{
	return make_int2(a.x < b.x, a.y < b.y);
}

__device_inline int2 operator<=(const float2 a, const float2 b)
{
	return make_int2(a.x <= b.x, a.y <= b.y);
}

__device_inline int2 operator==(const float2 a, const float2 b)
{
	return make_int2(a.x == b.x, a.y == b.y);
}

__device_inline int2 operator!=(const float2 a, const float2 b)
{
	return make_int2(a.x != b.x, a.y != b.y);
}

__device_inline int2 operator>=(const float2 a, const float2 b)
{
	return make_int2(a.x >= b.x, a.y >= b.y);
}

__device_inline int2 operator>(const float2 a, const float2 b)
{
	return make_int2(a.x > b.x, a.y > b.y);
}

__device_inline bool is_equal(const float2 a, const float2 b)
{
	return a.x == b.x && a.y == b.y;
}

__device_inline bool is_notequal(const float2 a, const float2 b)
{
	return !is_equal(a, b);
}

/* float3 comparisons */

__device_inline int3 operator<(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmplt_ps(a, b));
#else
	return make_int3(a.x < b.x, a.y < b.y, a.z < b.z);
#endif
}

__device_inline int3 operator<=(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmple_ps(a, b));
#else
	return make_int3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
#endif
}

__device_inline int3 operator==(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpeq_ps(a, b));
#else
	return make_int3(a.x == b.x, a.y == b.y, a.z == b.z);
#endif
}

__device_inline int3 operator!=(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpneq_ps(a, b));
#else
	return make_int3(a.x != b.x, a.y != b.y, a.z != b.z);
#endif
}

__device_inline int3 operator>=(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpge_ps(a, b));
#else
	return make_int3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
#endif
}

__device_inline int3 operator>(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpgt_ps(a, b));
#else
	return make_int3(a.x > b.x, a.y > b.y, a.z > b.z);
#endif
}

__device_inline bool is_equal(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return (_mm_movemask_ps(_mm_cmpeq_ps(a, b)) & 0x07) == 0x7;
#else
	return (a.x == b.x && a.y == b.y && a.z == b.z);
#endif
}

__device_inline bool is_notequal(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return (_mm_movemask_ps(_mm_cmpeq_ps(a, b)) & 0x07) != 0x7;
#else
	return !(a.x == b.x && a.y == b.y && a.z == b.z);
#endif
}

/* float4 comparisons */

__device_inline int4 operator<(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmplt_ps(a, b));
#else
	return make_int4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
#endif
}

__device_inline int4 operator<=(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmple_ps(a, b));
#else
	return make_int4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
#endif
}

__device_inline int4 operator==(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpeq_ps(a, b));
#else
	return make_int4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
#endif
}

__device_inline int4 operator!=(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpneq_ps(a, b));
#else
	return make_int4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
#endif
}

__device_inline int4 operator>=(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpge_ps(a, b));
#else
	return make_int4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
#endif
}

__device_inline int4 operator>(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_castps_si128(_mm_cmpgt_ps(a, b));
#else
	return make_int4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
#endif
}

__device_inline bool is_equal(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_movemask_ps(_mm_cmpeq_ps(a, b)) == 0xF;
#else
	return (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
#endif
}

__device_inline bool is_notequal(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_movemask_ps(_mm_cmpeq_ps(a, b)) != 0xF;
#else
	return !(a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
#endif
}

#endif

#ifndef __KERNEL_OPENCL__

/* uchar2 vector */

__device_inline uchar2 min(const uchar2 a, const uchar2 b)
{
	return make_uchar2(min(a.x, b.x), min(a.y, b.y));
}

__device_inline uchar2 max(const uchar2 a, const uchar2 b)
{
	return make_uchar2(max(a.x, b.x), max(a.y, b.y));
}

__device_inline uchar2 clamp(const uchar2 a, const uchar2 mn, const uchar2 mx)
{
	return min(max(a, mn), mx);
}

__device_inline uchar2 operator-(const uchar2 a)
{
	return make_uchar2(-a.x, -a.y);
}

__device_inline uchar2 operator*(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x*b.x, a.y*b.y);
}

__device_inline uchar2 operator*(const uchar2 a, uchar f)
{
	return make_uchar2(a.x*f, a.y*f);
}

__device_inline uchar2 operator*(uchar f, const uchar2 a)
{
	return make_uchar2(a.x*f, a.y*f);
}

__device_inline uchar2 operator/(uchar f, const uchar2 a)
{
	return make_uchar2(f/a.x, f/a.y);
}

__device_inline uchar2 operator/(const uchar2 a, uchar f)
{
	return make_uchar2(a.x/f, a.y/f);
}

__device_inline uchar2 operator/(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x/b.x, a.y/b.y);
}

__device_inline uchar2 operator+(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x+b.x, a.y+b.y);
}

__device_inline uchar2 operator-(const uchar2 a, const uchar2 b)
{
	return make_uchar2(a.x-b.x, a.y-b.y);
}

__device_inline uchar2 operator>>(uchar2& a, uchar f)
{
	return make_uchar2(a.x >> f, a.y >> f);
}

__device_inline uchar2 operator<<(uchar2& a, uchar f)
{
	return make_uchar2(a.x << f, a.y << f);
}

__device_inline uchar2& operator>>=(uchar2& a, uchar f)
{
	return a = a >> f;
}

__device_inline uchar2& operator<<=(uchar2& a, uchar f)
{
	return a = a << f;
}

__device_inline uchar2& operator-=(uchar2& a, const uchar2 b)
{
	return a = a - b;
}

__device_inline uchar2& operator-=(uchar2& a, const uchar b)
{
	return a = a - make_uchar2(b);
}

__device_inline uchar2& operator+=(uchar2& a, const uchar2 b)
{
	return a = a + b;
}

__device_inline uchar2& operator+=(uchar2& a, const uchar b)
{
	return a = a + make_uchar2(b);
}

__device_inline uchar2& operator*=(uchar2& a, const uchar2 b)
{
	return a = a * b;
}

__device_inline uchar2& operator*=(uchar2& a, uchar f)
{
	return a = a * f;
}

__device_inline uchar2& operator/=(uchar2& a, const uchar2 b)
{
	return a = a / b;
}

__device_inline uchar2& operator/=(uchar2& a, uchar f)
{
	return a = a / f;
}

__device_inline uchar dot(const uchar2 a, const uchar2 b)
{
	return a.x*b.x + a.y*b.y;
}

__device_inline uchar cross(const uchar2 a, const uchar2 b)
{
	return a.x*b.y - a.y*b.x;
}

#endif

#ifndef __KERNEL_OPENCL__

/* uchar3 Vector */

__device_inline uchar3 min(const uchar3 a, const uchar3 b)
{
	return make_uchar3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

__device_inline uchar3 max(const uchar3 a, const uchar3 b)
{
	return make_uchar3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

__device_inline uchar3 clamp(const uchar3 a, const uchar3 mn, const uchar3 mx)
{
	return min(max(a, mn), mx);
}

__device_inline uchar3 operator-(const uchar3 a)
{
	return make_uchar3(-a.x, -a.y, -a.z);
}

__device_inline uchar3 operator*(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device_inline uchar3 operator*(const uchar3 a, uchar f)
{
	return make_uchar3(a.x*f, a.y*f, a.z*f);
}

__device_inline uchar3 operator*(uchar f, const uchar3 a)
{
	return make_uchar3(a.x*f, a.y*f, a.z*f);
}

__device_inline uchar3 operator/(uchar f, const uchar3 a)
{
	return make_uchar3(f/a.x, f/a.y, f/a.z);
}

__device_inline uchar3 operator/(const uchar3 a, uchar f)
{
	return make_uchar3(a.x/f, a.y/f, a.z/f);
}

__device_inline uchar3 operator/(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x/b.x, a.y/b.y, a.z/b.z);
}

__device_inline uchar3 operator+(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device_inline uchar3 operator-(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device_inline uchar3 operator>>(uchar3& a, uchar f)
{
	return make_uchar3(a.x >> f, a.y >> f, a.z >> f);
}

__device_inline uchar3 operator<<(uchar3& a, uchar f)
{
	return make_uchar3(a.x << f, a.y << f, a.z << f);
}

__device_inline uchar3& operator>>=(uchar3& a, uchar f)
{
	return a = a >> f;
}

__device_inline uchar3& operator<<=(uchar3& a, uchar f)
{
	return a = a << f;
}

__device_inline uchar3& operator+=(uchar3& a, const uchar3 b)
{
	return a = a + b;
}

__device_inline uchar3& operator+=(uchar3& a, const uchar b)
{
	return a = a + make_uchar3(b);
}

__device_inline uchar3& operator-=(uchar3& a, const uchar3 b)
{
	return a = a - b;
}

__device_inline uchar3& operator-=(uchar3& a, const uchar b)
{
	return a = a - make_uchar3(b);
}

__device_inline uchar3& operator*=(uchar3& a, const uchar3 b)
{
	return a = a * b;
}

__device_inline uchar3& operator*=(uchar3& a, uchar f)
{
	return a = a * f;
}

__device_inline uchar3& operator/=(uchar3& a, const uchar3 b)
{
	return a = a / b;
}

__device_inline uchar3& operator/=(uchar3& a, uchar f)
{
	return a = a / f;
}

__device_inline uchar dot(const uchar3 a, const uchar3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device_inline uchar3 cross(const uchar3 a, const uchar3 b)
{
	return make_uchar3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

#endif

#ifndef __KERNEL_OPENCL__

/* uchar4 Vector */

__device_inline uchar4 min(const uchar4 a, const uchar4 b)
{
	return make_uchar4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

__device_inline uchar4 max(const uchar4 a, const uchar4 b)
{
	return make_uchar4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

__device_inline uchar4 clamp(const uchar4 a, const uchar4 mn, const uchar4 mx)
{
	return min(max(a, mn), mx);
}

__device_inline uchar4 operator-(const uchar4 a)
{
	return make_uchar4(-a.x, -a.y, -a.z, -a.w);
}

__device_inline uchar4 operator*(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

__device_inline uchar4 operator*(const uchar4 a, uchar f)
{
	return make_uchar4(a.x*f, a.y*f, a.z*f, a.w*f);
}

__device_inline uchar4 operator*(uchar f, const uchar4 a)
{
	return make_uchar4(a.x*f, a.y*f, a.z*f, a.w*f);
}

__device_inline uchar4 operator/(uchar f, const uchar4 a)
{
	return make_uchar4(f/a.x, f/a.y, f/a.z, f/a.w);
}

__device_inline uchar4 operator/(const uchar4 a, uchar f)
{
	return make_uchar4(a.x/f, a.y/f, a.z/f, a.w/f);
}

__device_inline uchar4 operator/(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

__device_inline uchar4 operator+(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device_inline uchar4 operator-(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

__device_inline uchar4 operator>>(uchar4& a, uchar f)
{
	return make_uchar4(a.x >> f, a.y >> f, a.z >> f, a.w >> f);
}

__device_inline uchar4 operator<<(uchar4& a, uchar f)
{
	return make_uchar4(a.x << f, a.y << f, a.z << f, a.w << f);
}

__device_inline uchar4& operator>>=(uchar4& a, uchar f)
{
	return a = a >> f;
}

__device_inline uchar4& operator<<=(uchar4& a, uchar f)
{
	return a = a << f;
}

__device_inline uchar4& operator+=(uchar4& a, const uchar4 b)
{
	return a = a + b;
}

__device_inline uchar4& operator+=(uchar4& a, const uchar b)
{
	return a = a + make_uchar4(b);
}

__device_inline uchar4& operator-=(uchar4& a, const uchar4 b)
{
	return a = a - b;
}

__device_inline uchar4& operator-=(uchar4& a, const uchar b)
{
	return a = a - make_uchar4(b);
}

__device_inline uchar4& operator*=(uchar4& a, const uchar4 b)
{
	return a = a * b;
}

__device_inline uchar4& operator*=(uchar4& a, uchar f)
{
	return a = a * f;
}

__device_inline uchar4& operator/=(uchar4& a, const uchar4 b)
{
	return a = a / b;
}

__device_inline uchar4& operator/=(uchar4& a, uchar f)
{
	return a = a / f;
}

__device_inline uchar dot(const uchar4 a, const uchar4 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__device_inline uchar4 cross(const uchar4 a, const uchar4 b)
{
	return make_uchar4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}

#endif

#ifndef __KERNEL_OPENCL__

/* uint2 vector */

__device_inline uint2 min(const uint2 a, const uint2 b)
{
	return make_uint2(min(a.x, b.x), min(a.y, b.y));
}

__device_inline uint2 max(const uint2 a, const uint2 b)
{
	return make_uint2(max(a.x, b.x), max(a.y, b.y));
}

__device_inline uint2 clamp(const uint2 a, const uint2 mn, const uint2 mx)
{
	return min(max(a, mn), mx);
}

__device_inline uint2 operator-(const uint2 a)
{
	return make_uint2(-a.x, -a.y);
}

__device_inline uint2 operator*(const uint2 a, const uint2 b)
{
	return make_uint2(a.x*b.x, a.y*b.y);
}

__device_inline uint2 operator*(const uint2 a, uint f)
{
	return make_uint2(a.x*f, a.y*f);
}

__device_inline uint2 operator*(uint f, const uint2 a)
{
	return make_uint2(a.x*f, a.y*f);
}

__device_inline uint2 operator/(uint f, const uint2 a)
{
	return make_uint2(f/a.x, f/a.y);
}

__device_inline uint2 operator/(const uint2 a, uint f)
{
	return make_uint2(a.x/f, a.y/f);
}

__device_inline uint2 operator/(const uint2 a, const uint2 b)
{
	return make_uint2(a.x/b.x, a.y/b.y);
}

__device_inline uint2 operator+(const uint2 a, const uint2 b)
{
	return make_uint2(a.x+b.x, a.y+b.y);
}

__device_inline uint2 operator-(const uint2 a, const uint2 b)
{
	return make_uint2(a.x-b.x, a.y-b.y);
}

__device_inline uint2 operator>>(uint2& a, uchar f)
{
	return make_uint2(a.x >> f, a.y >> f);
}

__device_inline uint2 operator<<(uint2& a, uchar f)
{
	return make_uint2(a.x << f, a.y << f);
}

__device_inline uint2& operator>>=(uint2& a, uchar f)
{
	return a = a >> f;
}

__device_inline uint2& operator<<=(uint2& a, uchar f)
{
	return a = a << f;
}

__device_inline uint2& operator+=(uint2& a, const uint2 b)
{
	return a = a + b;
}

__device_inline uint2& operator+=(uint2& a, const uint b)
{
	return a = a + make_uint2(b);
}

__device_inline uint2& operator-=(uint2& a, const uint2 b)
{
	return a = a - b;
}

__device_inline uint2& operator-=(uint2& a, const uint b)
{
	return a = a - make_uint2(b);
}

__device_inline uint2& operator*=(uint2& a, const uint2 b)
{
	return a = a * b;
}

__device_inline uint2& operator*=(uint2& a, uint f)
{
	return a = a * f;
}

__device_inline uint2& operator/=(uint2& a, const uint2 b)
{
	return a = a / b;
}

__device_inline uint2& operator/=(uint2& a, uint f)
{
	return a = a / f;
}

__device_inline uint dot(const uint2 a, const uint2 b)
{
	return a.x*b.x + a.y*b.y;
}

__device_inline uint cross(const uint2 a, const uint2 b)
{
	return a.x*b.y - a.y*b.x;
}

#endif

#ifndef __KERNEL_OPENCL__

/* uint3 Vector */

__device_inline uint3 min(const uint3 a, const uint3 b)
{
	return mask_select(a < b, a, b);
}

__device_inline uint3 max(const uint3 a, const uint3 b)
{
	return mask_select(a > b, a, b);
}

__device_inline uint3 clamp(uint3 a, uint3 mn, uint3 mx)
{
	return min(max(a, mn), mx);
}

__device_inline uint3 operator-(const uint3 a)
{
#ifdef __KERNEL_SSE__
	/* to negate, invert all bits and add one (subtract -1) */
	__m128i allones = _mm_cmpeq_epi32(a, a);
	__m128i t = _mm_xor_si128(a, allones);
	return _mm_sub_epi32(t, allones);
#else
	return make_uint3(-a.x, -a.y, -a.z);
#endif
}

__device_inline uint3 operator*(const uint3 a, const uint3 b)
{
	return make_uint3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device_inline uint3 operator*(const uint3 a, uint f)
{
	return make_uint3(a.x*f, a.y*f, a.z*f);
}

__device_inline uint3 operator*(uint f, const uint3 a)
{
#if defined __KERNEL_SSE__
	return sse_mul_32(_mm_set1_epi32(f), a);
#else
	return make_uint3(a.x*f, a.y*f, a.z*f);
#endif
}

__device_inline uint3 operator/(uint f, const uint3 a)
{
	return make_uint3(f/a.x, f/a.y, f/a.z);
}

__device_inline uint3 operator/(const uint3 a, uint f)
{
	return make_uint3(a.x/f, a.y/f, a.z/f);
}

__device_inline uint3 operator/(const uint3 a, const uint3 b)
{
	return make_uint3(a.x/b.x, a.y/b.y, a.z/b.z);
}

__device_inline uint3 operator+(const uint3 a, const uint3 b)
{
	return make_uint3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device_inline uint3 operator-(const uint3 a, const uint3 b)
{
	return make_uint3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device_inline uint3 operator>>(uint3& a, uchar f)
{
	return make_uint3(a.x >> f, a.y >> f, a.z >> f);
}

__device_inline uint3 operator<<(uint3& a, uchar f)
{
	return make_uint3(a.x << f, a.y << f, a.z << f);
}

__device_inline uint3& operator>>=(uint3& a, uchar f)
{
	return a = a >> f;
}

__device_inline uint3& operator<<=(uint3& a, uchar f)
{
	return a = a << f;
}

__device_inline uint3& operator+=(uint3& a, const uint3 b)
{
	return a = a + b;
}

__device_inline uint3& operator+=(uint3& a, const uint b)
{
	return a = a + make_uint3(b);
}

__device_inline uint3& operator-=(uint3& a, const uint3 b)
{
	return a = a - b;
}

__device_inline uint3& operator-=(uint3& a, const uint b)
{
	return a = a - make_uint3(b);
}

__device_inline uint3& operator*=(uint3& a, const uint3 b)
{
	return a = a * b;
}

__device_inline uint3& operator*=(uint3& a, uint f)
{
	return a = a * f;
}

__device_inline uint3& operator/=(uint3& a, const uint3 b)
{
	return a = a / b;
}

__device_inline uint3& operator/=(uint3& a, uint f)
{
	return a = a / f;
}

__device_inline int3 operator>>(const int3 a, int i)
{
#ifdef __KERNEL_SSE__
	return _mm_srai_epi32(a.m128, i);
#else
	return make_int3(a.x >> i, a.y >> i, a.z >> i);
#endif
}

__device_inline int3 operator<<(const int3 a, int i)
{
#ifdef __KERNEL_SSE__
	return _mm_slli_epi32(a.m128, i);
#else
	return make_int3(a.x << i, a.y << i, a.z << i);
#endif
}

__device_inline uint dot(const uint3 a, const uint3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device_inline uint3 cross(const uint3 a, const uint3 b)
{
	return make_uint3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

#endif

#ifndef __KERNEL_OPENCL__

/* uint4 Vector */
/* FIXME: SSE2 optimize */

__device_inline uint4 min(const uint4 a, const uint4 b)
{
	return mask_select(a < b, a, b);
}

__device_inline uint4 max(const uint4 a, const uint4 b)
{
	return mask_select(a > b, a, b);
}

__device_inline uint4 clamp(uint4 a, uint4 mn, uint4 mx)
{
	return min(max(a, mn), mx);
}

__device_inline uint4 operator-(const uint4 a)
{
#ifdef __KERNEL_SSE__
	/* to negate, invert all bits and add one (subtract -1) */
	__m128i allones = _mm_cmpeq_epi32(a, a);
	__m128i t = _mm_xor_si128(a, allones);
	return _mm_sub_epi32(t, allones);
#else
	return make_uint4(-a.x, -a.y, -a.z, -a.w);
#endif
}

__device_inline uint4 operator*(const uint4 a, const uint4 b)
{
#if defined __KERNEL_SSE__
	return sse_mul_32(a, b);
#else
	return make_uint4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
#endif
}

__device_inline uint4 operator*(const uint4 a, uint f)
{
#if defined __KERNEL_SSE__
	return sse_mul_32(a, _mm_set1_epi32(f));
#else
	return make_uint4(a.x*f, a.y*f, a.z*f, a.w*f);
#endif
}

__device_inline uint4 operator*(uint f, const uint4 a)
{
	return a * f;
}

__device_inline uint4 operator/(uint f, const uint4 a)
{
	return make_uint4(f/a.x, f/a.y, f/a.z, f/a.w);
}

__device_inline uint4 operator/(const uint4 a, uint f)
{
	return make_uint4(a.x/f, a.y/f, a.z/f, a.w/f);
}

__device_inline uint4 operator/(const uint4 a, const uint4 b)
{
	return make_uint4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

__device_inline uint4 operator+(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_add_epi32(a, b);
#else
	return make_uint4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
#endif
}

__device_inline uint4 operator-(const uint4 a, const uint4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_sub_epi32(a, b);
#else
	return make_uint4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
#endif
}

__device_inline uint4 operator>>(uint4 a, uchar f)
{
#ifdef __KERNEL_SSE__
	return _mm_srai_epi16(a, f);
#else
	return make_uint4(a.x >> f, a.y >> f, a.z >> f, a.w >> f);
#endif
}

__device_inline uint4 operator<<(uint4 a, uchar f)
{
#ifdef __KERNEL_SSE__
	return _mm_slli_epi32(a, f);
#else
	return make_uint4(a.x << f, a.y << f, a.z << f, a.w << f);
#endif
}

__device_inline uint4& operator>>=(uint4& a, uchar f)
{
#ifdef __KERNEL_SSE__
	return a = _mm_srai_epi32(a, f);
#else
	return a = a >> f;
#endif
}

__device_inline uint4& operator<<=(uint4& a, uchar f)
{
#ifdef __KERNEL_SSE__
	return a = _mm_slli_epi32(a, f);
#else
	return a = a << f;
#endif
}

__device_inline uint4& operator+=(uint4& a, const uint4 b)
{
	return a = a + b;
}

__device_inline uint4& operator+=(uint4& a, const uint b)
{
	return a = a + make_uint4(b);
}

__device_inline uint4& operator-=(uint4& a, const uint4 b)
{
	return a = a - b;
}

__device_inline uint4& operator-=(uint4& a, const uint b)
{
	return a = a - make_uint4(b);
}

__device_inline uint4& operator*=(uint4& a, const uint4 b)
{
	return a = a * b;
}

__device_inline uint4& operator*=(uint4& a, uint f)
{
	return a = a * f;
}

__device_inline uint4& operator/=(uint4& a, const uint4 b)
{
	return a = a / b;
}

__device_inline uint4& operator/=(uint4& a, uint f)
{
	return a = a / f;
}

__device_inline int4 operator>>(const int4 a, int i)
{
#ifdef __KERNEL_SSE__
	return _mm_srai_epi32(a.m128, i);
#else
	return make_int4(a.x >> i, a.y >> i, a.z >> i, a.w >> i);
#endif
}

__device_inline int4 operator<<(const int4 a, int i)
{
#ifdef __KERNEL_SSE__
	return _mm_slli_epi32(a.m128, i);
#else
	return make_int4(a.x << i, a.y << i, a.z << i, a.w << i);
#endif
}

__device_inline uint dot(const uint4 a, const uint4 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__device_inline uint4 cross(const uint4 a, const uint4 b)
{
	return make_uint4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}

#endif

#ifndef __KERNEL_OPENCL__

/* int2 vector */

__device_inline int2 min(const int2 a, const int2 b)
{
	return make_int2(min(a.x, b.x), min(a.y, b.y));
}

__device_inline int2 max(const int2 a, const int2 b)
{
	return make_int2(max(a.x, b.x), max(a.y, b.y));
}

__device_inline int2 clamp(int2 a, int2 mn, int2 mx)
{
	return min(max(a, mn), mx);
}

__device_inline int2 operator-(const int2 a)
{
	return make_int2(-a.x, -a.y);
}

__device_inline int2 operator*(const int2 a, const int2 b)
{
	return make_int2(a.x*b.x, a.y*b.y);
}

__device_inline int2 operator*(const int2 a, int f)
{
	return make_int2(a.x*f, a.y*f);
}

__device_inline int2 operator*(int f, const int2 a)
{
	return make_int2(a.x*f, a.y*f);
}

__device_inline int2 operator/(int f, const int2 a)
{
	return make_int2(f/a.x, f/a.y);
}

__device_inline int2 operator/(const int2 a, int f)
{
	return make_int2(a.x/f, a.y/f);
}

__device_inline int2 operator/(const int2 a, const int2 b)
{
	return make_int2(a.x/b.x, a.y/b.y);
}

__device_inline int2 operator+(const int2 a, const int2 b)
{
	return make_int2(a.x+b.x, a.y+b.y);
}

__device_inline int2 operator-(const int2 a, const int2 b)
{
	return make_int2(a.x-b.x, a.y-b.y);
}

__device_inline int2 operator>>(int2& a, uchar f)
{
	return make_int2(a.x >> f, a.y >> f);
}

__device_inline int2 operator<<(int2& a, uchar f)
{
	return make_int2(a.x << f, a.y << f);
}

__device_inline int2 operator>>=(int2& a, uchar f)
{
	return a = a >> f;
}

__device_inline int2 operator<<=(int2& a, uchar f)
{
	return a = a << f;
}

__device_inline int2 operator+=(int2& a, const int2 b)
{
	return a = a + b;
}

__device_inline int2 operator+=(int2& a, const int b)
{
	return a = a + make_int2(b);
}

__device_inline int2 operator-=(int2& a, const int2 b)
{
	return a = a - b;
}

__device_inline int2 operator-=(int2& a, const int b)
{
	return a = a - make_int2(b);
}

__device_inline int2 operator*=(int2& a, const int2 b)
{
	return a = a * b;
}

__device_inline int2 operator*=(int2& a, int f)
{
	return a = a * f;
}

__device_inline int2 operator/=(int2& a, const int2 b)
{
	return a = a / b;
}

__device_inline int2 operator/=(int2& a, int f)
{
	return a = a / f;
}

__device_inline int dot(const int2 a, const int2 b)
{
	return a.x*b.x + a.y*b.y;
}

__device_inline int cross(const int2 a, const int2 b)
{
	return a.x*b.y - a.y*b.x;
}

#endif

#ifndef __KERNEL_OPENCL__

/* int3 Vector */
/* FIXME: SSE2 optimize */

__device_inline int3 min(const int3 a, const int3 b)
{
	return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

__device_inline int3 max(const int3 a, const int3 b)
{
	return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

__device_inline int3 clamp(int3 a, int3 mn, int3 mx)
{
	return min(max(a, mn), mx);
}

__device_inline int3 operator-(const int3 a)
{
#ifdef __KERNEL_SSE__
	/* to negate, invert all bits and add one (subtract -1) */
	__m128i allones = _mm_cmpeq_epi32(a, a);
	__m128i t = _mm_xor_si128(a, allones);
	return _mm_sub_epi32(t, allones);
#else
	return make_int3(-a.x, -a.y, -a.z);
#endif
}

__device_inline int3 operator*(const int3 a, const int3 b)
{

	return make_int3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device_inline int3 operator*(const int3 a, int f)
{
	return make_int3(a.x*f, a.y*f, a.z*f);
}

__device_inline int3 operator*(int f, const int3 a)
{
#if defined __KERNEL_SSE__
	return sse_mul_32(_mm_set1_epi32(f), a);
#else
	return make_int3(a.x*f, a.y*f, a.z*f);
#endif
}

__device_inline int3 operator/(int f, const int3 a)
{
	return make_int3(f/a.x, f/a.y, f/a.z);
}

__device_inline int3 operator/(const int3 a, int f)
{
	return make_int3(a.x/f, a.y/f, a.z/f);
}

__device_inline int3 operator/(const int3 a, const int3 b)
{
	return make_int3(a.x/b.x, a.y/b.y, a.z/b.z);
}

__device_inline int3 operator+(const int3 a, const int3 b)
{
	return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device_inline int3 operator-(const int3 a, const int3 b)
{
	return make_int3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device_inline int3 operator>>(int3& a, uchar f)
{
	return make_int3(a.x >> f, a.y >> f, a.z >> f);
}

__device_inline int3 operator<<(int3& a, uchar f)
{
	return make_int3(a.x << f, a.y << f, a.z << f);
}

__device_inline int3 operator>>=(int3& a, uchar f)
{
	return a = a >> f;
}

__device_inline int3 operator<<=(int3& a, uchar f)
{
	return a = a << f;
}

__device_inline int3 operator+=(int3& a, const int3 b)
{
	return a = a + b;
}

__device_inline int3 operator+=(int3& a, const int b)
{
	return a = a + make_int3(b);
}

__device_inline int3 operator-=(int3& a, const int3 b)
{
	return a = a - b;
}

__device_inline int3 operator-=(int3& a, const int b)
{
	return a = a - make_int3(b);
}

__device_inline int3 operator*=(int3& a, const int3 b)
{
	return a = a * b;
}

__device_inline int3 operator*=(int3& a, int f)
{
	return a = a * f;
}

__device_inline int3 operator/=(int3& a, const int3 b)
{
	return a = a / b;
}

__device_inline int3 operator/=(int3& a, int f)
{
	return a = a / f;
}

__device_inline int dot(const int3 a, const int3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device_inline int3 cross(const int3 a, const int3 b)
{
	return make_int3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

#endif

#ifndef __KERNEL_OPENCL__

/* int4 Vector */
/* FIXME: SSE2 optimize */

__device_inline int4 min(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE4__
	return _mm_min_epi32(a.m128, b.m128);
#elif defined __KERNEL_SSE__
	return mask_select(a < b, a, b);
#else
	return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
#endif
}

__device_inline int4 max(int4 a, int4 b)
{
#ifdef __KERNEL_SSE4__
	return _mm_max_epi32(a.m128, b.m128);
#elif defined __KERNEL_SSE__
	return mask_select(a > b, a, b);
#else
	return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
#endif
}

__device_inline int4 clamp(const int4 a, const int4 mn, const int4 mx)
{
	return min(max(a, mn), mx);
}

__device_inline int4 operator-(const int4 a)
{
#ifdef __KERNEL_SSE__
	/* to negate, invert all bits and add one (subtract -1) */
	__m128i allones = _mm_cmpeq_epi32(a, a);
	__m128i t = _mm_xor_si128(a, allones);
	return _mm_sub_epi32(t, allones);
#else
	return make_int4(-a.x, -a.y, -a.z, -a.w);
#endif
}

__device_inline int4 operator*(const int4 a, const int4 b)
{
#if defined __KERNEL_SSE__
	return sse_mul_32(a, b);
#else
	return make_int4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
#endif
}

__device_inline int4 operator*(const int4 a, int f)
{
	return make_int4(a.x*f, a.y*f, a.z*f, a.w*f);
}

__device_inline int4 operator*(int f, const int4 a)
{
	return make_int4(a.x*f, a.y*f, a.z*f, a.w*f);
}

__device_inline int4 operator/(int f, const int4 a)
{
	return make_int4(f/a.x, f/a.y, f/a.z, f/a.w);
}

__device_inline int4 operator/(const int4 a, int f)
{
	return make_int4(a.x/f, a.y/f, a.z/f, a.w/f);
}

__device_inline int4 operator/(const int4 a, const int4 b)
{
	return make_int4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

__device_inline int4 operator+(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_add_epi32(a, b);
#else
	return make_int4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
#endif
}

__device_inline int4 operator-(const int4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_sub_epi32(a, b);
#else
	return make_int4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
#endif
}

__device_inline int4 operator>>(const int4 a, const uchar f)
{
#ifdef __KERNEL_SSE__
	return _mm_srai_epi32(a, f);
#else
	return make_int4(a.x >> f, a.y >> f, a.z >> f, a.w >> f);
#endif
}

__device_inline int4 operator<<(const int4 a, const uchar f)
{
#ifdef __KERNEL_SSE__
	return _mm_slli_epi32(a, f);
#else
	return make_int4(a.x << f, a.y << f, a.z << f, a.w << f);
#endif
}

__device_inline int4& operator>>=(int4& a, uchar f)
{
	return a = a >> f;
}

__device_inline int4& operator<<=(int4& a, uchar f)
{
	return a = a << f;
}

__device_inline int4& operator+=(int4& a, const int4 b)
{
	return a = a + b;
}

__device_inline int4& operator+=(int4& a, const int b)
{
	return a = a + make_int4(b);
}

__device_inline int4& operator-=(int4& a, const int4 b)
{
	return a = a - b;
}

__device_inline int4& operator-=(int4& a, const int b)
{
	return a = a - make_int4(b);
}

__device_inline int4& operator*=(int4& a, const int4 b)
{
	return a = a * b;
}

__device_inline int4& operator*=(int4& a, int f)
{
	return a = a * f;
}

__device_inline int4& operator/=(int4& a, const int4 b)
{
	return a = a / b;
}

__device_inline int4& operator/=(int4& a, int f)
{
	return a = a / f;
}

__device_inline int dot(const int4 a, const int4 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__device_inline int4 cross(const int4 a, const int4 b)
{
	return make_int4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}

#endif

#ifndef __KERNEL_OPENCL__

/* float2 vector */

__device_inline float2 rcp(const float2 a)
{
	return make_float2(1.0f / a.x, 1.0f / a.y);
}

__device_inline float2 min(const float2 a, const float2 b)
{
	return make_float2(min(a.x, b.x), min(a.y, b.y));
}

__device_inline float2 max(const float2 a, const float2 b)
{
	return make_float2(max(a.x, b.x), max(a.y, b.y));
}

__device_inline float2 clamp(const float2 a, const float2 mn, const float2 mx)
{
	return min(max(a, mn), mx);
}

__device_inline float2 operator-(const float2 a)
{
	return make_float2(-a.x, -a.y);
}

__device_inline float2 operator*(const float2 a, const float2 b)
{
	return make_float2(a.x*b.x, a.y*b.y);
}

__device_inline float2 operator*(const float2 a, float f)
{
	return make_float2(a.x*f, a.y*f);
}

__device_inline float2 operator*(float f, const float2 a)
{
	return make_float2(a.x*f, a.y*f);
}

__device_inline float2 operator/(float f, const float2 a)
{
	return make_float2(f/a.x, f/a.y);
}

__device_inline float2 operator/(const float2 a, float f)
{
	float invf = 1.0f/f;
	return make_float2(a.x*invf, a.y*invf);
}

__device_inline float2 operator/(const float2 a, const float2 b)
{
	return make_float2(a.x/b.x, a.y/b.y);
}

__device_inline float2 operator+(const float2 a, const float2 b)
{
	return make_float2(a.x+b.x, a.y+b.y);
}

__device_inline float2 operator-(const float2 a, const float2 b)
{
	return make_float2(a.x-b.x, a.y-b.y);
}

__device_inline float2 operator+=(float2& a, const float2 b)
{
	return a = a + b;
}

__device_inline float2 operator+=(float2& a, const float b)
{
	return a = a + make_float2(b);
}

__device_inline float2 operator-=(float2& a, const float2 b)
{
	return a = a - b;
}

__device_inline float2 operator-=(float2& a, const float b)
{
	return a = a - make_float2(b);
}

__device_inline float2 operator*=(float2& a, const float2 b)
{
	return a = a * b;
}

__device_inline float2 operator*=(float2& a, float f)
{
	return a = a * f;
}

__device_inline float2 operator/=(float2& a, const float2 b)
{
	return a = a / b;
}

__device_inline float2 operator/=(float2& a, float f)
{
	float invf = 1.0f/f;
	return a = a * invf;
}


__device_inline float dot(const float2 a, const float2 b)
{
	return a.x*b.x + a.y*b.y;
}

__device_inline float cross(const float2 a, const float2 b)
{
	return (a.x*b.y - a.y*b.x);
}

#endif

#ifndef __KERNEL_OPENCL__

__device_inline float len(const float2 a)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(dot(a,a))));
#else
	return sqrtf(dot(a, a));
#endif
}

__device_inline float len_rcp(const float2 a)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(dot(a,a))));
#else
	return 1.0f/sqrtf(dot(a, a));
#endif
}

__device_inline float2 normalize(const float2 a)
{
	return a*len_rcp(a);
}

__device_inline float2 normalize_len(const float2 a, float *t)
{
	*t = len(a);
	return a/(*t);
}

__device_inline float2 fabs(float2 a)
{
	return make_float2(fabsf(a.x), fabsf(a.y));
}

__device_inline float2 as_float2(const float4 a)
{
#ifdef __KERNEL_SSE__
	float t0 = _mm_cvtss_f32(a);
	float t1 = _mm_cvtss_f32(_mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 1)));
	return make_float2(t0, t1);
#else
	return make_float2(a.x, a.y);
#endif
}

/* return a with signs copied from b */
__device_inline float3 copysignf(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
    __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	__m128 abs_a = _mm_andnot_ps(signmask, a);
	__m128 signs = _mm_and_ps(signmask, b);
	return _mm_or_ps(abs_a, signs);
#else
	return float3(
		copysign(a.x, b.x),
		copysign(a.y, b.y),
		copysign(a.z, b.z));
#endif
}

/* return a with signs copied from b */
__device_inline float4 copysignf(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
    __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	__m128 abs_a = _mm_andnot_ps(signmask, a);
	__m128 signs = _mm_and_ps(signmask, b);
	return _mm_or_ps(abs_a, signs);
#else
    float4 r;
    r.x = copysign(a.x, b.x);
    r.y = copysign(a.y, b.y);
    r.z = copysign(a.z, b.z);
    r.w = copysign(a.w, b.w);
    return r;
#endif
}

#endif

#ifndef __KERNEL_GPU__

__device_inline void print_float2(const char *label, const float2 a)
{
	printf("%s: %.8f %.8f\n", label, (double)a.x, (double)a.y);
}

#endif

#ifndef __KERNEL_OPENCL__

__device_inline float2 interp(float2 a, float2 b, float t)
{
	return a + t*(b - a);
}

#endif

/* float3 Vector */

#ifndef __KERNEL_OPENCL__

__device_inline float3 rcp(const float3 a)
{
#if defined __KERNEL_SSE4__
	/* preserve a.w */
	__m128 r = _mm_rcp_ps(a);

	// extra precision
	//r = _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), t));

	return r;

	//return _mm_blend_ps(r, w, 1 << 3);
#elif defined __KERNEL_SSE__
	/* get 1.0 into high part */
	__m128 highmask = _mm_castsi128_ps(_mm_cvtsi32_si128(0xFFFFFFFF));
	highmask = _mm_shuffle_ps(highmask, highmask, _MM_SHUFFLE(0, 1, 1, 1));
	__m128 highone = _mm_set_ss(1.0f);
	highone = _mm_shuffle_ps(highone, highone, _MM_SHUFFLE(0, 1, 1, 1));

	__m128 t = _mm_andnot_ps(highmask, a);
	t = _mm_or_ps(t, highone);

	__m128 r = _mm_rcp_ps(t);

	// extra precision
	//r = _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), t));

	return r;
#else
	return make_float3(1.0f/a.x, 1.0f/a.y, 1.0f/a.z);
#endif
}

__device_inline float3 min(float3 a, float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_min_ps(a.m128, b.m128);
#else
	return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
#endif
}

__device_inline float3 max(float3 a, float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_max_ps(a.m128, b.m128);
#else
	return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
#endif
}

__device_inline float3 clamp(float3 a, float3 mn, float3 mx)
{
	return min(max(a, mn), mx);
}

__device_inline float3 operator-(const float3 a)
{
#ifdef __KERNEL_SSE__
	__m128 signbits = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	return _mm_xor_ps(a, signbits);
#else
	return make_float3(-a.x, -a.y, -a.z);
#endif
}

__device_inline float3 operator*(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_mul_ps(a, b);
#else
	return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
#endif
}

__device_inline float3 operator*(const float3 a, float f)
{
#ifdef __KERNEL_SSE__
	return _mm_mul_ps(a, _mm_set1_ps(f));
#else
	return make_float3(a.x*f, a.y*f, a.z*f);
#endif
}

__device_inline float3 operator*(float f, const float3 a)
{
#ifdef __KERNEL_SSE__
	return _mm_mul_ps(_mm_set1_ps(f), a);
#else
	return make_float3(a.x*f, a.y*f, a.z*f);
#endif
}

__device_inline float3 operator/(float f, const float3 a)
{
#ifdef __KERNEL_SSE__
	return f * rcp(a);
#else
	return make_float3(f/a.x, f/a.y, f/a.z);
#endif
}

__device_inline float3 operator/(const float3 a, float f)
{
#ifdef __KERNEL_SSE__
	float3 oof = fast_rcp(make_float3(f));
	return a * oof;
#else
	float invf = 1.0f/f;
	return make_float3(a.x*invf, a.y*invf, a.z*invf);
#endif
}

__device_inline float3 operator/(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return a * rcp(b);
#else
	return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
#endif
}

__device_inline float3 operator+(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_add_ps(a, b);
#else
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
#endif
}

__device_inline float3 operator-(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	return _mm_sub_ps(a, b);
#else
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
#endif
}

__device_inline float3 operator+=(float3& a, const float3 b)
{
#ifdef __KERNEL_SSE__
	a = _mm_add_ps(a, b);
	return a;
#else
	a = a + b;
	return a;
#endif
}

__device_inline float3 operator+=(float3& a, const float b)
{
#ifdef __KERNEL_SSE__
	a = _mm_add_ps(a, _mm_set1_ps(b));
	return a;
#else
	a = a + make_float3(b);
	return a;
#endif
}

__device_inline float3 operator-=(float3& a, const float3 b)
{
#ifdef __KERNEL_SSE__
	a = _mm_sub_ps(a, b);
	return a;
#else
	a = a - b;
	return a;
#endif
}

__device_inline float3 operator-=(float3& a, const float b)
{
#ifdef __KERNEL_SSE__
	a = _mm_sub_ps(a, _mm_set1_ps(b));
	return a;
#else
	a = a - make_float3(b);
	return a;
#endif
}

__device_inline float3 operator*=(float3& a, const float3 b)
{
#ifdef __KERNEL_SSE__
	a.m128 = _mm_mul_ps(a, b);
	return a;
#else
	a = a * b;
	return a;
#endif
}

__device_inline float3 operator*=(float3& a, float f)
{
#ifdef __KERNEL_SSE__
	a.m128 = _mm_mul_ps(a, _mm_set1_ps(f));
	return a;
#else
	return a = a * f;
#endif
}

__device_inline float3 operator/=(float3& a, const float3 b)
{
#ifdef __KERNEL_SSE__
	a.m128 = _mm_mul_ps(a, fast_rcp(b));
	return a;
#else
	return a = a / b;
#endif
}

__device_inline float3 operator/=(float3& a, float f)
{
#ifdef __KERNEL_SSE__
	a.m128 = _mm_mul_ps(a, _mm_rcp_ps(_mm_set1_ps(f)));
	return a;
#else
	return a = a / f;
#endif
}

__device_inline float dot(const float3 a, const float3 b)
{
#if defined __KERNEL_SSE4__
	/* 0x71 means do xyz dot product
	 * and put result only in low component, zero the rest */
	return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x71));
#elif defined __KERNEL_SSE3__
	/* zero out w components */
	__m128 mask = _mm_castsi128_ps(_mm_cvtsi32_si128(0xFFFFFFFF));
	mask = _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(0, 1, 1, 1));
	__m128 ta = _mm_andnot_ps(mask, a);
	__m128 tb = _mm_andnot_ps(mask, b);

	ta = _mm_mul_ps(ta, tb);	/* 0, a.z*b.z, a.y*b.y, a.x*b.x */
	ta = _mm_hadd_ps(ta, ta);	/* 0 + a.z*b.z, a.y*b.y + a.x*b.x (in both halves) */
	ta = _mm_hadd_ps(ta, ta);	/* 0 + a.z*b.z + a.y*b.y + a.x*b.x (in all four) */
	return _mm_cvtss_f32(ta);
#elif defined __KERNEL_SSE__
	__m128 t = _mm_mul_ps(a, b);
	__m128 ay = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 1));
	t = _mm_add_ss(t, ay);
	__m128 az = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 2));
	t = _mm_add_ss(t, az);
	return _mm_cvtss_f32(t);
#else
	return a.x*b.x + a.y*b.y + a.z*b.z;
#endif
}

__device_inline float3 cross(const float3 a, const float3 b)
{
#ifdef __KERNEL_SSE__
	// r.x = a.y * b.z - a.z * b.y
	// r.y = a.z * b.x - a.x * b.z
	// r.z = a.x * b.y - a.y * b.x
	//        A     B     C     D
	__m128 ta = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
	__m128 tb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
	__m128 tc = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
	__m128 td = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
	ta = _mm_mul_ps(ta, tb);
	tc = _mm_mul_ps(tc, td);
	ta = _mm_sub_ps(ta, tc);
	return float3(ta);
#else
	float3 r = make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	return r;
#endif
}

#endif

__device_inline float len(float3 a)
{
#ifdef __KERNEL_SSE__
	a.m128 = _mm_mul_ps(a, a);
	__m128 ay = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 1));
	__m128 az = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 2));
	a.m128 = _mm_add_ss(a, ay);
	a.m128 = _mm_add_ss(a, az);
	a.m128 = _mm_sqrt_ss(a);
	return _mm_cvtss_f32(a);
#else
	return sqrtf(dot(a, a));
#endif
}

__device_inline float len_squared(const float3 a)
{
	return dot(a, a);
}

#ifndef __KERNEL_OPENCL__

__device_inline float3 normalize(const float3 a)
{
	return a/len(a);
}

#endif

/* return a normalized copy of a, and store the original length to *t */
__device_inline float3 normalize_len(const float3 a, float *t)
{
	*t = len(a);
	return a/(*t);
}

#ifndef __KERNEL_OPENCL__

__device_inline float3 fabs(float3 a)
{
#ifdef __KERNEL_SSE__
	__m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	return _mm_and_ps(a.m128, mask);
#else
	return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
#endif
}

#endif

__device_inline float3 float2_to_float3(const float2 a)
{
	return make_float3(a.x, a.y, 0.0f);
}

__device_inline float3 float4_to_float3(const float4 a)
{
#ifdef __KERNEL_SSE__
	return a.m128;
#else
	return make_float3(a.x, a.y, a.z);
#endif
}

__device_inline float4 float3_to_float4(const float3 a)
{
#if defined __KERNEL_SSE4__
	return _mm_insert_ps(a, _mm_set_ss(1.0f), 3 << 4);
#elif defined __KERNEL_SSE__
	__m128 mask = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0, 0, 0));
	__m128 high1 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
	return _mm_or_ps(high1, _mm_andnot_ps(mask, a));
#else
	return make_float4(a.x, a.y, a.z, 1.0f);
#endif
}

#ifndef __KERNEL_GPU__

__device_inline void print_float3(const char *label, const float3 a)
{
	printf("%s: %.8f %.8f %.8f\n", label, (double)a.x, (double)a.y, (double)a.z);
}

__device_inline float rcp(float a)
{
#ifdef __KERNEL_SSE__
	__m128 ta = _mm_set_ss(a);
	float3 r = _mm_rcp_ss(ta);
	return _mm_cvtss_f32(r);
	//return _mm_cvtss_f32(_mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), ta)));
#else
	return 1.0f/a;
#endif
}

#endif

__device_inline float3 interp(const float3 a, const float3 b, float t)
{
	return a + t*(b - a);
}

__device_inline bool is_zero(const float3 a)
{
#ifdef __KERNEL_SSE__
	return (_mm_movemask_ps(_mm_cmpeq_ps(a, _mm_setzero_ps())) & 0x7) == 0x7;
#else
	return (a.x == 0.0f && a.y == 0.0f && a.z == 0.0f);
#endif
}

__device_inline float reduce_add(const float3 a)
{
#if defined __KERNEL_SSE4__
	return _mm_cvtss_f32(_mm_dp_ps(a, _mm_set1_ps(1.0f), 0x71));
#elif defined __KERNEL_SSE__
    __m128 tx;
    __m128 ty = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 tz = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 2));
	tx = _mm_add_ss(a, ty);
	tx = _mm_add_ss(tx, tz);
	return _mm_cvtss_f32(tx);
#else
	return (a.x + a.y + a.z);
#endif
}

__device_inline float average(const float3 a)
{
	return reduce_add(a)*(1.0f/3.0f);
}

#ifndef __KERNEL_OPENCL__
/* shuffle */
/* FIXME: SSE optimize */

template<size_t index_0, size_t index_1>
__forceinline uchar2 shuffle(const uchar2 b)
{
	return make_uchar2(b[index_0], b[index_1]);
}

template<size_t index_0, size_t index_1, size_t index_2>
__forceinline uchar3 shuffle(const uchar3 b)
{
	return make_uchar3(b[index_0], b[index_1], b[index_2]);
}

template<size_t index_0, size_t index_1, size_t index_2, size_t index_3>
__forceinline uchar4 shuffle(const uchar4 b)
{
	return make_uchar4(b[index_0], b[index_1], b[index_2], b[index_3]);
}

template<size_t index_0, size_t index_1>
__forceinline uint2 shuffle(const uint2 b)
{
	return make_uint2(b[index_0], b[index_1]);
}

template<size_t index_0, size_t index_1, size_t index_2>
__forceinline uint3 shuffle(const uint3 b)
{
	return make_uint3(b[index_0], b[index_1], b[index_2]);
}

template<size_t index_0, size_t index_1, size_t index_2, size_t index_3>
__forceinline uint4 shuffle(const uint4 b)
{
	return make_uint4(b[index_0], b[index_1], b[index_2], b[index_3]);
}

template<size_t index_0, size_t index_1>
__forceinline int2 shuffle(const int2 b)
{
	return make_int2(b[index_0], b[index_1]);
}

template<size_t index_0, size_t index_1, size_t index_2>
__forceinline int3 shuffle(const int3 b)
{
	return make_int3(b[index_0], b[index_1], b[index_2]);
}

template<size_t index_0, size_t index_1, size_t index_2, size_t index_3>
__forceinline int4 shuffle(const int4 b)
{
	return make_int4(b[index_0], b[index_1], b[index_2], b[index_3]);
}

template<size_t index_0, size_t index_1>
__forceinline float2 shuffle(const float2 b)
{
	return make_float2(b[index_0], b[index_1]);
}

#ifndef __KERNEL_SSE__

template<size_t index_0, size_t index_1, size_t index_2>
__forceinline float3 shuffle(const float3 b)
{
	return make_float3(b[index_0], b[index_1], b[index_2]);
}

template<size_t index_0, size_t index_1, size_t index_2, size_t index_3>
__forceinline float4 shuffle(const float4 b)
{
	return make_float4(b[index_0], b[index_1], b[index_2], b[index_3]);
}

#else

template<size_t index_0, size_t index_1, size_t index_2>
__forceinline float3 shuffle(const float3 b)
{
	return _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, index_2, index_1, index_0));
}

template<size_t index_0, size_t index_1, size_t index_2, size_t index_3>
__forceinline float4 shuffle(const float4 b)
{
	return _mm_shuffle_ps(b, b, _MM_SHUFFLE(index_3, index_2, index_1, index_0));
}

#ifdef __KERNEL_SSE3__
template<>
__forceinline float4 shuffle<0, 0, 2, 2>(const float4 b)
{
	return _mm_moveldup_ps(b);
}

template<>
__forceinline float4 shuffle<1, 1, 3, 3>(const float4 b)
{
	return _mm_movehdup_ps(b);
}

template<>
__forceinline float4 shuffle<0, 1, 0, 1>(const float4 b)
{
	return _mm_castpd_ps(_mm_movedup_pd(_mm_castps_pd(b)));
}
#endif // __KERNEL_SSE3__

#endif // __KERNEL_SSE__

#endif // __KERNEL_OPENCL__

#ifndef __KERNEL_OPENCL__

/* float4 vector */

__device_inline float4 min(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_min_ps(a, b);
#else
	return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
#endif
}

__device_inline float4 max(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_max_ps(a, b);
#else
	return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
#endif
}

__device_inline float4 clamp(float4 a, float4 mn, float4 mx)
{
	return min(max(a, mn), mx);
}

__device_inline float4 operator-(const float4 a)
{
#ifdef __KERNEL_SSE__
	__m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	return _mm_xor_ps(a.m128, mask);
#else
	return make_float4(-a.x, -a.y, -a.z, -a.w);
#endif
}

__device_inline float4 operator*(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_mul_ps(a, b);
#else
	return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
#endif
}

__device_inline float4 operator*(const float4 a, float f)
{
#ifdef __KERNEL_SSE__
	return a * make_float4(f);
#else
	return make_float4(a.x*f, a.y*f, a.z*f, a.w*f);
#endif
}

__device_inline float4 operator*(float f, const float4 a)
{
	return a * f;
}

__device_inline float4 rcp(const float4 a)
{
#ifdef __KERNEL_SSE__
	float4 r = _mm_rcp_ps(a.m128);
	return r;
	//return _mm_sub_ps(_mm_add_ps(r, r), _mm_mul_ps(_mm_mul_ps(r, r), a));
#else
	return make_float4(1.0f/a.x, 1.0f/a.y, 1.0f/a.z, 1.0f/a.w);
#endif
}

__device_inline float4 operator/(const float4 a, float f)
{
	return a * rcp(f);
}

__device_inline float4 operator/(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return a * rcp(b);
#else
	return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
#endif

}

__device_inline float4 operator+(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_add_ps(a.m128, b.m128);
#else
	return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
#endif
}

__device_inline float4 operator-(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_sub_ps(a.m128, b.m128);
#else
	return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
#endif
}

__device_inline float4 operator^(const float4 a, const int4 b)
{
#ifdef __KERNEL_SSE__
	return _mm_xor_ps(a, _mm_castsi128_ps(b));
#else
	union { float f; int i; } t[4];
	t[0].f = a[0];
	t[1].f = a[1];
	t[2].f = a[2];
	t[3].f = a[3];
	t[0].i ^= b[0];
	t[1].i ^= b[1];
	t[2].i ^= b[2];
	t[3].i ^= b[3];
	return float4(t[0].f, t[1].f, t[2].f, t[3].f);
#endif
}

__device_inline float4 operator+=(float4& a, const float4 b)
{
	return a = a + b;
}

__device_inline float4 operator+=(float4& a, const float b)
{
	return a = a + make_float4(b);
}

__device_inline float4 operator-=(float4& a, const float4 b)
{
	return a = a - b;
}

__device_inline float4 operator-=(float4& a, const float b)
{
	return a = a - make_float4(b);
}

__device_inline float4 operator*=(float4& a, const float4 b)
{
	return a = a * b;
}

__device_inline float4 operator*=(float4& a, const float b)
{
	return a = a * make_float4(b);
}

__device_inline float4 operator/=(float4& a, const float4 b)
{
	return a = a * rcp(b);
}

__device_inline float4 operator/=(float4& a, float f)
{
	return a = a / f;
}

__device_inline float dot(const float4 a, const float4 b)
{
#if defined __KERNEL_SSE4__
	/* 0xF1 means do xyzw dot product
	 * and put result only in low component, zero the rest */
	return _mm_cvtss_f32(_mm_dp_ps(a, b, 0xF1));
#elif defined __KERNEL_SSE3__
	__m128 ta;
	ta = _mm_mul_ps(a, b);		/* a.w*b.w, a.z*b.z, a.y*b.y, a.x*b.x */
	ta = _mm_hadd_ps(ta, ta);	/* a.w*b.w + a.z*b.z, a.y*b.y + a.x*b.x (in both halves) */
	ta = _mm_hadd_ps(ta, ta);	/* a.w*b.w + a.z*b.z + a.y*b.y + a.x*b.x (in all four) */
	return _mm_cvtss_f32(ta);
#elif defined __KERNEL_SSE__
	__m128 t = _mm_mul_ps(a, b);
	__m128 ay = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 1));
	__m128 az = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 2));
	__m128 aw = _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 3));
	t = _mm_add_ss(t, ay);
	az = _mm_add_ss(az, aw);
	t = _mm_add_ss(t, az);
	return _mm_cvtss_f32(t);
#else
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
#endif
}

__device_inline float4 cross(const float4 a, const float4 b)
{
#ifdef __KERNEL_SSE__
	// r.x = a.y * b.z - a.z * b.y
	// r.y = a.z * b.x - a.x * b.z
	// r.z = a.x * b.y - a.y * b.x
	//        A     B     C     D
	__m128 ta = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 2, 1));
	__m128 tb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 0, 2));
	__m128 tc = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 0, 2));
	__m128 td = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 2, 1));
	ta = _mm_mul_ps(ta, tb);
	tc = _mm_mul_ps(tc, td);
	ta = _mm_sub_ps(ta, tc);
	return float4(ta);
#else
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.0f);
#endif
}

__device_inline bool is_zero(const float4 a)
{
#ifdef __KERNEL_SSE__
	return _mm_movemask_ps(_mm_cmpeq_ps(a, _mm_setzero_ps())) == 0xF;
#else
	return (a.x == 0.0f && a.y == 0.0f && a.z == 0.0f && a.w == 0.0f);
#endif
}

__device_inline float reduce_add(const float4 a)
{
#ifdef __KERNEL_SSE__
	__m128 t0 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 3, 2));
	__m128 t1 = _mm_movehl_ps(a, a);
	t0 = _mm_add_ps(t0, t1);
	t1 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(3, 2, 1, 1));
	t0 = _mm_add_ps(t0, t1);
	t0 = _mm_mul_ss(t0, _mm_set_ss(0.25f));
	return _mm_cvtss_f32(t0);
#else
	return ((a.x + a.y) + (a.z + a.w));
#endif
}

__device_inline float average(const float4 a)
{
	return reduce_add(a) * 0.25f;
}

__device_inline float len(const float4 a)
{
	return sqrtf(dot(a, a));
}

__device_inline float4 normalize(const float4 a)
{
	return a/len(a);
}

#endif

#ifndef __KERNEL_GPU__

__device_inline float4 reduce_min(const float4 a)
{
#ifdef __KERNEL_SSE__
	float4 h = min(S_yxwz(a), a);
	return min(S_zwxy(h), h);
#else
	return make_float4(min(min(a.x, a.y), min(a.z, a.w)));
#endif
}

__device_inline float4 reduce_max(const float4 a)
{
#ifdef __KERNEL_SSE__
	float4 h = max(shuffle<1,0,3,2>(a), a);
	return max(shuffle<2,3,0,1>(h), h);
#else
	return make_float4(max(max(a.x, a.y), max(a.z, a.w)));
#endif
}

#if 0
__device_inline float4 reduce_add(const float4 a)
{
#ifdef __KERNEL_SSE__
	float4 h = shuffle<1,0,3,2>(a) + a;
	return shuffle<2,3,0,1>(h) + h;
#else
	return make_float4((a.x + a.y) + (a.z + a.w));
#endif
}
#endif

__device_inline void print_float4(const char *label, const float4 a)
{
	printf("%s: %.8f %.8f %.8f %.8f\n", label, (double)a.x, (double)a.y, (double)a.z, (double)a.w);
}

#endif

#ifndef __KERNEL_GPU__

__device_inline void print_int3(const char *label, const int3 a)
{
	printf("%s: %d %d %d\n", label, a.x, a.y, a.z);
}

#endif

#ifndef __KERNEL_GPU__

__device_inline void print_int4(const char *label, const int4 a)
{
	printf("%s: %d %d %d %d\n", label, a.x, a.y, a.z, a.w);
}

#endif

/* Int/Float conversion */

#ifndef __KERNEL_OPENCL__

__device_inline int as_int(uint i)
{
	return (int)i;
	//union { uint ui; int i; } u;
	//u.ui = i;
	//return u.i;
}

__device_inline uint as_uint(int i)
{
	return (uint)i;
	//union { uint ui; int i; } u;
	//u.i = i;
	//return u.ui;
}

__device_inline uint as_uint(float f)
{
	union { uint i; float f; } u;
	u.f = f;
	return u.i;
}

__device_inline int __float_as_int(float f)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtsi128_si32(_mm_castps_si128(_mm_set_ss(f)));
#else
	union { int i; float f; } u;
	u.f = f;
	return u.i;
#endif
}

__device_inline float __int_as_float(int i)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(i)));
#else
	union { int i; float f; } u;
	u.i = i;
	return u.f;
#endif
}

__device_inline uint __float_as_uint(float f)
{
	union { uint i; float f; } u;
	u.f = f;
	return u.i;
}

__device_inline float __uint_as_float(uint i)
{
#ifdef __KERNEL_SSE__
	return _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128((int)i)));
#else
	union { uint i; float f; } u;
	u.i = i;
	return u.f;
#endif
}

/* Interpolation */

template<class A, class B> A lerp(const A& a, const A& b, const B& t)
{
	return (A)(a * ((B)1 - t) + b * t);
}

/* Triangle */

__device_inline float triangle_area(const float3 v1, const float3 v2, const float3 v3)
{
	return len(cross(v3 - v2, v1 - v2))*0.5f;
}

#endif

/* Orthonormal vectors */

__device_inline void make_orthonormals(const float3 N, float3 *a, float3 *b)
{
	if(N.x != N.y || N.x != N.z)
		*a = make_float3(N.z-N.y, N.x-N.z, N.y-N.x);  //(1,1,1)x N
	else
		*a = make_float3(N.z-N.y, N.x+N.z, -N.y-N.x);  //(-1,1,1)x N

	*a = normalize(*a);
	*b = cross(N, *a);
}

/* Color division */

__device_inline float3 safe_divide_color(float3 a, float3 b)
{
	float3 inverseb = rcp(b);
	float3 zerof = make_float3(0);
	return mask_select(b != zerof, a * inverseb, zerof);
}

/* Rotation of point around axis and angle */

__device_inline float3 rotate_around_axis(const float3 p, const float3 axis, float angle)
{
	float costheta = cosf(angle);
	float sintheta = sinf(angle);
	float3 r;

	// todo: vectorize
	//r.x  = ((costheta + (1 - costheta) * axis.x * axis.x +   0    *     0   ) * p.x);
	//r.y  = ((   0     + (1 - costheta) * axis.x * axis.y + axis.z * sintheta) * p.x);
	//r.z  = ((   0     + (1 - costheta) * axis.x * axis.z - axis.y * sintheta) * p.x);

	//r.x += ((   0     + (1 - costheta) * axis.x * axis.y - axis.z * sintheta) * p.y);
	//r.y += ((costheta + (1 - costheta) * axis.y * axis.y +   0    *     0	  ) * p.y);
	//r.z += ((   0     + (1 - costheta) * axis.y * axis.z + axis.x * sintheta) * p.y);

	//r.x += ((   0     + (1 - costheta) * axis.x * axis.z + axis.y * sintheta) * p.z);
	//r.y += ((   0     + (1 - costheta) * axis.y * axis.z - axis.x * sintheta) * p.z);
	//r.z += ((costheta + (1 - costheta) * axis.z * axis.z +   0    *     0   ) * p.z);

	r.x = ((costheta + (1 - costheta) * axis.x * axis.x) * p.x) +
		(((1 - costheta) * axis.x * axis.y - axis.z * sintheta) * p.y) +
		(((1 - costheta) * axis.x * axis.z + axis.y * sintheta) * p.z);

	r.y = (((1 - costheta) * axis.x * axis.y + axis.z * sintheta) * p.x) +
		((costheta + (1 - costheta) * axis.y * axis.y) * p.y) +
		(((1 - costheta) * axis.y * axis.z - axis.x * sintheta) * p.z);

	r.z = (((1 - costheta) * axis.x * axis.z - axis.y * sintheta) * p.x) +
		(((1 - costheta) * axis.y * axis.z + axis.x * sintheta) * p.y) +
		((costheta + (1 - costheta) * axis.z * axis.z) * p.z);

	return r;
}

/* NaN-safe math ops */

__device float safe_asinf(float a)
{
	if(a <= -1.0f)
		return -M_PI_2_F;
	else if(a >= 1.0f)
		return M_PI_2_F;

	return asinf(a);
}

__device float safe_acosf(float a)
{
	if(a <= -1.0f)
		return M_PI_F;
	else if(a >= 1.0f)
		return 0.0f;

	return acosf(a);
}

__device float compatible_powf(float x, float y)
{
	/* GPU pow doesn't accept negative x, do manual checks here */
	if(x < 0.0f) {
		if(fmodf(-y, 2.0f) == 0.0f)
			return powf(-x, y);
		else
			return -powf(-x, y);
	}
	else if(x == 0.0f)
		return 0.0f;

	return powf(x, y);
}

__device float safe_powf(float a, float b)
{
	if(b == 0.0f)
		return 1.0f;
	if(a == 0.0f)
		return 0.0f;
	if(a < 0.0f && b != float_to_int(b))
		return 0.0f;
	
	return compatible_powf(a, b);
}

__device float safe_logf(float a, float b)
{
	if(a < 0.0f || b < 0.0f)
		return 0.0f;

	return logf(a)/logf(b);
}

__device float safe_divide(float a, float b)
{
	return (b != 0.0f)? a/b: 0.0f;
}

__device float safe_modulo(float a, float b)
{
	return (b != 0.0f)? fmodf(a, b): 0.0f;
}

/* Ray Intersection */

__device bool ray_sphere_intersect(
	float3 ray_P, float3 ray_D, float ray_t,
	float3 sphere_P, float sphere_radius,
	float3 *isect_P, float *isect_t)
{
	float3 d = sphere_P - ray_P;
	float radiussq = sphere_radius*sphere_radius;
	float tsq = dot(d, d);

	if(tsq > radiussq) { /* ray origin outside sphere */
		float tp = dot(d, ray_D);

		if(tp < 0.0f) /* dir points away from sphere */
			return false;

		float dsq = tsq - tp*tp; /* pythagoras */

		if(dsq > radiussq) /* closest point on ray outside sphere */
			return false;

		float t = tp - sqrtf(radiussq - dsq); /* pythagoras */

		if(t < ray_t) {
			*isect_t = t;
			*isect_P = ray_P + ray_D*t;
			return true;
		}
	}

	return false;
}

__device bool ray_aligned_disk_intersect(
	float3 ray_P, float3 ray_D, float ray_t,
	float3 disk_P, float disk_radius,
	float3 *isect_P, float *isect_t)
{
	/* aligned disk normal */
	float disk_t;
	float3 disk_N = normalize_len(ray_P - disk_P, &disk_t);
	float div = dot(ray_D, disk_N);

	if(div == 0.0f)
		return false;

	/* compute t to intersection point */
	float t = -disk_t/div;
	if(t < 0.0f || t > ray_t)
		return false;
	
	/* test if within radius */
	float3 P = ray_P + ray_D*t;
	if(len_squared(P - disk_P) > disk_radius*disk_radius)
		return false;

	*isect_P = P;
	*isect_t = t;

	return true;
}

__device bool ray_triangle_intersect(
	float3 ray_P, float3 ray_D, float ray_t,
	float3 v0, float3 v1, float3 v2,
	float3 *isect_P, float *isect_t)
{
	/* Calculate intersection */
	float3 e1 = v1 - v0;
	float3 e2 = v2 - v0;
	float3 s1 = cross(ray_D, e2);

	const float divisor = dot(s1, e1);
	if(divisor == 0.0f)
		return false;

	const float invdivisor = 1.0f/divisor;

	/* compute first barycentric coordinate */
	const float3 d = ray_P - v0;
	const float u = dot(d, s1)*invdivisor;
	if(u < 0.0f)
		return false;

	/* Compute second barycentric coordinate */
	const float3 s2 = cross(d, e1);
	const float v = dot(ray_D, s2)*invdivisor;
	if(v < 0.0f)
		return false;

	const float b0 = 1.0f - u - v;
	if(b0 < 0.0f)
		return false;

	/* compute t to intersection point */
	const float t = dot(e2, s2)*invdivisor;
	if(t < 0.0f || t > ray_t)
		return false;

	*isect_t = t;
	*isect_P = ray_P + ray_D*t;

	return true;
}

__device bool ray_quad_intersect(
	float3 ray_P, float3 ray_D, float ray_t,
	float3 quad_P, float3 quad_u, float3 quad_v,
	float3 *isect_P, float *isect_t)
{
	float3 v0 = quad_P - quad_u*0.5f - quad_v*0.5f;
	float3 v1 = quad_P + quad_u*0.5f - quad_v*0.5f;
	float3 v2 = quad_P + quad_u*0.5f + quad_v*0.5f;
	float3 v3 = quad_P - quad_u*0.5f + quad_v*0.5f;

	if(ray_triangle_intersect(ray_P, ray_D, ray_t, v0, v1, v2, isect_P, isect_t))
		return true;
	else if(ray_triangle_intersect(ray_P, ray_D, ray_t, v0, v2, v3, isect_P, isect_t))
		return true;
	
	return false;
}

CCL_NAMESPACE_END

#endif /* __UTIL_MATH_H__ */

