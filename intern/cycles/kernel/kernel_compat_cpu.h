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

#ifndef __KERNEL_COMPAT_CPU_H__
#define __KERNEL_COMPAT_CPU_H__

#define __KERNEL_CPU__

#include "util_debug.h"
#include "util_math.h"
#include "util_types.h"

CCL_NAMESPACE_BEGIN

/* Assertions inside the kernel only work for the CPU device, so we wrap it in
 * a macro which is empty for other devices */

#define kernel_assert(cond) assert(cond)

/* Texture types to be compatible with CUDA textures. These are really just
 * simple arrays and after inlining fetch hopefully revert to being a simple
 * pointer lookup. */

template<typename T> struct texture  {
	__forceinline T fetch(int index)
	{
		kernel_assert(index >= 0 && index < width);
		return data[index];
	}

#if 0
	__m128 fetch_m128(int index)
	{
		kernel_assert(index >= 0 && index < width);
		return ((__m128*)data)[index];
	}

	__m128i fetch_m128i(int index)
	{
		kernel_assert(index >= 0 && index < width);
		return ((__m128i*)data)[index];
	}
#endif

	T *data;
	int width;
};

template<typename T> struct texture_image  {
	__forceinline float4 read(float4 r)
	{
		return r;
	}

	__forceinline float4 read(uchar4 r)
	{
		float4 rcpf = make_float4_1(1.0f/255.0f);
		float4 color = convert_float4(r);
		return color * rcpf;
	}

	int wrap_periodic(int x, int width)
	{
		//if (x >= width || x <= 0)
			x %= width;
		if(x < 0)
			x += width;
		return x;
	}

	int wrap_clamp(int x, int width)
	{
		return clamp(x, 0, width-1);
	}

	float frac(float x, int *ix)
	{
		int i = float_to_int(x);// - ((x < 0.0f)? 1: 0);
		/* sign extend i to either -1 or 0, makes it branchless */
		i += i >> 31;
		*ix = i;
		return x - (float)i;
	}

	/* bilinear interpolation */
	float4 interp(float x, float y, bool periodic = true)
	{
		if(!data)
			return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		int ix, iy, nix, niy;
		float tx = frac(x*width - 0.5f, &ix);
		float ty = frac(y*height - 0.5f, &iy);

		if(periodic) {
			ix = wrap_periodic(ix, width);
			iy = wrap_periodic(iy, height);

			nix = wrap_periodic(ix+1, width);
			niy = wrap_periodic(iy+1, height);
		}
		else {
			ix = wrap_clamp(ix, width);
			iy = wrap_clamp(iy, height);

			nix = wrap_clamp(ix+1, width);
			niy = wrap_clamp(iy+1, height);
		}

		/* do all the uchar4 to float4 conversions in a batch to help cse */
		/* top left, top right, bottom left, bottom right */
		float4 tl = read(data[ix + iy*width]);
		float4 tr = read(data[nix + iy*width]);
		float4 bl = read(data[ix + niy*width]);
		float4 br = read(data[nix + niy*width]);

		float4 r;
		r  = ((1.0f - ty) * (1.0f - tx)) * tl;
		r += ((1.0f - ty) *         tx ) * tr;
		r += (        ty  * (1.0f - tx)) * bl;
		r += (        ty  *         tx ) * br;
		return r;

		//float4 r = (1.0f - ty)*(1.0f - tx)*read(data[ix + iy*width]);
		//r += (1.0f - ty)*tx*read(data[nix + iy*width]);
		//r += ty*(1.0f - tx)*read(data[ix + niy*width]);
		//r += ty*tx*read(data[nix + niy*width]);
		//return r;
	}

	T *data;
	int width, height;
};

typedef texture<float4> texture_float4;
typedef texture<float2> texture_float2;
typedef texture<float> texture_float;
typedef texture<uint> texture_uint;
typedef texture<int> texture_int;
typedef texture<uint4> texture_uint4;
typedef texture<uchar4> texture_uchar4;
typedef texture_image<float4> texture_image_float4;
typedef texture_image<uchar4> texture_image_uchar4;

/* Macros to handle different memory storage on different devices */

#define kernel_tex_fetch(tex, index) (kg->tex.fetch(index))
#define kernel_tex_fetch_m128(tex, index) (kg->tex.fetch_m128(index))
#define kernel_tex_fetch_m128i(tex, index) (kg->tex.fetch_m128i(index))
#define kernel_tex_lookup(tex, t, offset, size) (kg->tex.lookup(t, offset, size))
#define kernel_tex_image_interp(tex, x, y) ((tex < MAX_FLOAT_IMAGES) ? kg->texture_float_images[tex].interp(x, y) : kg->texture_byte_images[tex - MAX_FLOAT_IMAGES].interp(x, y))

#define kernel_data (kg->__data)

CCL_NAMESPACE_END

#endif /* __KERNEL_COMPAT_CPU_H__ */

