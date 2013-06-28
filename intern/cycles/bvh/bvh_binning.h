/*
 * Adapted from code copyright 2009-2011 Intel Corporation
 * Modifications Copyright 2012, Blender Foundation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __BVH_BINNING_H__
#define __BVH_BINNING_H__

#include "bvh_params.h"

#include "util_types.h"

#define ENABLE_TRACE_BIN
#ifdef ENABLE_TRACE_BIN
#define TRACE_BIN_ONLY(e) e
#define TRACE_BIN(...) ((void)printf(__VA_ARGS__), (void)fflush(stdout))
#else
#define TRACE_BIN_ONLY(e) ((void)0)
#define TRACE_BIN(...) ((void)0)
#endif

CCL_NAMESPACE_BEGIN

/* Single threaded object binner. Finds the split with the best SAH heuristic
 * by testing for each dimension multiple partitionings for regular spaced
 * partition locations. A partitioning for a partition location is computed,
 * by putting primitives whose centroid is on the left and right of the split
 * location to different sets. The SAH is evaluated by computing the number of
 * blocks occupied by the primitives in the partitions. */

class BVHObjectBinning : public BVHRange
{
public:
	__forceinline BVHObjectBinning() {}
	BVHObjectBinning(const BVHRange& job, BVHReference *prims);

	void split(BVHReference *prims, BVHObjectBinning& left_o, BVHObjectBinning& right_o) const;

	float splitSAH;	/* SAH cost of the best split */
	float leafSAH;	/* SAH cost of creating a leaf */

protected:
	int dim;			/* best split dimension */
	int pos;			/* best split position */
	size_t num_bins;	/* actual number of bins to use */
	float3 scale;		/* scaling factor to compute bin */

	enum { MAX_BINS = 32 };
	enum { LOG_BLOCK_SIZE = 2 };

	/* computes the bin numbers for each dimension for a box. */
	__forceinline int4 get_bin(const BoundBox& box) const
	{
		float4 fa = make_float4((box.center2() - cent_bounds().min)*scale - make_float3(0.5f), 0.0f);
		TRACE_BIN("fa = %.1e %.1e %.1e %.1e\n", fa.x, fa.y, fa.z, fa.w);

		int4 a = convert_int4(fa);
		TRACE_BIN("a = %d %d %d %d\n", a.x, a.y, a.z, a.w);

		int4 mn = make_int4(0);
		int4 mx = make_int4((int)num_bins-1);

		int4 r = clamp(a, mn, mx);
		TRACE_BIN("r = %d %d %d %d\n", r.x, r.y, r.z, r.w);

		return r;
	}

	/* computes the bin numbers for each dimension for a point. */
	__forceinline int4 get_bin(const float3& c) const
	{
		float4 f = make_float4((c - cent_bounds().min)*scale - make_float3(0.5f), 0);
		return convert_int4(f);
	}

	/* compute the number of blocks occupied for each dimension. */
	__forceinline float4 blocks(const int4& a) const
	{
		return convert_float4((a + make_int4((1 << LOG_BLOCK_SIZE)-1)) >> LOG_BLOCK_SIZE);
	}

	/* compute the number of blocks occupied in one dimension. */
	__forceinline int blocks(size_t a) const
	{
		return (int)((a+((1LL << LOG_BLOCK_SIZE)-1)) >> LOG_BLOCK_SIZE);
	}
};

CCL_NAMESPACE_END

#endif

