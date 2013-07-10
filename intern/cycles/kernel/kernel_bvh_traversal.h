/*
 * Adapted from code Copyright 2009-2010 NVIDIA Corporation,
 * and code copyright 2009-2012 Intel Corporation
 *
 * Modifications Copyright 2011-2013, Blender Foundation.
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

/* This is a template BVH traversal function, where various features can be
 * enabled/disabled. This way we can compile optimized versions for each case
 * without new features slowing things down.
 *
 * BVH_INSTANCING: object instancing
 * BVH_HAIR: hair curve rendering
 * BVH_HAIR_MINIMUM_WIDTH: hair curve rendering with minimum width
 * BVH_SUBSURFACE: subsurface same object, random triangle intersection
 * BVH_MOTION: motion blur rendering
 *
 */

/* note: can't use variadic macros at all in opencl, using old technique */
//#define ENABLE_TRACE_BVH_INTERSECT
#ifdef ENABLE_TRACE_BVH_INTERSECT
#define TRACE_BVH_INTERSECT(p) printf p
#else
#define TRACE_BVH_INTERSECT(p) ((void)0)
#endif

//#define ENABLE_TRACE_LOOPS
#ifdef ENABLE_TRACE_BVH_LOOPS
#define TRACE_BVH_LOOPS(p)	p
#else
#define TRACE_BVH_LOOPS(p) ((void)0)
#endif

#define FEATURE(f) (((BVH_FUNCTION_FEATURES) & (f)) != 0)

__device bool BVH_FUNCTION_NAME
(KernelGlobals *kg, const Ray *ray, Intersection *isect
#if FEATURE(BVH_SUBSURFACE)
, int subsurface_object, float subsurface_random
#else
, const uint visibility
#endif
#if FEATURE(BVH_HAIR_MINIMUM_WIDTH) && !FEATURE(BVH_SUBSURFACE)
, uint *lcg_state, float difl, float extmax
#endif
)
{
	/* todo:
	 * - test if pushing distance on the stack helps (for non shadow rays)
	 * - separate version for shadow rays
	 * - likely and unlikely for if() statements
	 * - SSE for hair
	 * - test restrict attribute for pointers
	 */
	
	/* traversal stack in CUDA thread-local memory */
	int traversalStack[BVH_STACK_SIZE];
	traversalStack[0] = ENTRYPOINT_SENTINEL;

	/* traversal variables in registers */
	int stackPtr = 0;
	int nodeAddr = kernel_data.bvh.root;

	/* ray parameters in registers */
	const float tmax = ray->t;
	float3 P = ray->P;
	float3 idir = bvh_inverse_direction(ray->D);
	int object = ~0;

#if FEATURE(BVH_SUBSURFACE)
	const uint visibility = ~0;
	int num_hits = 0;
#endif

#if FEATURE(BVH_MOTION)
	Transform ob_tfm;
#endif

	isect->t = tmax;
	isect->object = ~0;
	isect->prim = ~0;
	isect->u = 0.0f;
	isect->v = 0.0f;

#if defined(__KERNEL_SSE2__) && !FEATURE(BVH_HAIR_MINIMUM_WIDTH)
	const shuffle_swap_t shuf_identity = shuffle_swap_identity();
	const shuffle_swap_t shuf_swap = shuffle_swap_swap();
	
	const int4 pn = make_int4(0, 0, 0x80000000, 0x80000000);
	float4 Psplat[3], idirsplat[3];

	float4 tmp4 = as_float4(P);
	Psplat[0] = S_xxxx(tmp4);// _mm_set_ps1(P.x);
	Psplat[1] = S_yyyy(tmp4);// _mm_set_ps1(P.y);
	Psplat[2] = S_zzzz(tmp4);// _mm_set_ps1(P.z);

	tmp4 = as_float4(idir);
	idirsplat[0] = _mm_xor_ps(S_xxxx(tmp4), _mm_castsi128_ps(pn));
	idirsplat[1] = _mm_xor_ps(S_yyyy(tmp4), _mm_castsi128_ps(pn));
	idirsplat[2] = _mm_xor_ps(S_zzzz(tmp4), _mm_castsi128_ps(pn));

	__m128 tsplat = _mm_set_ps(-isect->t, -isect->t, 0.0f, 0.0f);

#if 1
	shuffle_swap_t shufflex = (idir.x >= 0)? shuf_identity: shuf_swap;
	shuffle_swap_t shuffley = (idir.y >= 0)? shuf_identity: shuf_swap;
	shuffle_swap_t shufflez = (idir.z >= 0)? shuf_identity: shuf_swap;
#else

#if defined __KERNEL_SSSE3__
	float4 tmp2 = make_float4(0, 0, 0, 0);
	shuffle_swap_t shufflex = mask_select(S_xxxx(tmp4) >= tmp2, shuf_identity, shuf_swap);
	shuffle_swap_t shuffley = mask_select(S_yyyy(tmp4) >= tmp2, shuf_identity, shuf_swap);
	shuffle_swap_t shufflez = mask_select(S_zzzz(tmp4) >= tmp2, shuf_identity, shuf_swap);
#else
	shuffle_swap_t shufflex = mask_select(idir.x >= 0, shuf_identity, shuf_swap);
	shuffle_swap_t shuffley = mask_select(idir.y >= 0, shuf_identity, shuf_swap);
	shuffle_swap_t shufflez = mask_select(idir.z >= 0, shuf_identity, shuf_swap);
#endif

#endif

#endif

	TRACE_BVH_LOOPS(unsigned loops[4]);
	TRACE_BVH_LOOPS(loops[0] = 0);
	TRACE_BVH_LOOPS(loops[1] = 0);
	TRACE_BVH_LOOPS(loops[2] = 0);
	TRACE_BVH_LOOPS(loops[3] = 0);

	/* traversal loop */
	do {
		TRACE_BVH_LOOPS(++loops[0]);
		do {
			TRACE_BVH_LOOPS(++loops[1]);
			/* traverse internal nodes */
			while(nodeAddr >= 0 && nodeAddr != ENTRYPOINT_SENTINEL) {
				TRACE_BVH_LOOPS(++loops[2]);
				bool traverseChild0, traverseChild1;
				int nodeAddrChild1;

#if !defined(__KERNEL_SSE2__) || FEATURE(BVH_HAIR_MINIMUM_WIDTH)
				/* Intersect two child bounding boxes, non-SSE version */
				float t = isect->t;

				/* fetch node data */
				float4 node0 = kernel_tex_fetch(__bvh_nodes, nodeAddr*BVH_NODE_SIZE+0);
				float4 node1 = kernel_tex_fetch(__bvh_nodes, nodeAddr*BVH_NODE_SIZE+1);
				float4 node2 = kernel_tex_fetch(__bvh_nodes, nodeAddr*BVH_NODE_SIZE+2);
				float4 cnodes = kernel_tex_fetch(__bvh_nodes, nodeAddr*BVH_NODE_SIZE+3);

				//float3 c0lo;
				//float3 c0hi;

				//float3 c0lo = S_xxzz(node0);
				//float3 c0hi = S_

				/* intersect ray against child nodes */
				NO_EXTENDED_PRECISION float c0lox = (node0.x - P.x) * idir.x;
				NO_EXTENDED_PRECISION float c0hix = (node0.z - P.x) * idir.x;
				NO_EXTENDED_PRECISION float c0loy = (node1.x - P.y) * idir.y;
				NO_EXTENDED_PRECISION float c0hiy = (node1.z - P.y) * idir.y;
				NO_EXTENDED_PRECISION float c0loz = (node2.x - P.z) * idir.z;
				NO_EXTENDED_PRECISION float c0hiz = (node2.z - P.z) * idir.z;
				NO_EXTENDED_PRECISION float c0min = max4(min(c0lox, c0hix), min(c0loy, c0hiy), min(c0loz, c0hiz), 0.0f);
				NO_EXTENDED_PRECISION float c0max = min4(max(c0lox, c0hix), max(c0loy, c0hiy), max(c0loz, c0hiz), t);

				NO_EXTENDED_PRECISION float c1lox = (node0.y - P.x) * idir.x;
				NO_EXTENDED_PRECISION float c1hix = (node0.w - P.x) * idir.x;
				NO_EXTENDED_PRECISION float c1loy = (node1.y - P.y) * idir.y;
				NO_EXTENDED_PRECISION float c1hiy = (node1.w - P.y) * idir.y;
				NO_EXTENDED_PRECISION float c1loz = (node2.y - P.z) * idir.z;
				NO_EXTENDED_PRECISION float c1hiz = (node2.w - P.z) * idir.z;
				NO_EXTENDED_PRECISION float c1min = max4(min(c1lox, c1hix), min(c1loy, c1hiy), min(c1loz, c1hiz), 0.0f);
				NO_EXTENDED_PRECISION float c1max = min4(max(c1lox, c1hix), max(c1loy, c1hiy), max(c1loz, c1hiz), t);

#if FEATURE(BVH_HAIR_MINIMUM_WIDTH) && !FEATURE(BVH_SUBSURFACE)
				if(difl != 0.0f) {
					float hdiff = 1.0f + difl;
					float ldiff = 1.0f - difl;
					if(__float_as_int(cnodes.z) & PATH_RAY_CURVE) {
						c0min = max(ldiff * c0min, c0min - extmax);
						c0max = min(hdiff * c0max, c0max + extmax);
					}
					if(__float_as_int(cnodes.w) & PATH_RAY_CURVE) {
						c1min = max(ldiff * c1min, c1min - extmax);
						c1max = min(hdiff * c1max, c1max + extmax);
					}
				}
#endif

				/* decide which nodes to traverse next */
#ifdef __VISIBILITY_FLAG__
				/* this visibility test gives a 5% performance hit, how to solve? */
				traverseChild0 = (c0max >= c0min) && (__float_as_uint(cnodes.z) & visibility);
				traverseChild1 = (c1max >= c1min) && (__float_as_uint(cnodes.w) & visibility);
#else
				traverseChild0 = (c0max >= c0min);
				traverseChild1 = (c1max >= c1min);
#endif

#else // __KERNEL_SSE2__
				/* Intersect two child bounding boxes, SSE3 version adapted from Embree */

				/* fetch node data */
				__m128 *bvh_nodes = (__m128*)kg->__bvh_nodes.data + nodeAddr*BVH_NODE_SIZE;
				float4 cnodes = ((float4*)bvh_nodes)[3];

				/* intersect ray against child nodes */
				const __m128 tminmaxx = _mm_mul_ps(_mm_sub_ps(shuffle_swap(bvh_nodes[0], shufflex), Psplat[0]), idirsplat[0]);
				const __m128 tminmaxy = _mm_mul_ps(_mm_sub_ps(shuffle_swap(bvh_nodes[1], shuffley), Psplat[1]), idirsplat[1]);
				const __m128 tminmaxz = _mm_mul_ps(_mm_sub_ps(shuffle_swap(bvh_nodes[2], shufflez), Psplat[2]), idirsplat[2]);
				const __m128 tminmax = _mm_xor_ps(_mm_max_ps(_mm_max_ps(tminmaxx, tminmaxy), _mm_max_ps(tminmaxz, tsplat)), _mm_castsi128_ps(pn));
				const __m128 lrhit = _mm_cmple_ps(tminmax, shuffle_swap(tminmax, shuf_swap));

				/* decide which nodes to traverse next */
#ifdef __VISIBILITY_FLAG__
				/* this visibility test gives a 5% performance hit, how to solve? */
				traverseChild0 = (_mm_movemask_ps(lrhit) & 1) && (__float_as_uint(cnodes.z) & visibility);
				traverseChild1 = (_mm_movemask_ps(lrhit) & 2) && (__float_as_uint(cnodes.w) & visibility);
#else
				traverseChild0 = (_mm_movemask_ps(lrhit) & 1);
				traverseChild1 = (_mm_movemask_ps(lrhit) & 2);
#endif
#endif // __KERNEL_SSE2__

				nodeAddr = __float_as_int(cnodes.x);
				nodeAddrChild1 = __float_as_int(cnodes.y);

				if(traverseChild0 && traverseChild1) {
					/* both children were intersected, push the farther one */
#if !defined(__KERNEL_SSE2__) || FEATURE(BVH_HAIR_MINIMUM_WIDTH)
					bool closestChild1 = (c1min < c0min);
#else

#if 0
					union { __m128 m128; float v[4]; } uminmax;
					uminmax.m128 = tminmax;
					bool closestChild1 = uminmax.v[1] < uminmax.v[0];
#else
					__m128 closeCmp = _mm_cmplt_ss(_mm_shuffle_ps(tminmax, tminmax, _MM_SHUFFLE(3, 2, 1, 1)), tminmax);
					bool closestChild1 = _mm_movemask_ps(closeCmp) & 1;
#endif

#endif

					if(closestChild1) {
						int tmp = nodeAddr;
						nodeAddr = nodeAddrChild1;
						nodeAddrChild1 = tmp;
					}

					++stackPtr;
					kernel_assert(stackPtr < BVH_STACK_SIZE);
					traversalStack[stackPtr] = nodeAddrChild1;
				}
				else {
					/* one child was intersected */
					if(traverseChild1) {
						nodeAddr = nodeAddrChild1;
					}
					else if(!traverseChild0) {
						/* neither child was intersected */
						kernel_assert(stackPtr >= 0);
						nodeAddr = traversalStack[stackPtr];
						--stackPtr;
					}
				}
			}

			/* if node is leaf, fetch triangle list */
			if(nodeAddr < 0) {
				float4 leaf = kernel_tex_fetch(__bvh_nodes, (-nodeAddr-1)*BVH_NODE_SIZE+(BVH_NODE_SIZE-1));
				int primAddr = __float_as_int(leaf.x);

#if FEATURE(BVH_INSTANCING)
				if(primAddr >= 0) {
#endif
					int primAddr2 = __float_as_int(leaf.y);

					TRACE_BVH_INTERSECT(("Leaf with %d primitives\n", primAddr2 - primAddr));

					/* pop */
					nodeAddr = traversalStack[stackPtr];
					--stackPtr;

					/* primitive intersection */
					while(primAddr < primAddr2) {
						TRACE_BVH_LOOPS(++loops[3]);
						bool hit;

#if FEATURE(BVH_SUBSURFACE)
						/* only primitives from the same object */
						uint tri_object = (object == ~0)? kernel_tex_fetch(__prim_object, primAddr): object;

						if(tri_object == subsurface_object) {
#endif

							/* intersect ray against primitive */
#if FEATURE(BVH_HAIR)
							uint segment = kernel_tex_fetch(__prim_segment, primAddr);
#if !FEATURE(BVH_SUBSURFACE)
							if(segment != ~0) {

								if(kernel_data.curve_kernel_data.curveflags & CURVE_KN_INTERPOLATE) 
#if FEATURE(BVH_HAIR_MINIMUM_WIDTH)
									hit = bvh_cardinal_curve_intersect(kg, isect, P, idir, visibility, object, primAddr, segment, lcg_state, difl, extmax);
								else
									hit = bvh_curve_intersect(kg, isect, P, idir, visibility, object, primAddr, segment, lcg_state, difl, extmax);
#else
									hit = bvh_cardinal_curve_intersect(kg, isect, P, idir, visibility, object, primAddr, segment);
								else
									hit = bvh_curve_intersect(kg, isect, P, idir, visibility, object, primAddr, segment);
#endif
							}
							else
#endif
#endif
#if FEATURE(BVH_SUBSURFACE)
#if FEATURE(BVH_HAIR)
							if(segment == ~0)
#endif
							{
								hit = bvh_triangle_intersect_subsurface(kg, isect, P, idir, object, primAddr, tmax, &num_hits, subsurface_random);
								(void)hit;
							}

						}
#else
							hit = bvh_triangle_intersect(kg, isect, P, idir, visibility, object, primAddr);

							/* shadow ray early termination */
#if defined(__KERNEL_SSE2__) && !FEATURE(BVH_HAIR_MINIMUM_WIDTH)
							if(hit) {
								if(visibility == PATH_RAY_SHADOW_OPAQUE) {
									TRACE_BVH_LOOPS(printf("visi, loop counts: %3u { %3u { %3u %3u }}\n", loops[0], loops[1], loops[2], loops[3]));
									return true;
								}

								tsplat = _mm_set_ps(-isect->t, -isect->t, 0.0f, 0.0f);
							}
#else
							if(hit && visibility == PATH_RAY_SHADOW_OPAQUE) {
								TRACE_BVH_INTERSECT(("Returning: hit && visibility == PATH_RAY_SHADOW_OPAQUE\n"));
								TRACE_BVH_LOOPS(printf("htop, loop counts: %3u { %3u { %3u %3u }}\n", loops[0], loops[1], loops[2], loops[3]));
								return true;
							}
#endif

#endif

						primAddr++;
					}
				}
#if FEATURE(BVH_INSTANCING)
				else {
					/* instance push */
#if FEATURE(BVH_SUBSURFACE)
					if(subsurface_object == kernel_tex_fetch(__prim_object, -primAddr-1)) {
						object = subsurface_object;
#else
						object = kernel_tex_fetch(__prim_object, -primAddr-1);
#endif

#if FEATURE(BVH_MOTION)
						bvh_instance_motion_push(kg, object, ray, &P, &idir, &isect->t, &ob_tfm, tmax);
#else
						bvh_instance_push(kg, object, ray, &P, &idir, &isect->t, tmax);
#endif

#if defined(__KERNEL_SSE2__) && !FEATURE(BVH_HAIR_MINIMUM_WIDTH)
						float4 tmp;

						tmp = as_float4(P);
						Psplat[0] = S_xxxx(tmp);// _mm_set_ps1(P.x);
						Psplat[1] = S_yyyy(tmp);// _mm_set_ps1(P.y);
						Psplat[2] = S_zzzz(tmp);// _mm_set_ps1(P.z);

						tmp = as_float4(idir);
						idirsplat[0] = _mm_xor_ps(S_xxxx(tmp)/*_mm_set_ps1(idir.x)*/, _mm_castsi128_ps(pn));
						idirsplat[1] = _mm_xor_ps(S_yyyy(tmp)/*_mm_set_ps1(idir.y)*/, _mm_castsi128_ps(pn));
						idirsplat[2] = _mm_xor_ps(S_zzzz(tmp)/*_mm_set_ps1(idir.z)*/, _mm_castsi128_ps(pn));

						tsplat = _mm_set_ps(-isect->t, -isect->t, 0.0f, 0.0f);

						shufflex = (idir.x >= 0)? shuf_identity: shuf_swap;
						shuffley = (idir.y >= 0)? shuf_identity: shuf_swap;
						shufflez = (idir.z >= 0)? shuf_identity: shuf_swap;
#endif

						++stackPtr;
						traversalStack[stackPtr] = ENTRYPOINT_SENTINEL;

						nodeAddr = kernel_tex_fetch(__object_node, object);
#if FEATURE(BVH_SUBSURFACE)
					}
					else {
						/* pop */
						nodeAddr = traversalStack[stackPtr];
						--stackPtr;
					}
#endif
				}
			}
#endif
		} while(nodeAddr != ENTRYPOINT_SENTINEL);

		TRACE_BVH_INTERSECT(("Done internal nodes loop\n"));

#if FEATURE(BVH_INSTANCING)
		if(stackPtr >= 0) {
			kernel_assert(object != ~0);

			/* instance pop */
#if FEATURE(BVH_MOTION)
			bvh_instance_motion_pop(kg, object, ray, &P, &idir, &isect->t, &ob_tfm, tmax);
#else
			bvh_instance_pop(kg, object, ray, &P, &idir, &isect->t, tmax);
#endif

#if defined(__KERNEL_SSE2__) && !FEATURE(BVH_HAIR_MINIMUM_WIDTH)
			Psplat[0] = _mm_set_ps1(P.x);
			Psplat[1] = _mm_set_ps1(P.y);
			Psplat[2] = _mm_set_ps1(P.z);

			idirsplat[0] = _mm_xor_ps(_mm_set_ps1(idir.x), _mm_castsi128_ps(pn));
			idirsplat[1] = _mm_xor_ps(_mm_set_ps1(idir.y), _mm_castsi128_ps(pn));
			idirsplat[2] = _mm_xor_ps(_mm_set_ps1(idir.z), _mm_castsi128_ps(pn));

			tsplat = _mm_set_ps(-isect->t, -isect->t, 0.0f, 0.0f);

			shufflex = (idir.x >= 0)? shuf_identity: shuf_swap;
			shuffley = (idir.y >= 0)? shuf_identity: shuf_swap;
			shufflez = (idir.z >= 0)? shuf_identity: shuf_swap;
#endif

			object = ~0;
			nodeAddr = traversalStack[stackPtr];
			--stackPtr;
		}
#endif
	} while(nodeAddr != ENTRYPOINT_SENTINEL);

	TRACE_BVH_LOOPS(printf("done, loop counts: %3u { %3u { %3u %3u }}\n", loops[0], loops[1], loops[2], loops[3]));

#if FEATURE(BVH_SUBSURFACE)
	TRACE_BVH_INTERSECT(("Done (subsurface), hits = %d\n", num_hits));
	return (num_hits != 0);
#else
	TRACE_BVH_INTERSECT(("Done (non-subsurface), returns %s\n", (isect->prim != ~0) ? "true" : "false"));
	return (isect->prim != ~0);
#endif
}

#undef FEATURE
#undef BVH_FUNCTION_NAME
#undef BVH_FUNCTION_FEATURES

