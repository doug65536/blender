#ifndef TEST_VARIATION
#error Preprocessor trick requires TEST_VARIATION defined
#endif
#ifndef VECTOR_TYPE
#error Preprocessor trick requires VECTOR_TYPE defined
#endif
#ifndef VECTOR_SIZE
#error Preprocessor trick requires VECTOR_SIZE defined
#endif

/* each recursion expands a nested expansion, do lots just in case */
#define JOIN6(a,b) a##b
#define JOIN5(a,b) JOIN6(a,b)
#define JOIN4(a,b) JOIN5(a,b)
#define JOIN3(a,b) JOIN4(a,b)
#define JOIN2(a,b) JOIN3(a,b)
#define JOIN1(a,b) JOIN2(a,b)
#define JOIN(a,b) JOIN1(a,b)

/* example: _SSE2 */
#define ARCH_SUFFIX JOIN(_SSE, TEST_VARIATION)

/* example: float4 */
#define VECTOR_TYPE_NAME JOIN(VECTOR_TYPE, VECTOR_SIZE)

/* example: int4 */
#undef MASK_TYPE_NAME
#ifdef VECTOR_IS_FLOAT
#define MASK_TYPE_NAME JOIN(int, VECTOR_SIZE)
#else
#define MASK_TYPE_NAME VECTOR_TYPE_NAME
#endif

/* example: float4 */
#undef FLOAT_TYPE_NAME
#define FLOAT_TYPE_NAME JOIN(float, VECTOR_SIZE)

/* example: int4 */
#undef INT_TYPE_NAME
#define INT_TYPE_NAME JOIN(int, VECTOR_SIZE)

/* example: make_float4 */
#define MAKE_VECTOR_FUNCTION JOIN(make_, VECTOR_TYPE_NAME)

/* example: make_int4 */
#define MAKE_INT_VECTOR_FUNCTION JOIN(make_, INT_TYPE_NAME)

/* example: make_int4 */
#define MAKE_FLOAT_VECTOR_FUNCTION JOIN(make_, FLOAT_TYPE_NAME)

/* example: make_int4 */
#define MAKE_MASK_FUNCTION JOIN(make_, MASK_TYPE_NAME)

/* example: test_float4 */
#define TEST_FUNCTION_NAME_TYPE JOIN(vector_test_, VECTOR_TYPE_NAME)

/* example: test_float4_SSE2 */
#define TEST_FUNCTION_NAME_TYPE_ARCH JOIN(TEST_FUNCTION_NAME_TYPE, ARCH_SUFFIX)

/* example: test_float4_SSE2_add */
#define TEST_FUNCTION_NAME(op) JOIN(JOIN(TEST_FUNCTION_NAME_TYPE_ARCH, _), op)()

/* prototype followed by definition - to silence warning */
#define TEST_FUNCTION(op) void TEST_FUNCTION_NAME(op); void TEST_FUNCTION_NAME(op)

/* generate arguments for call to make_<type><width>,
 * ignore values for nonexistent members */

#undef MAKE_VECTOR_PARAMS
#if VECTOR_SIZE==2
#define MAKE_VECTOR_PARAMS(a,b,c,d) ((VECTOR_TYPE)a, (VECTOR_TYPE)b)
#elif VECTOR_SIZE==3
#define MAKE_VECTOR_PARAMS(a,b,c,d) ((VECTOR_TYPE)a, (VECTOR_TYPE)b, (VECTOR_TYPE)c)
#elif VECTOR_SIZE==4
#define MAKE_VECTOR_PARAMS(a,b,c,d) ((VECTOR_TYPE)a, (VECTOR_TYPE)b, (VECTOR_TYPE)c, (VECTOR_TYPE)d)
#else
#error invalid VECTOR_SIZE
#endif

#undef VERIFY
#define VERIFY(actual, expect) test_assert_equal((actual), (expect))

/* verify the values of vector members, ignore checks of members that don't exist */

#undef VERIFY_X
#undef VERIFY_Y
#undef VERIFY_Z
#undef VERIFY_W

#define VERIFY_X(vec,value) test_assert_equal(vec .x, (VECTOR_TYPE)value)
#define VERIFY_Y(vec,value) test_assert_equal(vec .y, (VECTOR_TYPE)value)

#if VECTOR_SIZE > 2
#define VERIFY_Z(vec,value) test_assert_equal(vec .z, (VECTOR_TYPE)value)
#else
#define VERIFY_Z(vec,value) (void)0
#endif

#if VECTOR_SIZE > 3
#define VERIFY_W(vec,value) test_assert_equal(vec .w, (VECTOR_TYPE)(value));
#else
#define VERIFY_W(vec,value) (void)0
#endif

#if VECTOR_SIZE > 3
#define VERIFY_W(vec,value) test_assert_equal(vec .w, (VECTOR_TYPE)(value));
#else
#define VERIFY_W(vec,value) (void)0
#endif

#define VERIFY_XYZW(vec,xv,yv,zv,wv) \
	do { \
		VERIFY_X(vec,xv); \
		VERIFY_Y(vec,yv); \
		VERIFY_Z(vec,zv); \
		VERIFY_W(vec,wv); \
	} while (0)

TEST_FUNCTION(make_vector)
{
	VECTOR_TYPE_NAME n = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);

	VERIFY_XYZW(n, 2, 3, 4, 5);
}

TEST_FUNCTION(make_scalar)
{
	VECTOR_TYPE_NAME n = MAKE_VECTOR_FUNCTION (2);

	VERIFY_XYZW(n, 2, 2, 2, 2);
}

TEST_FUNCTION(neg)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	VECTOR_TYPE_NAME n = -n0;

	VERIFY_XYZW(n, -2, -3, -4, -5);
}

TEST_FUNCTION(add)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	VECTOR_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(6, 7, 8, 9);

	VECTOR_TYPE_NAME n = n0 + n1;
	VERIFY_XYZW(n, 8, 10, 12, 14);
}

TEST_FUNCTION(sub)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	VECTOR_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(6, 7, 8, 9);

	VECTOR_TYPE_NAME n = n1 - n0;
	VERIFY_XYZW(n, 4, 4, 4, 4);
}

TEST_FUNCTION(mul)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	VECTOR_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(6, 7, 8, 9);

	VECTOR_TYPE_NAME n = n0 * n1;
	VERIFY_XYZW(n, 2*6, 3*7, 4*8, 5*9);
}

TEST_FUNCTION(rcp)
{
#ifdef VECTOR_IS_FLOAT
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);

	VECTOR_TYPE_NAME n = rcp(n0);

	VERIFY_X(n, (VECTOR_TYPE)1 / (VECTOR_TYPE)2);
	VERIFY_Y(n, (VECTOR_TYPE)1 / (VECTOR_TYPE)3);
	VERIFY_Z(n, (VECTOR_TYPE)1 / (VECTOR_TYPE)4);
	VERIFY_W(n, (VECTOR_TYPE)1 / (VECTOR_TYPE)5);
#endif
}

TEST_FUNCTION(div)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);

	VECTOR_TYPE_NAME n;

	n = (n0 * n0);
	VERIFY_XYZW(n, 2*2, 3*3, 4*4, 5*5);

	n = n / n0;
	VERIFY_XYZW(n, 2, 3, 4, 5);
}

TEST_FUNCTION(shr)
{
#ifdef VECTOR_IS_INTEGER
#endif
}

TEST_FUNCTION(shl)
{
#ifdef VECTOR_IS_INTEGER
#endif
}

TEST_FUNCTION(min)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	VECTOR_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(6, 7, 8, 9);

	VECTOR_TYPE_NAME t0, t1;

	t0 = min(n0, n1);
	VERIFY_XYZW(t0, 2, 3, 4, 5);

	t1 = min(n1, n0);
	VERIFY_XYZW(t1, 2, 3, 4, 5);

#ifndef VECTOR_IS_UNSIGNED
	/* test negative numbers */

	n0 = -n0;
	n1 = -n1;

	t0 = min(n0, n1);
	t1 = min(n1, n0);

	VERIFY_XYZW(t0, -6, -7, -8, -9);
	VERIFY_XYZW(t1, -6, -7, -8, -9);
#endif
}

TEST_FUNCTION(max)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	VECTOR_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(6, 7, 8, 9);

	VECTOR_TYPE_NAME t0 = max(n0, n1);
	VECTOR_TYPE_NAME t1 = max(n1, n0);

	VERIFY_XYZW(t0, 6, 7, 8, 9);
	VERIFY_XYZW(t1, 6, 7, 8, 9);
}

TEST_FUNCTION(clamp)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);

	/* vector of low numbers */
	VECTOR_TYPE_NAME clo = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(0,0,0,0);

	/* vector of high numbers */
	VECTOR_TYPE_NAME chi = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(10,10,10,10);

	/* vector where even elements get clamped */
	VECTOR_TYPE_NAME cehi = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(10,0,10,0);

	/* vector where odd elements get clamped */
	VECTOR_TYPE_NAME cohi = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(0,10,0,10);

	VECTOR_TYPE_NAME t0;

	/* should not affect */
	t0 = clamp(n0, clo, chi);
	VERIFY_XYZW(t0, 2, 3, 4, 5);

	/* should only increase even members */
	t0 = clamp(n0, cehi, chi);
	VERIFY_XYZW(t0, 10, 3, 10, 5);

	/* should only increase odd members */
	t0 = clamp(n0, cohi, chi);
	VERIFY_XYZW(t0, 2, 10, 4, 10);

	/* should only decrease even members */
	t0 = clamp(n0, clo, cohi);
	VERIFY_XYZW(t0, 0, 3, 0, 5);

	/* should only decrease odd members */
	t0 = clamp(n0, clo, cehi);
	VERIFY_XYZW(t0, 2, 0, 4, 0);

	/* should increase all members */
	t0 = clamp(n0, chi, chi);
	VERIFY_XYZW(t0, 10, 10, 10, 10);

	/* should decrease all members */
	t0 = clamp(n0, clo, clo);
	VERIFY_XYZW(t0, 0, 0, 0, 0);
}

TEST_FUNCTION(shuffle)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);

	VECTOR_TYPE_NAME t0;

#if VECTOR_SIZE == 2
	t0 = shuffle<1, 0>(n0);

	VERIFY_X(t0, 3);
	VERIFY_Y(t0, 2);
#elif VECTOR_SIZE == 3
	t0 = shuffle<2, 0, 1>(n0);

	VERIFY_X(t0, 4);
	VERIFY_Y(t0, 2);
	VERIFY_Z(t0, 3);
#elif VECTOR_SIZE == 4
	t0 = shuffle<2, 3, 0, 1>(n0);
	VERIFY_XYZW(t0, 4, 5, 2, 3);

	/* test SSE3 template specializations */

	t0 = shuffle<0, 0, 2, 2>(n0);
	VERIFY_XYZW(t0, 2, 2, 4, 4);

	t0 = shuffle<1, 1, 3, 3>(n0);
	VERIFY_XYZW(t0, 3, 3, 5, 5);

	t0 = shuffle<0, 1, 0, 1>(n0);
	VERIFY_XYZW(t0, 2, 3, 2, 3);
#endif
}

TEST_FUNCTION(extract)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(1, 2, 3, 4);
	VECTOR_TYPE t0;

	t0 = extract<0>(n0);
	VERIFY(t0, 1);
	t0 = extract<1>(n0);
	VERIFY(t0, 2);
#if VECTOR_SIZE > 2
	t0 = extract<2>(n0);
	VERIFY(t0, 3);
#endif
#if VECTOR_SIZE > 3
	t0 = extract<3>(n0);
	VERIFY(t0, 4);
#endif
}

TEST_FUNCTION(insert)
{

}

TEST_FUNCTION(makemask_compare)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(0, 1, 2, 3);
	VECTOR_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 1, 2, 1);
	/*                                                            < == ==  > */

	MASK_TYPE_NAME t0 = n0 < n1;
	MASK_TYPE_NAME t1 = n0 <= n1;
	MASK_TYPE_NAME t2 = n0 == n1;
	MASK_TYPE_NAME t3 = n0 != n1;
	MASK_TYPE_NAME t4 = n0 >= n1;
	MASK_TYPE_NAME t5 = n0 > n1;

	MASK_TYPE_NAME zero = MAKE_MASK_FUNCTION MAKE_VECTOR_PARAMS (0,0,0,0);
	MASK_TYPE_NAME one =  MAKE_MASK_FUNCTION MAKE_VECTOR_PARAMS (1,1,1,1);

	MASK_TYPE_NAME tc;

	/* verify lt */
	tc = mask_select(t0 != zero, one, zero);
	VERIFY_XYZW(tc, 1, 0, 0, 0);

	/* verify le */
	tc = mask_select(t1 != zero, one, zero);
	VERIFY_XYZW(tc, 1, 1, 1, 0);

	/* verify eq */
	tc = mask_select(t2 != zero, one, zero);
	VERIFY_XYZW(tc, 0, 1, 1, 0);

	/* verify ne */
	tc = mask_select(t3 != zero, one, zero);
	VERIFY_XYZW(tc, 1, 0, 0, 1);

	/* verify ge */
	tc = mask_select(t4 != zero, one, zero);
	VERIFY_XYZW(tc, 0, 1, 1, 1);

	/* verify gt */
	tc = mask_select(t5 != zero, one, zero);
	VERIFY_XYZW(tc, 0, 0, 0, 1);

	/* do the tests in reverse to fully test small vectors */

	n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(3, 2, 1, 0);
	n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(1, 2, 1, 2);
	/*                                           > == == < */

	t0 = n0 < n1;
	t1 = n0 <= n1;
	t2 = n0 == n1;
	t3 = n0 != n1;
	t4 = n0 >= n1;
	t5 = n0 > n1;

	/* verify lt */
	tc = mask_select(t0 != zero, one, zero);
	VERIFY_XYZW(tc, 0, 0, 0, 1);

	/* verify le */
	tc = mask_select(t1 != zero, one, zero);
	VERIFY_XYZW(tc, 0, 1, 1, 1);

	/* verify eq */
	tc = mask_select(t2 != zero, one, zero);
	VERIFY_XYZW(tc, 0, 1, 1, 0);

	/* verify ne */
	tc = mask_select(t3 != zero, one, zero);
	VERIFY_XYZW(tc, 1, 0, 0, 1);

	/* verify ge */
	tc = mask_select(t4 != zero, one, zero);
	VERIFY_XYZW(tc, 1, 1, 1, 0);

	/* verify gt */
	tc = mask_select(t5 != zero, one, zero);
	VERIFY_XYZW(tc, 1, 0, 0, 0);
}

TEST_FUNCTION(convert)
{
	//FLOAT_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	//INT_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(6, 7, 8, 9);
}

TEST_FUNCTION(dot)
{
	VECTOR_TYPE_NAME n0 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(2, 3, 4, 5);
	VECTOR_TYPE_NAME n1 = MAKE_VECTOR_FUNCTION MAKE_VECTOR_PARAMS(6, 7, 8, 9);

	VECTOR_TYPE n;

	n = dot(n0, n1);

#if VECTOR_SIZE == 2
	VERIFY(n, 2*6 + 3*7);
#elif VECTOR_SIZE == 3
	VERIFY(n, 2*6 + 3*7 + 4*8);
#elif VECTOR_SIZE == 4
	VERIFY(n, 2*6 + 3*7 + 4*8 + 5*9);
#endif
}

TEST_FUNCTION(cross)
{

}