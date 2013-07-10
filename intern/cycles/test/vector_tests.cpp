
/* Due to the huge number of tests required to fully exercise the vector classes
 * these tests use preprocessor tricks extensively to avoid having to make hundreds
 * of test cases manually. Here's why:
 *
 * uchar2 uchar3 uchar4 uint2 uint3 uint4 float2 float3 float4 (9)
 * no SSE, SSE2, SSE3, SSE4 (5)
 *
 *  = 45 tests per test case
 *
 * 33 cases -> 1485 tests
 */

#include <string.h>
#include <iostream>

#include "util_types.h"

CCL_NAMESPACE_BEGIN

bool verbose;
bool failed;

/* generate the prototype for a specific test case */
#define DECLARE_TEST(type, width, arch, op) \
	void vector_test_##type##width##_##arch##_##op()

/* generate the prototypes for all architectures */
#define DECLARE_ARCH_TESTS(type, width, op) \
	DECLARE_TEST(type, width, SSE0, op); \
	DECLARE_TEST(type, width, SSE2, op); \
	DECLARE_TEST(type, width, SSE3, op); \
	DECLARE_TEST(type, width, SSE4, op)

/* generate the prototypes for all widths */
#define DECLARE_WIDTH_TESTS(type, op) \
	DECLARE_ARCH_TESTS(type, 2, op); \
	DECLARE_ARCH_TESTS(type, 3, op); \
	DECLARE_ARCH_TESTS(type, 4, op)

/* generate the prototypes for all types */
#define DECLARE_TESTS(op) \
	DECLARE_WIDTH_TESTS(uchar, op); \
	DECLARE_WIDTH_TESTS(uint, op); \
	DECLARE_WIDTH_TESTS(int, op); \
	DECLARE_WIDTH_TESTS(float, op)

/* generate the prototype for a specific test case */
#define INVOKE_TEST(type, width, arch, op) \
	vector_test_##type##width##_##arch##_##op()

/* generate the prototypes for all architectures */
#define INVOKE_ARCH_TESTS(type, width, op) \
	INVOKE_TEST(type, width, SSE0, op); \
	INVOKE_TEST(type, width, SSE2, op); \
	INVOKE_TEST(type, width, SSE3, op); \
	INVOKE_TEST(type, width, SSE4, op)

/* generate the prototypes for all widths */
#define INVOKE_WIDTH_TESTS(type, op) \
	INVOKE_ARCH_TESTS(type, 2, op); \
	INVOKE_ARCH_TESTS(type, 3, op); \
	INVOKE_ARCH_TESTS(type, 4, op)

/* generate the prototypes for all types */
#define INVOKE_TESTS(op) \
	INVOKE_WIDTH_TESTS(uchar, op); \
	INVOKE_WIDTH_TESTS(uint, op); \
	INVOKE_WIDTH_TESTS(int, op); \
	INVOKE_WIDTH_TESTS(float, op)

DECLARE_TESTS(make_vector);
DECLARE_TESTS(make_scalar);

DECLARE_TESTS(rcp);
DECLARE_TESTS(neg);

DECLARE_TESTS(add);
DECLARE_TESTS(sub);
DECLARE_TESTS(mul);
DECLARE_TESTS(div);

DECLARE_TESTS(shr);
DECLARE_TESTS(shl);

DECLARE_TESTS(add_assign_vector);
DECLARE_TESTS(sub_assign_vector);
DECLARE_TESTS(mul_assign_vector);
DECLARE_TESTS(div_assign_vector);

DECLARE_TESTS(add_assign_scalar);
DECLARE_TESTS(sub_assign_scalar);
DECLARE_TESTS(mul_assign_scalar);
DECLARE_TESTS(div_assign_scalar);

DECLARE_TESTS(shr_assign);
DECLARE_TESTS(shl_assign);

DECLARE_TESTS(min);
DECLARE_TESTS(max);
DECLARE_TESTS(clamp);

DECLARE_TESTS(shuffle);
DECLARE_TESTS(extract);
DECLARE_TESTS(insert);

DECLARE_TESTS(makemask_compare);

DECLARE_TESTS(convert);

DECLARE_TESTS(dot);
DECLARE_TESTS(cross);
DECLARE_TESTS(length);
DECLARE_TESTS(reduce_add);

/* scalars */

DECLARE_TESTS(float_as_int);
DECLARE_TESTS(int_as_float);

/* performance */

DECLARE_TESTS(perf);

static void runtests(bool perf_test)
{
	INVOKE_TESTS(make_vector);
	INVOKE_TESTS(make_scalar);

	INVOKE_TESTS(rcp);
	INVOKE_TESTS(length);
	INVOKE_TESTS(reduce_add);
	INVOKE_TESTS(neg);

	INVOKE_TESTS(add);
	INVOKE_TESTS(sub);
	INVOKE_TESTS(mul);
	INVOKE_TESTS(div);
	INVOKE_TESTS(shr);
	INVOKE_TESTS(shl);

	INVOKE_TESTS(add_assign_vector);
	INVOKE_TESTS(sub_assign_vector);
	INVOKE_TESTS(mul_assign_vector);
	INVOKE_TESTS(div_assign_vector);

	INVOKE_TESTS(add_assign_scalar);
	INVOKE_TESTS(sub_assign_scalar);
	INVOKE_TESTS(mul_assign_scalar);
	INVOKE_TESTS(div_assign_scalar);

	INVOKE_TESTS(shr_assign);
	INVOKE_TESTS(shl_assign);

	INVOKE_TESTS(min);
	INVOKE_TESTS(max);
	INVOKE_TESTS(clamp);

	INVOKE_TESTS(shuffle);
	INVOKE_TESTS(extract);
	INVOKE_TESTS(insert);

	INVOKE_TESTS(makemask_compare);

	INVOKE_TESTS(convert);

	INVOKE_TESTS(dot);
	INVOKE_TESTS(cross);

	/* scalars */

	INVOKE_TESTS(float_as_int);
	INVOKE_TESTS(int_as_float);

	if (perf_test) {
		/* performance */
		INVOKE_TESTS(perf);
	}
}

CCL_NAMESPACE_END

static void bad_argument(const char *bad_arg)
{
	std::cout << "Unknown command line argument: " << bad_arg << std::endl;
	ccl::failed = true;
}

static void usage()
{
	std::cout << "Usage:" << std::endl;
	std::cout << " -v or --verbose: verbose output" << std::endl;
	exit(0);
}

int main(int argc, char **argv)
{
	bool test_perf = false;

	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-?"))
			usage();
		else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v"))
			ccl::verbose = true;
		else if (!strcmp(argv[i], "--no-verbose"))
			ccl::verbose = false;
		else if (!strcmp(argv[i], "--perf") || !strcmp(argv[i], "-p"))
			test_perf = true;
		else if (!strcmp(argv[i], "--no-perf"))
			test_perf = false;
		else
			bad_argument(argv[i]);
	}

	if (!ccl::failed)
		ccl::runtests(test_perf);

	return ccl::failed ? 1 : 0;
}
