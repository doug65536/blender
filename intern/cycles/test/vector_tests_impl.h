/* this file is included in each cpp file with different SSE code generation options
 * TEST_VARIATION is 0 for no sse, 2 for sse2, 3 for sse3, 4 for sse4.
 FIXME: add avx1 case */

#if TEST_VARIATION == 0

#define __KERNEL_SSE_DISABLED__ 1

#elif TEST_VARIATION == 2

#define __KERNEL_SSE__ 1

#elif TEST_VARIATION == 3

#define __KERNEL_SSE__ 1
#define __KERNEL_SSE3__ 1

#elif TEST_VARIATION == 4

#define __KERNEL_SSE__ 1
#define __KERNEL_SSE3__ 1
#define __KERNEL_SSSE3__ 1
#define __KERNEL_SSE4__ 1

#else
#error You must #define TEST_VARIATION appropriately before including this header
#endif

#include "util_math.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <signal.h>

CCL_NAMESPACE_BEGIN

extern bool verbose;
extern bool failed;

template<typename T, typename U>
static void assert_check_value(const char *func_name, const char *file, int line, const char *expr, const U& expect, const T& actual)
{
	if (std::numeric_limits<T>::is_integer) {
		if (expect == actual) {
			if (verbose)
				std::cout << func_name << " PASSED at " << file << ':' << line << std::endl;
			return;
		}

		if (!std::numeric_limits<T>::is_signed) {
			/* expand to unsigned so uchar won't be printed as a number, not a character */
			std::cerr << "*U*" << func_name << " FAILED. " << file << ":" << line << std::endl <<
				"  Expected: " << expr << "==" << (unsigned)expect << std::endl <<
				"    Actual: " << (unsigned)actual << std::endl;
		}
		else {
			/* expand to int so char won't be printed as a number, not a character */
			std::cerr << "*I*" << func_name << " FAILED. " << file << ":" << line << std::endl <<
				"  Expected: " << expr << "==" << (int)expect << std::endl <<
				"    Actual: " << (int)actual << std::endl;
		}
	}
	else {
		/* convert to limited precision scientific notation strings to do approximate equality */
		std::string expect_str, actual_str;
		std::stringstream ss;
		ss << std::scientific << std::setprecision(3) << (T)expect;
		expect_str = ss.str();

		ss.str(std::string());
		ss << std::scientific << std::setprecision(3) << actual;
		actual_str = ss.str();

		if (expect_str == actual_str) {
			if (verbose)
				std::cout << func_name << " PASSED at " << file << ':' << line << std::endl;
			return;
		}

		std::cerr << "*F*" << func_name << " FAILED. " << file << ":" << line << std::endl <<
			"  Expected: " << expr << "==" << std::scientific << expect << std::endl <<
			"    Actual: " << std::scientific << actual << std::endl;
	}

	failed = true;

	/* break into debugger if debug build */
#ifndef NDEBUG
	raise(SIGTRAP);
#endif
}

#if defined _MSC_VER
#define test_assert_equal(expr, expect) \
	assert_check_value(__FUNCDNAME__, __FILE__, __LINE__, #expr, (expect), (expr))
#elif defined __GNUC__
#define test_assert_equal(expr, expect) \
	assert_check_value(__FUNCTION__, __FILE__, __LINE__, #expr, (expect), (expr))
#else
/* don't know how to get function name on this compiler */
#define test_assert_equal(expr, expect) \
	assert_check_value("", __FILE__, __LINE__, #expr, (expect), (expr))
#endif

/* each generated test can check whether the type is integer (non-float) or unsigned */
#define VECTOR_IS_INTEGER
#define VECTOR_IS_UNSIGNED

#define VECTOR_TYPE uchar
#include "vector_test_make_vector.h"
#undef VECTOR_TYPE

#define VECTOR_TYPE uint
#include "vector_test_make_vector.h"
#undef VECTOR_TYPE

/* the rest are not unsigned */
#undef VECTOR_IS_UNSIGNED

#define VECTOR_TYPE int
#include "vector_test_make_vector.h"
#undef VECTOR_TYPE

#undef VECTOR_IS_INTEGER
#define VECTOR_IS_FLOAT

#define VECTOR_TYPE float
#include "vector_test_make_vector.h"
#undef VECTOR_TYPE

#undef VECTOR_IS_FLOAT

CCL_NAMESPACE_END
