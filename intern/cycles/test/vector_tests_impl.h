
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

CCL_NAMESPACE_BEGIN

extern bool verbose;
extern bool failed;

template<typename T>
static void assert_check_value(const char *func_name, const char *file, int line, const char *expr, T expect, T actual)
{
	if (expect != actual) {
		std::cerr << "***" << func_name << " FAILED. " << file << ":" << line << std::endl <<
			"  Expected: " << expr << "==" << expect << std::endl <<
			"    Actual: " << actual << std::endl;
		failed = true;
	} else if (verbose) {
		std::cout << func_name << " PASSED at " << file << ':' << line << std::endl;
	}
}

// Overload to handle approximate comparisons
static void assert_check_value(const char *func_name, const char *file, int line, const char *expr, float expect, float actual)
{
	std::string expect_str, actual_str;
	std::stringstream ss;
	ss << std::scientific << std::setprecision(2) << expect;
	expect_str = ss.str();

	ss.str(std::string());
	ss << std::scientific << std::setprecision(2) << actual;
	actual_str = ss.str();

	if (expect_str != actual_str) {
		std::cerr << "***" << func_name << " FAILED. " << file << ":" << line << std::endl <<
			"  Expected: " << expr << "==" << std::scientific << expect << std::endl <<
			"    Actual: " << std::scientific << actual << std::endl;
		failed = true;
	} else if (verbose) {
		std::cout << func_name << " PASSED" << std::endl;
	}
}

#if defined _MSC_VER
#define test_assert_equal(expr, expect) \
	assert_check_value(__FUNCDNAME__, __FILE__, __LINE__, #expr, expect, (expr))
#elif defined __GNUC__
#define test_assert_equal(expr, expect) \
	assert_check_value(__FUNCTION__, __FILE__, __LINE__, #expr, expect, (expr))
#else
/* don't know how to get function name on this compiler */
#define test_assert_equal(expr, expect) \
	assert_check_value("", __FILE__, __LINE__, #expr, expect, (expr))
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
