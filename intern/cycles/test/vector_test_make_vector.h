#ifndef VECTOR_TYPE
#error Preprocessor trick requires VECTOR_TYPE defined
#endif

/* each recursion expands a nested expansion, do lots just in case */
#define JOIN6(a,b) a##b
#define JOIN5(a,b) JOIN6(a,b)
#define JOIN4(a,b) JOIN5(a,b)
#define JOIN3(a,b) JOIN4(a,b)
#define JOIN2(a,b) JOIN3(a,b)
#define JOIN1(a,b) JOIN2(a,b)
#define JOIN(a,b) JOIN1(a,b)

#define VERIFY(actual, expect) test_assert_equal((actual), (expect))

#undef VECTOR_SIZE
#define VECTOR_SIZE 2
#include "vector_test_make_vector_impl.h"

#undef VECTOR_SIZE
#define VECTOR_SIZE 3
#include "vector_test_make_vector_impl.h"

#undef VECTOR_SIZE
#define VECTOR_SIZE 4
#include "vector_test_make_vector_impl.h"

