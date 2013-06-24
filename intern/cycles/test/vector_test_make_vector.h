#ifndef VECTOR_TYPE
#error Preprocessor trick requires VECTOR_TYPE defined
#endif

#undef VECTOR_SIZE
#define VECTOR_SIZE 2
#include "vector_test_make_vector_impl.h"

#undef VECTOR_SIZE
#define VECTOR_SIZE 3
#include "vector_test_make_vector_impl.h"

#undef VECTOR_SIZE
#define VECTOR_SIZE 4
#include "vector_test_make_vector_impl.h"

