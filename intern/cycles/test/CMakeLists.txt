# Copy stuff from cycles_util target
foreach(i INCLUDE_DIRECTORIES LINK_DIRECTORIES LINK_DEPENDS)
	get_property(CYCLES_UTIL_${i} TARGET cycles_util PROPERTY ${i})
	message(STATUS "CYCLES_UTIL_${i}=${CYCLES_UTIL_${i}}")
endforeach()

include_directories(${CYCLES_UTIL_INCLUDE_DIRECTORIES})

# FIXME: support different compilers
# FIXME: differentiate between SSE3 SSSE3 and SSE4 SSE4.1 SSE4.2
get_property(EXISTING_FLAGS SOURCE vector_tests_sse2.cpp PROPERTY COMPILE_FLAGS)
set_property(SOURCE vector_tests_sse2.cpp PROPERTY COMPILE_FLAGS "${EXISTING_FLAGS} -msse2")

get_property(EXISTING_FLAGS SOURCE vector_tests_sse3.cpp PROPERTY COMPILE_FLAGS)
set_property(SOURCE vector_tests_sse3.cpp PROPERTY COMPILE_FLAGS "${EXISTING_FLAGS} -msse3")

get_property(EXISTING_FLAGS SOURCE vector_tests_sse4.cpp PROPERTY COMPILE_FLAGS)
set_property(SOURCE vector_tests_sse4.cpp PROPERTY COMPILE_FLAGS "${EXISTING_FLAGS} -msse4")

set(EXISTING_FLAGS)

add_executable(intern_cycles_test
	vector_tests.cpp
	vector_tests_impl.h
	vector_tests_nosse.cpp
	vector_tests_sse2.cpp
	vector_tests_sse3.cpp
	vector_tests_sse4.cpp
	vector_test_make_vector.h
	vector_test_make_vector_impl.h
)
