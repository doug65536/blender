#ifndef UTIL_ATOMIC_H
#define UTIL_ATOMIC_H

/* Implements atomic compare and swap operation for either GCC or MSVC compilers.
 * This implements a simplified atomic model, pointers are not specially handled,
 * a simple intptr_t is used. AtomicIntN and AtomicIntPtr are provided to allow
 * code using this wrapper to be portable */

#if defined(__GNUCXX__)

/* Atomically:
 * oldval = val;
 * if (oldval == expect_val) {
 *   val = new_val;
 *   return expect_val;
 * }
 * else {
 *   return oldval;
 * }
 */
template<typename T>
inline T atomic_cas(volatile T &val, T new_val, T expect_val)
{
	return __sync_val_compare_and_swap(&val, new_val, expect_val);
}

#elif defined(_MSC_VER)

template<typename T>
struct MsvcAtomicImpl;

template<>
struct MsvcAtomicImpl<1>
{
	typedef char value_type;
	static inline value_type cas(volatile value_type *val, value_type new_val, value_type expect_val)
	{
		return (value_type)_InterlockedCompareExchange8(val, new_val, expect_val);
	}
};

template<>
struct MsvcAtomicImpl<2>
{
	typedef short value_type;
	static inline value_type cas(volatile value_type *val, value_type new_val, value_type expect_val)
	{
		return (value_type)_InterlockedCompareExchange16(val, new_val, expect_val);
	}
};

template<>
struct MsvcAtomicImpl<4>
{
	typedef int value_type;
	static inline value_type cas(volatile value_type *val, value_type new_val, value_type expect_val)
	{
		return (value_type)_InterlockedCompareExchange(val, new_val, expect_val);
	}
};

template<>
struct MsvcAtomicImpl<8>
{
	typedef __int64 value_type;
	static inline value_type cas(volatile value_type *val, value_type new_val, value_type expect_val)
	{
		return (value_type)_InterlockedCompareExchange64(val, new_val, expect_val);
	}
};

template<typename T>
inline void atomic_cas(volatile T &val, T new_val, T expect_val)
{
	typedef typename MsvcAtomicImpl<sizeof(T)> ImplType;
	typedef ImplType::value_type CastType;
	return ImplType::cas(
		reinterpret_cast<volatile CastType*>(&val),
		*reinterpret_cast<CastType*>(&new_val),
		*reinterpret_cast<CastType*>(&expect_val));
}

/* declare types */
typedef typename MsvcAtomicImpl<1> AtomicInt8;
typedef typename MsvcAtomicImpl<2> AtomicInt16;
typedef typename MsvcAtomicImpl<4> AtomicInt32;
typedef typename MsvcAtomicImpl<8> AtomicInt64;
typedef typename MsvcAtomicImpl<sizeof(void*)> AtomicIntPtr;

#elif __cplusplus >= 201103L

template<typename T>
inline T atomic_cas(std::atomic<T> &val, T new_val, T expect_val)
{
	/* if val != expect_val { expect_val = val } else { val = new_val } */
	val.compare_exchange_strong(expect_val, new_val);
	return expect_val;
}

/* declare types */
typedef std::atomic_int_least8_t AtomicInt8;
typedef std::atomic_int_least16_t AtomicInt16;
typedef std::atomic_int_least32_t AtomicInt32;
typedef std::atomic_int_least64_t AtomicInt64;
typedef std::atomic_intptr_t AtomicIntPtr;
#endif

#ifdef UTIL_ATOMIC_TEST
void test_util_atomic_h()
{
	AtomicInt8 i8 = 8, o8;
	AtomicInt16 i16 = 16, o16;
	AtomicInt32 i32 = 32, o32;
	AtomicInt64 i64 = 64, o64;
	int x0 = 33, x1 = 66;
	AtomicIntPtr p = (intptr_t)&x0, op;

	/* do tests where they don't change the value */
	o8 = atomic_cas(&i8, 0, 1);
	o16 = atomic_cas(&i8, 0, 1);
	o32 = atomic_cas(&i8, 0, 1);
	o64 = atomic_cas(&i8, 0, 1);
	op = atomic_cas(&p, (intptr_t)0, (intptr_t)1);

	/* make sure none of them changed */
	assert(i8 == 8);
	assert(i16 == 16);
	assert(i32 == 32);
	assert(i64 == 64);
	assert(i64 == (intptr_t)&x0);

	/* make sure they correctly returned the existing value */
	assert(o8 == 8);
	assert(o16 == 16);
	assert(o32 == 32);
	assert(o64 == 64);
	assert(o64 == (intptr_t)&x0);

	/* do tests where they DO change the value */
	o8 = atomic_cas(&i8, 108, 8);
	o16 = atomic_cas(&i8, 116, 16);
	o32 = atomic_cas(&i8, 132, 32);
	o64 = atomic_cas(&i8, 164, 64);
	op = atomic_cas(&p, (uintptr_t)&x1, (uintptr_t)&x0);

	/* make sure they correctly returned the existing value */
	assert(o8 == 8);
	assert(o16 == 16);
	assert(o32 == 32);
	assert(o64 == 64);
	assert(o64 == (uintptr_t)&x0);

	/* make sure they changed */
	assert(i8 == 108);
	assert(i16 == 116);
	assert(i32 == 132);
	assert(i64 == 164);
	assert(p == (uintptr_t)&x1);
}

#endif

#endif // UTIL_ATOMIC_H
