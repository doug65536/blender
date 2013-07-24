/*
 * Copyright 2011, Blender Foundation.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef __DEVICE_NETWORK_H__
#define __DEVICE_NETWORK_H__

#if defined(WITH_NETWORK)

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/thread.hpp>

#include <iostream>
#include <sstream>
#include <deque>
#include <limits>
#include <stdint.h>
#include <stdlib.h>

#include "buffers.h"

#include "util_foreach.h"
#include "util_list.h"
#include "util_map.h"
#include "util_string.h"
#include "util_debugtrace.h"

CCL_NAMESPACE_BEGIN

using std::cout;
using std::cerr;
using std::hex;
using std::setw;
using std::exception;

using boost::asio::ip::tcp;

static const int SERVER_PORT = 5120;
static const int DISCOVER_PORT = 5121;
static const string DISCOVER_REQUEST_MSG = "REQUEST_RENDER_SERVER_IP";
static const string DISCOVER_REPLY_MSG = "REPLY_RENDER_SERVER_IP";

/* RPC protocol:
 * Each outgoing packet could be a call or it could be a response from a call.
 * Stream format for a packet:
 *
 *  <header size=8>
 *  <parameters size=header::len>
 *  <blob size=header::blob_len>
 *
 *  RPCHeader structure: tag, id, length, signature
 *   tag: a unique identifier used to match up requests and responses. The
 *   response will have the same tag as the request.
 *   id: an number that identifies which call is being done. Calls and responses
 *   have different IDs.
 *   length: the length of the following serialized call parameters data
 *   signature: always 0xBB
 *   blob_len: 32-bit length of blob payload after serialized parameters
 *  serialized parameters (length = RPCHeader::length) (optional):
 *   The RPCCallBase class defines the layout of these serialized values
 *  blob (length = RPCHeader::blob_len) (optional):
 *
 */

/* template which compiles to a no-op on little endian platform */
template<bool need_swap>
struct EndianSwap
{
	static void swap64(char *data) {}
	static void swap32(char *data) {}
	static void swap16(char *data) {}
	static void swap8(char *data) {}
};

/* byte swapper that works on any platform.
 * C++ standard guarantees that aliasing works properly through char pointer. */
template<>
struct EndianSwap<true>
{
	static void swap64(char *data)
	{
		std::swap(data[0], data[7]);
		std::swap(data[1], data[6]);
		std::swap(data[2], data[5]);
		std::swap(data[3], data[4]);
	}
	static void swap32(char *data)
	{
		std::swap(data[0], data[3]);
		std::swap(data[1], data[2]);
	}
	static void swap16(char *data)
	{
		std::swap(data[0], data[1]);
	}
	static void swap8(char *data)
	{
	}
};

class Endian
{

#ifdef __LITTLE_ENDIAN__
	typedef EndianSwap<false> Swapper;
#else
	typedef EndianSwap<true> Swapper;
#endif

public:
	/* specialize to compiler intrinsics for efficient byteswap on supported compilers */
#if defined(__GNUC__)
	static inline uint64_t swap(uint64_t x) { return __builtin_bswap64(x); }
	static inline uint64_t swap(int64_t x)  { return __builtin_bswap64((uint64_t)x); }
	static inline uint32_t swap(uint32_t x) { return __builtin_bswap32(x); }
	static inline uint32_t swap(int32_t x)  { return __builtin_bswap32((uint32_t)x); }
	static inline uint16_t swap(uint16_t x) { return __builtin_bswap16(x); }
	static inline uint16_t swap(int16_t x)  { return __builtin_bswap16((uint16_t)x); }
	static inline uint8_t swap(uint8_t x)   { return x; }
	static inline uint8_t swap(int8_t x)    { return (uint8_t)x; }
#elif defined(_MSC_VER)
	static inline uint64_t swap(uint64_t x) { return _byteswap_uint64(x); }
	static inline uint64_t swap(int64_t x)  { return _byteswap_uint64((uint64_t)x); }
	static inline uint32_t swap(uint32_t x) { return _byteswap_ulong(x); }
	static inline uint32_t swap(int32_t x)  { return _byteswap_ulong((uint32_t)x); }
	static inline uint16_t swap(uint16_t x) { return _byteswap_ushort(x); }
	static inline uint16_t swap(int16_t x)  { return _byteswap_ushort((uint16_t)x); }
	static inline uint8_t swap(uint8_t x)   { return x; }
	static inline uint8_t swap(int8_t x)    { return (uint8_t)x; }
#else
	static inline uint64_t swap(uint64_t x) { Swapper::swap64((char*)&x); return x; }
	static inline uint64_t swap(int64_t x)  { Swapper::swap64((char*)&x); return x; }
	static inline uint32_t swap(uint32_t x) { Swapper::swap32((char*)&x); return x; }
	static inline uint32_t swap(int32_t x)  { Swapper::swap32((char*)&x); return x; }
	static inline uint16_t swap(uint16_t x) { Swapper::swap16((char*)&x); return x; }
	static inline uint16_t swap(int16_t x)  { Swapper::swap16((char*)&x); return x; }
	static inline uint8_t swap(uint8_t x)   { return x; }
	static inline uint8_t swap(int8_t x)    { return (uint8_t)x; }
#endif

	/* all compilers are stupid about endian conversions with floating point types */
	static inline uint32_t swap(float x)    { Swapper::swap32((char*)&x); return x; }
	static inline uint64_t swap(double x)   { Swapper::swap64((char*)&x); return x; }
};

/* helper needed because we don't use c++11.
 * when we have c++11 we should use 'typename std::make_unsigned<T>::type' */
template<typename T> struct ToUnsigned { typedef T type; };
template<> struct ToUnsigned<int64_t>  { typedef uint64_t type; };
template<> struct ToUnsigned<int32_t>  { typedef uint32_t type; };
template<> struct ToUnsigned<int16_t>  { typedef uint16_t type; };
template<> struct ToUnsigned<int8_t>   { typedef uint8_t type; };
template<> struct ToUnsigned<float>    { typedef uint32_t type; };
template<> struct ToUnsigned<double>   { typedef uint64_t type; };

/* each RPC call has this fixed-size header for variable sized data that follows */
struct RPCHeader
{
	/* each request gets an unused tag value.
	 * the response will have the same tag as the request */
	uint8_t tag;

	/* the id identifies the type of the packet.
	 * requests and responses have different identifiers */
	uint8_t id;

	/* size in bytes of the following payload (not including this header) */
	uint8_t length;

	/* signature to ensure synchronization */
	uint8_t signature;

	/* length of blob, which comes after serialized parameters */
	uint32_t blob_len;
};

template<size_t buffer_max>
class RPCCallBase
{
private:
	void *blob_ptr;
	size_t blob_size;

protected:
	uint8_t call_id;

private:
	/* for performance, a statically allocated buffer is used */
	uint8_t buffer[buffer_max];
	uint8_t add_point;
	uint8_t read_point;

	/* masks for type size prefixes */
	enum SizeFlags {
		size_mask = 0x0F,
		is_unsigned = 0x10,
		is_float = 0x20,
		is_zero = 0x40
	};

protected:
	inline RPCCallBase(uint8_t call_id)
		: blob_ptr(NULL)
		, blob_size(0)
		, call_id(call_id)
		, add_point(0)
		, read_point(0)
	{
	}

	template<typename Tcheck, typename Tactual>
	static inline bool in_range(Tactual a)
	{
		return a >= (std::numeric_limits<Tcheck>::min)()
				&& a <= (std::numeric_limits<Tcheck>::max)();
	}

	/* used on caller to serialize */
	template<typename T>
	void add(T a)
	{
		typename ToUnsigned<T>::type little_endian_a = Endian::swap(a);

		/* make sure there's enough room for a type prefix */
		assert(add_point + 1 <= buffer_max);

		uint8_t type_size = 0;

		if (!std::numeric_limits<T>::is_integer) {
			/* floating point type */
			if (memcmp(&a, "\0\0\0\0\0\0\0\0", sizeof(a)) == 0)
				type_size = 0 | is_zero;
			else if (sizeof(a) == 4)
				type_size = 4 | is_float;
			else
				assert(!"This shouldn't be possible");
		}
		else if (std::numeric_limits<T>::is_signed) {
			/* signed type */
			if (a == 0)
				type_size = 0 | is_zero;
			else if (in_range<int8_t,T>(a))
				type_size = 1;
			else if (in_range<int16_t,T>(a))
				type_size = 2;
			else if (in_range<int32_t,T>(a))
				type_size = 4;
			else if (in_range<int64_t,T>(a))
				type_size = 8;
			else
				assert(!"This shouldn't be possible");
		}
		else {
			/* unsigned type */
			if (a == 0)
				type_size = 0 | is_zero;
			else if (in_range<uint8_t,T>(a))
				type_size = 1 | is_unsigned;
			else if (in_range<uint16_t,T>(a))
				type_size = 2 | is_unsigned;
			else if (in_range<uint32_t,T>(a))
				type_size = 4 | is_unsigned;
			else if (in_range<uint64_t,T>(a))
				type_size = 8 | is_unsigned;
			else
				assert(!"This shouldn't be possible");
		}

		/* make sure there's enough room for a type prefix and data */
		assert(add_point + 1 + (type_size & size_mask) <= buffer_max);

		/* size of type, followed by data */
		buffer[add_point++] = type_size;
		if ((type_size & is_zero) == 0) {
			memcpy(buffer + add_point, &little_endian_a, type_size & size_mask);
			add_point += type_size & size_mask;
		}
	}

	/* overload to avoid problem with shifting float
	 * no code path will call this, it is to make compiler happy */
	static inline float sign_extend(float a, unsigned)
	{
		return a;
	}

	/* sign extend integer type */
	template<typename T>
	static inline T sign_extend(T a, unsigned size)
	{
		int shift = 32 - (size * 8);
		a <<= shift;
		a >>= shift;
		return a;
	}

	/* used by callee to deserialize */
	template<typename T>
	void read(T &result)
	{
		/* make sure we can read type_size
		 * and make sure we can read type length bytes */
		assert(read_point < buffer_max);

		/* get type prefix */
		uint8_t type_size = buffer[read_point];

		/* extract compressed data length */
		unsigned size_bytes = type_size & size_mask;

		/* make sure we can read the specified size */
		assert(read_point + size_bytes < buffer_max);

		/* it's only safe to memset for POD types!
		 * When we can use c++11 we can make sure at compile time */
		memset(&result, 0, sizeof(result));

		typename ToUnsigned<T>::type little_endian_value;

		memcpy(&little_endian_value, buffer + read_point + 1, size_bytes);

		/* sign extend */
		if ((type_size & (is_unsigned|is_zero|is_float)) == 0
				&& (type_size & size_mask) != sizeof(result)) {
			little_endian_value = sign_extend(little_endian_value, size_bytes);
		}

		typename ToUnsigned<T>::type native_endian_value;
		native_endian_value = Endian::swap(little_endian_value);

		/* at this point, any_endian_value is the native endianness */
		memcpy(&result, &native_endian_value, sizeof(result));

		read_point += 1 + size_bytes;
	}

public:
	/* returns true if there is a response */
	virtual bool send_request() = 0;

	/* receives the response payload this request */
	virtual void receive_response() {}
};

/* cycles-specifics are in here */
class CyclesRPCCallBase : public RPCCallBase<80>
{
	typedef RPCCallBase<80> Base;

public:
	static const size_t max_payload = 80;

	enum CallID
	{
		invalid_call,
		mem_alloc_request,
		mem_alloc_response,
		last_CallID
	};

	CallID get_call_id() const
	{
		return (CallID)call_id;
	}

protected:
	CyclesRPCCallBase(CallID call_id)
		: RPCCallBase(call_id)
	{
	}

	/* serialize a device_memory */
	void add(const device_memory& mem)
	{
		int type = (int)mem.data_type;
		Base::add(type);
		Base::add(mem.data_elements);
		Base::add(mem.data_size);
		Base::add(mem.data_width);
		Base::add(mem.data_height);
		Base::add(mem.device_pointer);
	}

	/* serialize a DeviceTask */
	void add(const DeviceTask& task)
	{
		int type = (int)task.type;
		Base::add(type);
		Base::add(task.x);
		Base::add(task.y);
		Base::add(task.w);
		Base::add(task.h);
		Base::add(task.rgba);
		Base::add(task.buffer);
		Base::add(task.sample);
		Base::add(task.num_samples);
		Base::add(task.offset);
		Base::add(task.stride);
		Base::add(task.shader_input);
		Base::add(task.shader_output);
		Base::add(task.shader_eval_type);
		Base::add(task.shader_x);
		Base::add(task.shader_w);
		Base::add(task.need_finish_queue);
	}

	/* serialize a RenderTile */
	void add(const RenderTile& tile)
	{
		Base::add(tile.x);
		Base::add(tile.y);
		Base::add(tile.w);
		Base::add(tile.h);
		Base::add(tile.start_sample);
		Base::add(tile.num_samples);
		Base::add(tile.sample);
		Base::add(tile.resolution);
		Base::add(tile.offset);
		Base::add(tile.stride);
		Base::add(tile.buffer);
		Base::add(tile.rng_state);
		Base::add(tile.rgba);
	}

	template<typename T>
	void add(const T &a)
	{
		Base::add(a);
	}
};

class RPCCall_mem_alloc : public CyclesRPCCallBase
{
private:
	device_memory &mem;
	MemoryType type;

	bool send_request()
	{
		int inttype = (int)type;
		add(mem);
		add(inttype);
		return false;
	}

public:
	RPCCall_mem_alloc(device_memory& mem, MemoryType type)
		: CyclesRPCCallBase(mem_alloc_request)
		, mem(mem), type(type)
	{}
};

/* RPC stream manager
 *  on the server side:
 *   - it provides a way for the main thread to wait for and return
 *	   incoming RPC requests, and later send back responses for those
 *     requests.
 *   - it provides a way for arbitrary server threads to make calls
 *     back to the client. If the call returns data, it blocks until
 *     the response is received, receives the response, and returns
 *     the result
 *  on the client side:
 *   - it provides a way for the main thread to send calls to the
 *     server. If the call returns data, it blocks until the response
 *     is received, receives the response, and returns the result
 *   - it receives calls from the server, invokes them, and if they
 *     return a result, sends the response
 */
class RPCDispatcher
{
	boost::asio::io_service io_service;
	boost::asio::ip::tcp::socket socket;
	thread_mutex send_lock;

	/* receive stream management data */

	enum ReceiveState
	{
		receiving_header,
		receiving_args,
		receiving_blob,
		receive_aborted
	};

	ReceiveState recv_state;

	RPCHeader recv_header;
	std::vector<char> recv_args_buffer;
	std::vector<char> recv_blob_buffer;

	/* object upon which to block when waiting for results */

	class Waiter
	{
		/* mutex can't be copied, so use a pointer and enforce
		 * that this object is never copied after it is created */
		thread_mutex *done_lock;
		thread_condition_variable *done_cond;

		CyclesRPCCallBase::CallID call_id;
		bool done;

		void assert_uninitalized()
		{
			assert(done_lock == NULL);
			assert(done_cond == NULL);
		}

		void assert_initalized()
		{
			assert(done_lock == NULL);
			assert(done_cond == NULL);
		}

	public:
		Waiter()
			: done_lock(NULL)
			, done_cond(NULL)
			, call_id(CyclesRPCCallBase::invalid_call)
			, done(false)
		{
		}

		Waiter(CyclesRPCCallBase::CallID call_id)
			: done_lock(NULL)
			, done_cond(NULL)
			, call_id(call_id)
			, done(false)
		{
		}

		Waiter(const Waiter& rhs)
		{
			assert_uninitalized();
			call_id = rhs.call_id;
			done = rhs.done;
		}

		~Waiter()
		{
			delete done_lock;
			delete done_cond;
		}

		void init()
		{
			assert_uninitalized();
			done_lock = new thread_mutex;
			done_cond = new thread_condition_variable;
		}

		void wait()
		{
			assert_initalized();
			thread_scoped_lock lock(*done_lock);
			while (!done)
				done_cond->wait(lock);
		}

		void notify_done()
		{
			assert_initalized();
			thread_scoped_lock lock(*done_lock);
			done_cond->notify_one();
		}
	};

	/* we don't want to be constantly creating and destroying
	 * mutices and condition variables, so pool them */
	class WaiterPool
	{
		thread_mutex waiter_pool_lock;

		/* using deque so growing it won't cause any of the items to move
		 * to another address, and also maximize data locality */
		typedef std::deque<Waiter> WaiterPoolStorage;
		WaiterPoolStorage waiter_pool_storage;

		/* pointers to unused waiters get are stored in here */
		typedef std::vector<Waiter*> WaiterPoolFreeList;
		WaiterPoolFreeList waiter_pool_free_list;

	public:

		Waiter *alloc_waiter()
		{
			Waiter *result;
			thread_scoped_lock lock(waiter_pool_lock);
			if (waiter_pool_free_list.empty()) {
				/* use waiter in free list */
				result = waiter_pool_free_list.back();
				waiter_pool_free_list.pop_back();
			}
			else {
				/* need to create a new Waiter */
				waiter_pool_storage.push_back(Waiter());
				result = &waiter_pool_storage.back();
			}
			return result;
		}

		void free_waiter(Waiter *waiter)
		{
			thread_scoped_lock lock(waiter_pool_lock);
			waiter_pool_free_list.push_back(waiter);
		}
	};

	WaiterPool waiter_pool;

	/* send implementation */

	void register_for_unblock(CyclesRPCCallBase::CallID id)
	{
	}

	/* send something from any thread */
	void send_next_item(CyclesRPCCallBase &item)
	{
		/* if we need to block this thread until response comes back,
		 * we need to register for unblock before sending */

		if (item.send_request()) {
			register_for_unblock(item.get_call_id());
		}

		boost::system::error_code send_err;

//		RPCHeader header = item.make_call_header();

//		boost::asio::write(socket,
//				boost::asio::buffer(item.header, sizeof(*item.header)),
//				send_err);
//		boost::asio::write(socket,
//				boost::asio::buffer(item.header, sizeof(*item.header)),
//				send_err);
//		boost::asio::write(socket,
//				boost::asio::buffer(item.header, sizeof(*item.header)),
//				send_err);
	}

	/* receive stream implementation */

	void post_async_recv_header()
	{
		recv_state = receiving_header;

		boost::asio::async_read(socket,
				boost::asio::buffer(&recv_header, sizeof(recv_header)),
				boost::bind(&RPCDispatcher::handle_recv_header, this,
				boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}

	void handle_recv_header(const boost::system::error_code& error, size_t size)
	{
		if (recv_header.length) {
			post_async_recv_args();
		}
	}

	void post_async_recv_args()
	{
		recv_state = receiving_args;
		boost::asio::async_read(socket,
				boost::asio::buffer(&recv_args_buffer[0], recv_header.length),
				boost::bind(&RPCDispatcher::handle_recv_args, this,
				boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}

	void handle_recv_args(const boost::system::error_code& error, size_t size)
	{
		if (recv_header.blob_len) {
			post_async_recv_blob();
		}
	}

	void post_async_recv_blob()
	{
		recv_state = receiving_blob;
		boost::asio::async_read(socket,
				boost::asio::buffer(&recv_blob_buffer[0], recv_header.blob_len),
				boost::bind(&RPCDispatcher::handle_recv_blob, this,
				boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}

	void handle_recv_blob(const boost::system::error_code& error, size_t size)
	{
		recv_state = receiving_header;

		deliver_recv();

		post_async_recv_header();
	}

	void deliver_recv()
	{
	}

	boost::system::error_code connect_impl(const std::string &address)
	{
		boost::system::error_code err;
		stringstream portstr;

		/* use specified port if address with host:port format was passed */
		std::string::const_iterator colon = std::find(address.begin(), address.end(), ':');
		if (colon != address.end())
			portstr.str(std::string(colon + 1, address.end()));
		else
			portstr << SERVER_PORT;

		tcp::resolver resolver(io_service);
		tcp::resolver::query query(address, portstr.str());

		/* try all of the addresses the resolver found */
		for (tcp::resolver::iterator e, i = resolver.resolve(query, err); i != e; ++i) {
			/* if resolver encountered an error, return it */
			if (err)
				return err;

			/* try to connect to the address */
			socket.connect(*i, err);

			/* if it succeeded, stop trying addresses */
			if (!err)
				break;

			/* close the failed socket */
			socket.close();
		}

		return err;
	}

public:
	/* constructor called when operating as a client that makes
	 * an outbound connection to a server */
	RPCDispatcher(const std::string &address)
		: socket(io_service)
		, recv_args_buffer(CyclesRPCCallBase::max_payload, 0)
	{
	}

	std::string connect_to_server(const std::string &address)
	{
		std::string err;
		boost::system::error_code error_code = connect_impl(address);
		err = error_code.message();
		return err;
	}
};

/* RAM copy of device memory on server side. On some devices, this is the
 * actual data storage and the device doesn't have a copy, it uses it directly */

class network_device_memory : public device_memory
{
public:
	network_device_memory() {}
	~network_device_memory() { device_pointer = 0; }

	vector<char> local_data;
};

/* Remote procedure call Send */

class RPCSend {
public:
	RPCSend(tcp::socket& socket_, const string& name_ = "")
	: name(name_), socket(socket_), archive(archive_stream), sent(false)
	{
		SyncOutputStream() << "Constructing RPC send: " << name;
		archive & name_;
	}

	~RPCSend()
	{
		if(!sent)
			SyncOutputStream() << "Error: RPC " << name << " not sent";
	}

	void add(const device_memory& mem)
	{
		archive & mem.data_type & mem.data_elements & mem.data_size;
		archive & mem.data_width & mem.data_height & mem.device_pointer;
	}

	template<typename T> void add(const T& data)
	{
		archive & data;
	}

	void add(const DeviceTask& task)
	{
		int type = (int)task.type;

		archive & type & task.x & task.y & task.w & task.h;
		archive & task.rgba & task.buffer & task.sample & task.num_samples;
		archive & task.offset & task.stride;
		archive & task.shader_input & task.shader_output & task.shader_eval_type;
		archive & task.shader_x & task.shader_w;
		archive & task.need_finish_queue;
	}

	void add(const RenderTile& tile)
	{
		archive & tile.x & tile.y & tile.w & tile.h;
		archive & tile.start_sample & tile.num_samples & tile.sample;
		archive & tile.resolution & tile.offset & tile.stride;
		archive & tile.buffer & tile.rng_state & tile.rgba;
	}

	void write()
	{
		boost::system::error_code error;

		/* get string from stream */
		string archive_str = archive_stream.str();

		/* first send fixed size header with size of following data */
		ostringstream header_stream;
		header_stream << setw(8) << hex << archive_str.size();
		string header_str = header_stream.str();

		SyncOutputStream() << "Sending output header, len=" << header_str.length();

		boost::asio::write(socket,
			boost::asio::buffer(header_str),
			boost::asio::transfer_all(), error);

		if(error.value())
			SyncOutputStream() << "Network send error: " << error.message();
		else
			SyncOutputStream() << "Sending output header done";

		SyncOutputStream() << "Writing output data, len=" << archive_str.length();

		/* then send actual data */
		boost::asio::write(socket,
			boost::asio::buffer(archive_str),
			boost::asio::transfer_all(), error);

		if(error.value())
			SyncOutputStream() << "Network send error: " << error.message();
		else
			SyncOutputStream() << "Writing output data done" << archive_str.length();

		sent = true;
	}

	void write_buffer(void *buffer, size_t size)
	{
		SyncOutputStream() << "Writing BLOB, size=" << size;

		boost::system::error_code error;

		boost::asio::write(socket,
			boost::asio::buffer(buffer, size),
			boost::asio::transfer_all(), error);

		if(error.value())
			SyncOutputStream() << "Network send error: " << error.message();
	}

protected:
	string name;
	tcp::socket& socket;
	ostringstream archive_stream;
	boost::archive::text_oarchive archive;
	bool sent;
};

/* Remote procedure call Receive */

class RPCReceive {
public:
	RPCReceive(tcp::socket& socket_)
	: socket(socket_), archive_stream(NULL), archive(NULL)
	{
		SyncOutputStream() << "Reading input header";

		/* read head with fixed size */
		vector<char> header(8);
		size_t len = boost::asio::read(socket, boost::asio::buffer(header));

		static int race_detect;
		if (__sync_add_and_fetch(&race_detect, 1) != 1) {
			raise(SIGTRAP);
		}

		SyncOutputStream() << "Input header length=" << len;

		/* verify if we got something */
		if(len == header.size()) {
			/* decode header */
			string header_str(&header[0], header.size());
			istringstream header_stream(header_str);

			size_t data_size;

			if((header_stream >> hex >> data_size)) {
				SyncOutputStream() << "Reading data, size=" << data_size;

				vector<char> data(data_size);
				size_t len = boost::asio::read(socket, boost::asio::buffer(data));

				if(len == data_size) {
					archive_str = (data.size())? string(&data[0], data.size()): string("");
#if 0
					istringstream archive_stream(archive_str);
					boost::archive::text_iarchive archive(archive_stream);
#endif
					archive_stream = new istringstream(archive_str);
					archive = new boost::archive::text_iarchive(*archive_stream);

					*archive & name;

					SyncOutputStream() << "Got RPCReceive op: " << name;
				}
				else {
					SyncOutputStream() << "Network receive error: data size doesn't match header";
					raise(SIGTRAP);
				}
			}
			else {
				SyncOutputStream() << "Network receive error: can't decode data size from header";
				raise(SIGTRAP);
			}
		}
		else {
			SyncOutputStream() << "Network receive error: invalid header size";
			raise(SIGTRAP);
		}

		if (__sync_add_and_fetch(&race_detect, -1) != 0) {
			raise(SIGTRAP);
		}
	}

	~RPCReceive()
	{
		delete archive;
		delete archive_stream;
	}

	void read(network_device_memory& mem)
	{
		*archive & mem.data_type & mem.data_elements & mem.data_size;
		*archive & mem.data_width & mem.data_height & mem.device_pointer;

		mem.data_pointer = 0;
	}

	template<typename T> void read(T& data)
	{
		*archive & data;
	}

	void read_buffer(void *buffer, size_t size)
	{
		size_t len = boost::asio::read(socket, boost::asio::buffer(buffer, size));

		if(len != size)
			SyncOutputStream() << "Network receive error: buffer size doesn't match expected size";
	}

	void read(DeviceTask& task)
	{
		int type;

		*archive & type & task.x & task.y & task.w & task.h;
		*archive & task.rgba & task.buffer & task.sample & task.num_samples;
		*archive & task.offset & task.stride;
		*archive & task.shader_input & task.shader_output & task.shader_eval_type;
		*archive & task.shader_x & task.shader_w;
		*archive & task.need_finish_queue;

		task.type = (DeviceTask::Type)type;
	}

	void read(RenderTile& tile)
	{
		*archive & tile.x & tile.y & tile.w & tile.h;
		*archive & tile.start_sample & tile.num_samples & tile.sample;
		*archive & tile.resolution & tile.offset & tile.stride;
		*archive & tile.buffer & tile.rng_state & tile.rgba;

		tile.buffers = NULL;
	}

	string name;

protected:
	tcp::socket& socket;
	string archive_str;
	istringstream *archive_stream;
	boost::archive::text_iarchive *archive;
};

/* Server auto discovery */

class ServerDiscovery {
public:
	ServerDiscovery(bool discover = false)
	: listen_socket(io_service), collect_servers(false)
	{
		/* setup listen socket */
		listen_endpoint.address(boost::asio::ip::address_v4::any());
		listen_endpoint.port(DISCOVER_PORT);

		listen_socket.open(listen_endpoint.protocol());

		boost::asio::socket_base::reuse_address option(true);
		listen_socket.set_option(option);

		listen_socket.bind(listen_endpoint);

		/* setup receive callback */
		async_receive();

		/* start server discovery */
		if(discover) {
			collect_servers = true;
			servers.clear();

			broadcast_message(DISCOVER_REQUEST_MSG);
		}

		/* start thread */
		work = new boost::asio::io_service::work(io_service);
		thread = new boost::thread(boost::bind(&boost::asio::io_service::run, &io_service));
	}

	~ServerDiscovery()
	{
		io_service.stop();
		thread->join();
		delete thread;
		delete work;
	}

	vector<string> get_server_list()
	{
		vector<string> result;

		mutex.lock();
		result = vector<string>(servers.begin(), servers.end());
		mutex.unlock();

		return result;
	}

private:
	void handle_receive_from(const boost::system::error_code& error, size_t size)
	{
		if(error) {
			SyncOutputStream() << "Server discovery receive error: " << error.message();
			return;
		}

		if(size > 0) {
			string msg = string(receive_buffer, size);

			/* handle incoming message */
			if(collect_servers) {
				if(msg == DISCOVER_REPLY_MSG) {
					string address = receive_endpoint.address().to_string();

					mutex.lock();

					/* add address if it's not already in the list */
					bool found = std::find(servers.begin(), servers.end(),
							address) != servers.end();

					if(!found)
						servers.push_back(address);

					mutex.unlock();
				}
			}
			else {
				/* reply to request */
				if(msg == DISCOVER_REQUEST_MSG)
					broadcast_message(DISCOVER_REPLY_MSG);
			}
		}

		async_receive();
	}

	void async_receive()
	{
		listen_socket.async_receive_from(
			boost::asio::buffer(receive_buffer), receive_endpoint,
			boost::bind(&ServerDiscovery::handle_receive_from, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}

	void broadcast_message(const string& msg)
	{
		/* setup broadcast socket */
		boost::asio::ip::udp::socket socket(io_service);

		socket.open(boost::asio::ip::udp::v4());

		boost::asio::socket_base::broadcast option(true);
		socket.set_option(option);

		boost::asio::ip::udp::endpoint broadcast_endpoint(
			boost::asio::ip::address::from_string("255.255.255.255"), DISCOVER_PORT);

		/* broadcast message */
		socket.send_to(boost::asio::buffer(msg), broadcast_endpoint);
	}

	/* network service and socket */
	boost::asio::io_service io_service;
	boost::asio::ip::udp::endpoint listen_endpoint;
	boost::asio::ip::udp::socket listen_socket;

	/* threading */
	boost::thread *thread;
	boost::asio::io_service::work *work;
	boost::mutex mutex;

	/* buffer and endpoint for receiving messages */
	char receive_buffer[256];
	boost::asio::ip::udp::endpoint receive_endpoint;

	// os, version, devices, status, host name, group name, ip as far as fields go
	struct ServerInfo {
		string blender_version;
		string os;
		int device_count;
		string status;
		string host_name;
		string group_name;
		string host_addr;
	};

	/* collection of server addresses in list */
	bool collect_servers;
	vector<string> servers;
};

CCL_NAMESPACE_END

#endif

#endif /* __DEVICE_NETWORK_H__ */

