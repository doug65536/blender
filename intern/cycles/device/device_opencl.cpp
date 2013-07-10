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

#ifdef WITH_OPENCL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include <set>

#include "device.h"
#include "device_intern.h"

#include "buffers.h"

#include "util_foreach.h"
#include "util_map.h"
#include "util_math.h"
#include "util_md5.h"
#include "util_opencl.h"
#include "util_opengl.h"
#include "util_path.h"
#include "util_time.h"

CCL_NAMESPACE_BEGIN

#define CL_MEM_PTR(p) ((cl_mem)(uintptr_t)(p))

static cl_device_type opencl_device_type()
{
	char *device = getenv("CYCLES_OPENCL_TEST");

	if(device) {
		if(strcmp(device, "ALL") == 0)
			return CL_DEVICE_TYPE_ALL;
		else if(strcmp(device, "DEFAULT") == 0)
			return CL_DEVICE_TYPE_DEFAULT;
		else if(strcmp(device, "CPU") == 0)
			return CL_DEVICE_TYPE_CPU;
		else if(strcmp(device, "GPU") == 0)
			return CL_DEVICE_TYPE_GPU;
		else if(strcmp(device, "ACCELERATOR") == 0)
			return CL_DEVICE_TYPE_ACCELERATOR;
	}

	return CL_DEVICE_TYPE_ALL;
}

static bool opencl_kernel_use_debug()
{
	return (getenv("CYCLES_OPENCL_DEBUG") != NULL);
}

static bool opencl_kernel_use_advanced_shading(const string& platform)
{
	/* keep this in sync with kernel_types.h! */
	if(platform == "NVIDIA CUDA")
		return true;
	else if(platform == "Apple")
		return false;
	else if(platform == "AMD Accelerated Parallel Processing")
		return false;
	else if(platform == "Intel(R) OpenCL")
		return true;

	return false;
}

static string opencl_kernel_build_options(const string& platform, const string *debug_src = NULL)
{
	string build_options = " -cl-fast-relaxed-math ";

	if(platform == "NVIDIA CUDA")
		build_options += "-D __KERNEL_OPENCL_NVIDIA__ -cl-nv-maxrregcount=32 -cl-nv-verbose ";

	else if(platform == "Apple")
		build_options += "-D __KERNEL_OPENCL_APPLE__ -Wno-missing-prototypes ";

	else if(platform == "AMD Accelerated Parallel Processing")
		build_options += "-D __KERNEL_OPENCL_AMD__ ";

	else if(platform == "Intel(R) OpenCL") {
		build_options += "-D __KERNEL_OPENCL_INTEL_CPU__";

		/* options for gdb source level kernel debugging. this segfaults on linux currently */
		if(opencl_kernel_use_debug() && debug_src)
			build_options += "-g -s \"" + *debug_src + "\" ";
	}

	if(opencl_kernel_use_debug())
		build_options += "-D __KERNEL_OPENCL_DEBUG__ ";

	if(opencl_kernel_use_advanced_shading(platform))
		build_options += "-D __KERNEL_OPENCL_NEED_ADVANCED_SHADING__ ";

	return build_options;
}

static std::set<string> get_device_extensions(cl_device_id device, cl_int *ret_err)
{
	cl_int ciErr;
	std::set<string> extensions;

	/* always initialize error output value */
	if (ret_err)
		*ret_err = CL_SUCCESS;

	size_t extension_string_len = 0;
	ciErr = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &extension_string_len);
	if (ciErr != CL_SUCCESS) {
		if (ret_err)
			*ret_err = ciErr;
		/* return empty set */
		return extensions;
	}

	if (!extension_string_len)
		return extensions;

	vector<char> extension_str;
	extension_str.resize(extension_string_len);

	ciErr = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS,
			extension_string_len, &extension_str[0], NULL);
	if (ciErr != CL_SUCCESS) {
		if (ret_err)
			*ret_err = ciErr;
		/* return empty set */
		return extensions;
	}

	vector<char>::iterator st, en, i;
	st = extension_str.begin();
	en = extension_str.end();
	i = st;
	while (st != en) {
		/* try to insert string and/or skip whitespace if
		 * we have something and, we're at the end
		 * or we've hit a space */
		if (i != st && (i == en || isspace(*i))) {
			/* insert a string if we have characters
			 * and it's not a null terminator */
			if (st != i && *st)
				extensions.insert(std::set<string>::value_type(string(st, i)));

			/* eat extra spaces or null terminators between strings */
			while (i != en && (!*i || isspace(*++i)));

			/* found beginning of another string (or end) */
			st = i;
		}
		else {
			++i;
		}
	}

	return extensions;
}

/* thread safe cache for contexts and programs */
class OpenCLCache
{
	struct Slot
	{
		thread_mutex *mutex;
		cl_context context;
		cl_program program;

		Slot() : mutex(NULL), context(NULL), program(NULL) {}

		Slot(const Slot &rhs)
			: mutex(rhs.mutex)
			, context(rhs.context)
			, program(rhs.program)
		{
			/* copy can only happen in map insert, assert that */
			assert(mutex == NULL);
		}

		~Slot()
		{
			delete mutex;
			mutex = NULL;
		}
	};

	/* key is combination of platform ID and device ID */
	typedef pair<cl_platform_id, cl_device_id> PlatformDevicePair;

	/* map of Slot objects */
	typedef map<PlatformDevicePair, Slot> CacheMap;
	CacheMap cache;

	thread_mutex cache_lock;

	/* lazy instantiate */
	static OpenCLCache &global_instance()
	{
		static OpenCLCache instance;
		return instance;
	}

	OpenCLCache()
	{
	}

	~OpenCLCache()
	{
		/* Intel OpenCL bug raises SIGABRT due to pure virtual call
		 * so this is disabled. It's not necessary to free objects
		 * at process exit anyway.
		 * http://software.intel.com/en-us/forums/topic/370083#comments */

		//flush();
	}

	/* lookup something in the cache. If this returns NULL, slot_locker
	 * will be holding a lock for the cache. slot_locker should refer to a
	 * default constructed thread_scoped_lock */
	template<typename T>
	static T get_something(cl_platform_id platform, cl_device_id device,
		T Slot::*member, thread_scoped_lock &slot_locker)
	{
		assert(platform != NULL);

		OpenCLCache &self = global_instance();

		thread_scoped_lock cache_lock(self.cache_lock);

		pair<CacheMap::iterator,bool> ins = self.cache.insert(
			CacheMap::value_type(PlatformDevicePair(platform, device), Slot()));

		Slot &slot = ins.first->second;

		/* create slot lock only while holding cache lock */
		if(!slot.mutex)
			slot.mutex = new thread_mutex;

		/* need to unlock cache before locking slot, to allow store to complete */
		cache_lock.unlock();

		/* lock the slot */
		slot_locker = thread_scoped_lock(*slot.mutex);

		/* If the thing isn't cached */
		if(slot.*member == NULL) {
			/* return with the caller's lock holder holding the slot lock */
			return NULL;
		}

		/* the item was already cached, release the slot lock */
		slot_locker.unlock();

		return slot.*member;
	}

	/* store something in the cache. you MUST have tried to get the item before storing to it */
	template<typename T>
	static void store_something(cl_platform_id platform, cl_device_id device, T thing,
		T Slot::*member, thread_scoped_lock &slot_locker)
	{
		assert(platform != NULL);
		assert(device != NULL);
		assert(thing != NULL);

		OpenCLCache &self = global_instance();

		thread_scoped_lock cache_lock(self.cache_lock);
		CacheMap::iterator i = self.cache.find(PlatformDevicePair(platform, device));
		cache_lock.unlock();

		Slot &slot = i->second;

		/* sanity check */
		assert(i != self.cache.end());
		assert(slot.*member == NULL);

		slot.*member = thing;

		/* unlock the slot */
		slot_locker.unlock();
	}

public:
	/* see get_something comment */
	static cl_context get_context(cl_platform_id platform, cl_device_id device,
		thread_scoped_lock &slot_locker)
	{
		cl_context context = get_something<cl_context>(platform, device, &Slot::context, slot_locker);

		if(!context)
			return NULL;

		/* caller is going to release it when done with it, so retain it */
		cl_int ciErr = clRetainContext(context);
		assert(ciErr == CL_SUCCESS);
		(void)ciErr;

		return context;
	}

	/* see get_something comment */
	static cl_program get_program(cl_platform_id platform, cl_device_id device,
		thread_scoped_lock &slot_locker)
	{
		cl_program program = get_something<cl_program>(platform, device, &Slot::program, slot_locker);

		if(!program)
			return NULL;

		/* caller is going to release it when done with it, so retain it */
		cl_int ciErr = clRetainProgram(program);
		assert(ciErr == CL_SUCCESS);
		(void)ciErr;

		return program;
	}

	/* see store_something comment */
	static void store_context(cl_platform_id platform, cl_device_id device, cl_context context,
		thread_scoped_lock &slot_locker)
	{
		store_something<cl_context>(platform, device, context, &Slot::context, slot_locker);

		/* increment reference count in OpenCL.
		 * The caller is going to release the object when done with it. */
		cl_int ciErr = clRetainContext(context);
		assert(ciErr == CL_SUCCESS);
		(void)ciErr;
	}

	/* see store_something comment */
	static void store_program(cl_platform_id platform, cl_device_id device, cl_program program,
		thread_scoped_lock &slot_locker)
	{
		store_something<cl_program>(platform, device, program, &Slot::program, slot_locker);

		/* increment reference count in OpenCL.
		 * The caller is going to release the object when done with it. */
		cl_int ciErr = clRetainProgram(program);
		assert(ciErr == CL_SUCCESS);
		(void)ciErr;
	}

	/* discard all cached contexts and programs
	 * the parameter is a temporary workaround. See OpenCLCache::~OpenCLCache */
	static void flush()
	{
		OpenCLCache &self = global_instance();
		thread_scoped_lock cache_lock(self.cache_lock);

		foreach(CacheMap::value_type &item, self.cache) {
			if(item.second.program != NULL)
				clReleaseProgram(item.second.program);
			if(item.second.context != NULL)
				clReleaseContext(item.second.context);
		}

		self.cache.clear();
	}
};

/* http://www.khronos.org/registry/cl/extensions/ext/cl_ext_device_fission.txt */
class DeviceFissionExt
{
public:
	typedef cl_bitfield cl_device_partition_property_ext;

	typedef cl_int (*clCreateSubDevicesEXT_function)(
			cl_device_id in_device,
			const cl_device_partition_property_ext * properties,
			cl_uint num_entries,
			cl_device_id *out_devices,
			cl_uint *num_devices);

	typedef cl_int (*clReleaseDeviceEXT_function)(cl_device_id device);

	typedef cl_int (*clRetainDeviceEXT_function)(cl_device_id device);

	clCreateSubDevicesEXT_function clCreateSubDevicesEXT;
	clReleaseDeviceEXT_function clReleaseDeviceEXT;
	clRetainDeviceEXT_function clRetainDeviceEXT;

	enum {
		CL_DEVICE_PARTITION_EQUALLY_EXT            = 0x4050,
		CL_DEVICE_PARTITION_BY_COUNTS_EXT          = 0x4051,
		CL_DEVICE_PARTITION_BY_NAMES_EXT           = 0x4052,
		CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT = 0x4053,

		CL_AFFINITY_DOMAIN_L1_CACHE_EXT         = 0x1,
		CL_AFFINITY_DOMAIN_L2_CACHE_EXT         = 0x2,
		CL_AFFINITY_DOMAIN_L3_CACHE_EXT         = 0x3,
		CL_AFFINITY_DOMAIN_L4_CACHE_EXT         = 0x4,
		CL_AFFINITY_DOMAIN_NUMA_EXT             = 0x10,
		CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT = 0x100,

		CL_DEVICE_PARENT_DEVICE_EXT             = 0x4054,
		CL_DEVICE_PARITION_TYPES_EXT            = 0x4055,
		CL_DEVICE_AFFINITY_DOMAINS_EXT          = 0x4056,
		CL_DEVICE_REFERENCE_COUNT_EXT		    = 0x4057,
		CL_DEVICE_PARTITION_STYLE_EXT		    = 0x4058,

		CL_PROPERTIES_LIST_END_EXT              = 0x0,
		CL_PARTITION_BY_COUNTS_LIST_END_EXT     = 0x0,
		CL_PARTITION_BY_NAMES_LIST_END_EXT      = -1,
		CL_DEVICE_PARTITION_FAILED_EXT          = -1057,
		CL_INVALID_PARTITION_COUNT_EXT          = -1058,
		CL_INVALID_PARTITION_NAME_EXT           = -1059
	};

	DeviceFissionExt()
		: clCreateSubDevicesEXT(NULL)
		, clReleaseDeviceEXT(NULL)
		, clRetainDeviceEXT(NULL)
	{
	}

	bool initialize()
	{
		if (!clCreateSubDevicesEXT) {
			clCreateSubDevicesEXT = (clCreateSubDevicesEXT_function)
					clGetExtensionFunctionAddress("clCreateSubDevicesEXT");
			clReleaseDeviceEXT = (clReleaseDeviceEXT_function)
					clGetExtensionFunctionAddress("clReleaseDeviceEXT");
			clRetainDeviceEXT = (clRetainDeviceEXT_function)
					clGetExtensionFunctionAddress("clRetainDeviceEXT");
		}

		return clCreateSubDevicesEXT != NULL;
	}
};

class OpenCLDevice : public Device
{
public:
	TaskPool task_pool;
	cl_context cxContext;
	cl_command_queue cqCommandQueue;
	cl_platform_id cpPlatform;
	cl_device_id cdDevice;
	cl_program cpProgram;
	cl_kernel ckPathTraceKernel;
	cl_kernel ckFilmConvertKernel;
	cl_kernel ckShaderKernel;
	cl_int ciErr;

	/* set to true if device can directly use CPU memory */
	cl_bool use_unified_memory;

	typedef map<string, device_vector<uchar>*> ConstMemMap;
	typedef map<string, device_ptr> MemMap;

	ConstMemMap const_mem_map;
	MemMap mem_map;
	device_ptr null_mem;

	/* device lock only needed and used when device fission is being used */
	thread_mutex device_lock;

	bool device_initialized;
	string platform_name;

	std::set<string> extensions;

	DeviceFissionExt fission_ext;

	const char *opencl_error_string(cl_int err)
	{
		switch (err) {
			case CL_SUCCESS: return "Success!";
			case CL_DEVICE_NOT_FOUND: return "Device not found.";
			case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
			case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
			case CL_OUT_OF_RESOURCES: return "Out of resources";
			case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
			case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information not available";
			case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
			case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
			case CL_BUILD_PROGRAM_FAILURE: return "Program build failure";
			case CL_MAP_FAILURE: return "Map failure";
			case CL_INVALID_VALUE: return "Invalid value";
			case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
			case CL_INVALID_PLATFORM: return "Invalid platform";
			case CL_INVALID_DEVICE: return "Invalid device";
			case CL_INVALID_CONTEXT: return "Invalid context";
			case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
			case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
			case CL_INVALID_HOST_PTR: return "Invalid host pointer";
			case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
			case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
			case CL_INVALID_SAMPLER: return "Invalid sampler";
			case CL_INVALID_BINARY: return "Invalid binary";
			case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
			case CL_INVALID_PROGRAM: return "Invalid program";
			case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
			case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
			case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
			case CL_INVALID_KERNEL: return "Invalid kernel";
			case CL_INVALID_ARG_INDEX: return "Invalid argument index";
			case CL_INVALID_ARG_VALUE: return "Invalid argument value";
			case CL_INVALID_ARG_SIZE: return "Invalid argument size";
			case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
			case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
			case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
			case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
			case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
			case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
			case CL_INVALID_EVENT: return "Invalid event";
			case CL_INVALID_OPERATION: return "Invalid operation";
			case CL_INVALID_GL_OBJECT: return "Invalid OpenGL object";
			case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
			case CL_INVALID_MIP_LEVEL: return "Invalid mip-map level";
			default: return "Unknown";
		}
	}

	bool opencl_error(cl_int err)
	{
		if(err != CL_SUCCESS) {
			string message = string_printf("OpenCL error (%d): %s", err, opencl_error_string(err));
			if(error_msg == "")
				error_msg = message;
			fprintf(stderr, "%s\n", message.c_str());
			fflush(stderr);
			return true;
		}

		return false;
	}

	void opencl_error(const string& message)
	{
		if(error_msg.empty())
			error_msg = message;
		fprintf(stderr, "%s\n", message.c_str());
		fflush(stderr);
	}

	void opencl_assert(cl_int err)
	{
		if(err != CL_SUCCESS) {
			string message = string_printf("OpenCL error (%d): %s", err, opencl_error_string(err));
			if(error_msg == "")
				error_msg = message;
			fprintf(stderr, "%s\n", message.c_str());
			fflush(stderr);
#ifndef NDEBUG
			raise(SIGTRAP);
#endif
		}
	}

	static Device *create_fission_device(DeviceInfo& info, Stats& stats, bool background)
	{
		/* create a device from which to spawn fission devices */
		OpenCLDevice *temp_device;
		temp_device = new OpenCLDevice(info, stats, background);

		std::vector<cl_device_id> fission_devices;
		temp_device->get_fission_devices(fission_devices);

		/* explicitly create device objects */
		std::vector<Device*> subdevices;

		delete temp_device;

		//return device_multi_create_from(fission_devices);
	}

	/* use fission api to break this device into as many devices as possible */
	cl_int get_fission_devices(std::vector<cl_device_id>& devices_out)
	{
		if (!fission_ext.initialize())
			return CL_SUCCESS;

		DeviceFissionExt::cl_device_partition_property_ext props[] = {
			DeviceFissionExt::CL_DEVICE_PARTITION_EQUALLY_EXT, 1,
			DeviceFissionExt::CL_PROPERTIES_LIST_END_EXT
		};

		cl_uint subdevice_count;

		/* get subdevice count */
		ciErr = fission_ext.clCreateSubDevicesEXT(cdDevice,
			props, 0, NULL, &subdevice_count);
		if (opencl_error(ciErr))
			return ciErr;

		devices_out.resize(subdevice_count);

		/* create subdevices */
		ciErr = fission_ext.clCreateSubDevicesEXT(cdDevice,
			props, 0, &devices_out[0], &subdevice_count);
		if (opencl_error(ciErr))
			return ciErr;

		return CL_SUCCESS;
	}

	OpenCLDevice(DeviceInfo& info, Stats &stats, bool background_)
	  : Device(stats)
	{
		background = background_;
		cpPlatform = NULL;
		cdDevice = NULL;
		cxContext = NULL;
		cqCommandQueue = NULL;
		cpProgram = NULL;
		ckPathTraceKernel = NULL;
		ckFilmConvertKernel = NULL;
		ckShaderKernel = NULL;
		null_mem = 0;
		device_initialized = false;

		/* setup platform */
		cl_uint num_platforms;

		ciErr = clGetPlatformIDs(0, NULL, &num_platforms);
		if(opencl_error(ciErr))
			return;

		if(num_platforms == 0) {
			opencl_error("OpenCL: no platforms found.");
			return;
		}

		vector<cl_platform_id> platforms(num_platforms, NULL);

		ciErr = clGetPlatformIDs(num_platforms, &platforms[0], NULL);
		if(opencl_error(ciErr))
			return;

		int num_base = 0;
		int total_devices = 0;

		for (int platform = 0; platform < num_platforms; platform++) {
			cl_uint num_devices;

			if(opencl_error(clGetDeviceIDs(platforms[platform], opencl_device_type(), 0, NULL, &num_devices)))
				return;

			total_devices += num_devices;

			if(info.num - num_base >= num_devices) {
				/* num doesn't refer to a device in this platform */
				num_base += num_devices;
				continue;
			}

			/* device is in this platform */
			cpPlatform = platforms[platform];

			/* get devices */
			vector<cl_device_id> device_ids(num_devices, NULL);

			if(opencl_error(clGetDeviceIDs(cpPlatform, opencl_device_type(), num_devices, &device_ids[0], NULL)))
				return;

			cdDevice = device_ids[info.num - num_base];

			char name[256];
			clGetPlatformInfo(cpPlatform, CL_PLATFORM_NAME, sizeof(name), &name, NULL);
			platform_name = name;

			break;
		}

		if(total_devices == 0) {
			opencl_error("OpenCL: no devices found.");
			return;
		}
		else if(!cdDevice) {
			opencl_error("OpenCL: specified device not found.");
			return;
		}

		{
			/* try to use cached context */
			thread_scoped_lock cache_locker;
			cxContext = OpenCLCache::get_context(cpPlatform, cdDevice, cache_locker);

			if(cxContext == NULL) {
				/* create context properties array to specify platform */
				const cl_context_properties context_props[] = {
					CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
					0, 0
				};

				/* create context */
				cxContext = clCreateContext(context_props, 1, &cdDevice,
					context_notify_callback, cdDevice, &ciErr);

				if(opencl_error(ciErr))
					return;

				/* cache it */
				OpenCLCache::store_context(cpPlatform, cdDevice, cxContext, cache_locker);
			}
		}

		extensions = get_device_extensions(cdDevice, &ciErr);

		cqCommandQueue = clCreateCommandQueue(cxContext, cdDevice, 0, &ciErr);
		if(opencl_error(ciErr))
			return;

		null_mem = (device_ptr)clCreateBuffer(cxContext, CL_MEM_READ_ONLY, 1, NULL, &ciErr);
		if(opencl_error(ciErr))
			return;

		/* detect unified memory support */
		opencl_assert(clGetDeviceInfo(cdDevice, CL_DEVICE_HOST_UNIFIED_MEMORY,
				sizeof(cl_bool),  &use_unified_memory, NULL));

		device_initialized = true;
	}

	static void context_notify_callback(const char *err_info,
		const void *private_info, size_t cb, void *user_data)
	{
		char name[256];
		clGetDeviceInfo((cl_device_id)user_data, CL_DEVICE_NAME, sizeof(name), &name, NULL);

		fprintf(stderr, "OpenCL error (%s): %s\n", name, err_info);
		fflush(stderr);
	}


	bool opencl_version_check()
	{
		char version[256];

		int major, minor, req_major = 1, req_minor = 1;

		clGetPlatformInfo(cpPlatform, CL_PLATFORM_VERSION, sizeof(version), &version, NULL);

		if(sscanf(version, "OpenCL %d.%d", &major, &minor) < 2) {
			opencl_error(string_printf("OpenCL: failed to parse platform version string (%s).", version));
			return false;
		}

		if(!((major == req_major && minor >= req_minor) || (major > req_major))) {
			opencl_error(string_printf("OpenCL: platform version 1.1 or later required, found %d.%d", major, minor));
			return false;
		}

		clGetDeviceInfo(cdDevice, CL_DEVICE_OPENCL_C_VERSION, sizeof(version), &version, NULL);

		if(sscanf(version, "OpenCL C %d.%d", &major, &minor) < 2) {
			opencl_error(string_printf("OpenCL: failed to parse OpenCL C version string (%s).", version));
			return false;
		}

		if(!((major == req_major && minor >= req_minor) || (major > req_major))) {
			opencl_error(string_printf("OpenCL: C version 1.1 or later required, found %d.%d", major, minor));
			return false;
		}

		return true;
	}

	bool load_binary(const string& kernel_path, const string& clbin, const string *debug_src = NULL)
	{
		/* read binary into memory */
		vector<uint8_t> binary;

		if(!path_read_binary(clbin, binary)) {
			opencl_error(string_printf("OpenCL failed to read cached binary %s.", clbin.c_str()));
			return false;
		}

		/* create program */
		cl_int status;
		size_t size = binary.size();
		const uint8_t *bytes = &binary[0];

		cpProgram = clCreateProgramWithBinary(cxContext, 1, &cdDevice,
			&size, &bytes, &status, &ciErr);

		if(opencl_error(status) || opencl_error(ciErr)) {
			opencl_error(string_printf("OpenCL failed create program from cached binary %s.", clbin.c_str()));
			return false;
		}

		if(!build_kernel(kernel_path, debug_src))
			return false;

		return true;
	}

	bool save_binary(const string& clbin)
	{
		size_t size = 0;
		clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL);

		if(!size)
			return false;

		vector<uint8_t> binary(size);
		uint8_t *bytes = &binary[0];

		clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, sizeof(uint8_t*), &bytes, NULL);

		if(!path_write_binary(clbin, binary)) {
			opencl_error(string_printf("OpenCL failed to write cached binary %s.", clbin.c_str()));
			return false;
		}

		return true;
	}

	bool build_kernel(const string& kernel_path, const string *debug_src = NULL)
	{
		string build_options = opencl_kernel_build_options(platform_name, debug_src);
	
		ciErr = clBuildProgram(cpProgram, 0, NULL, build_options.c_str(), NULL, NULL);

		/* show warnings even if build is successful */
		size_t ret_val_size = 0;

		clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

		if(ret_val_size > 1) {
			vector<char> build_log(ret_val_size+1);
			clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, &build_log[0], NULL);

			build_log[ret_val_size] = '\0';
			fprintf(stderr, "OpenCL kernel build output:\n");
			fprintf(stderr, "%s\n", &build_log[0]);
			fflush(stderr);
		}

		if(ciErr != CL_SUCCESS) {
			opencl_error("OpenCL build failed: errors in console");
			return false;
		}

		return true;
	}

	bool compile_kernel(const string& kernel_path, const string& kernel_md5, const string *debug_src = NULL)
	{
		/* we compile kernels consisting of many files. unfortunately opencl
		 * kernel caches do not seem to recognize changes in included files.
		 * so we force recompile on changes by adding the md5 hash of all files */
		string source = "#include \"kernel.cl\" // " + kernel_md5 + "\n";
		source = path_source_replace_includes(source, kernel_path);

		if(debug_src)
			path_write_text(*debug_src, source);

		size_t source_len = source.size();
		const char *source_str = source.c_str();

		cpProgram = clCreateProgramWithSource(cxContext, 1, &source_str, &source_len, &ciErr);

		if(opencl_error(ciErr))
			return false;

		double starttime = time_dt();
		printf("Compiling OpenCL kernel ...\n");

		if(!build_kernel(kernel_path, debug_src))
			return false;

		printf("Kernel compilation finished in %.2lfs.\n", time_dt() - starttime);

		return true;
	}

	string device_md5_hash()
	{
		MD5Hash md5;
		char version[256], driver[256], name[256], vendor[256];

		clGetPlatformInfo(cpPlatform, CL_PLATFORM_VENDOR, sizeof(vendor), &vendor, NULL);
		clGetDeviceInfo(cdDevice, CL_DEVICE_VERSION, sizeof(version), &version, NULL);
		clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(name), &name, NULL);
		clGetDeviceInfo(cdDevice, CL_DRIVER_VERSION, sizeof(driver), &driver, NULL);

		md5.append((uint8_t*)vendor, strlen(vendor));
		md5.append((uint8_t*)version, strlen(version));
		md5.append((uint8_t*)name, strlen(name));
		md5.append((uint8_t*)driver, strlen(driver));

		string options = opencl_kernel_build_options(platform_name);
		md5.append((uint8_t*)options.c_str(), options.size());

		return md5.get_hex();
	}

	bool load_kernels(bool experimental)
	{
		/* verify if device was initialized */
		if(!device_initialized) {
			fprintf(stderr, "OpenCL: failed to initialize device.\n");
			fflush(stderr);
			return false;
		}

		/* try to use cached kernel */
		thread_scoped_lock cache_locker;
		cpProgram = OpenCLCache::get_program(cpPlatform, cdDevice, cache_locker);

		if(!cpProgram) {
			/* verify we have right opencl version */
			if(!opencl_version_check())
				return false;

			/* md5 hash to detect changes */
			string kernel_path = path_get("kernel");
			string kernel_md5 = path_files_md5_hash(kernel_path);
			string device_md5 = device_md5_hash();

			/* path to cached binary */
			string clbin = string_printf("cycles_kernel_%s_%s.clbin", device_md5.c_str(), kernel_md5.c_str());
			clbin = path_user_get(path_join("cache", clbin));

			/* path to preprocessed source for debugging */
			string clsrc, *debug_src = NULL;

			if(opencl_kernel_use_debug()) {
				clsrc = string_printf("cycles_kernel_%s_%s.cl", device_md5.c_str(), kernel_md5.c_str());
				clsrc = path_user_get(path_join("cache", clsrc));
				debug_src = &clsrc;
			}

			/* if exists already, try use it */
			if(path_exists(clbin) && load_binary(kernel_path, clbin, debug_src)) {
				/* kernel loaded from binary */
			}
			else {
				/* if does not exist or loading binary failed, compile kernel */
				if(!compile_kernel(kernel_path, kernel_md5, debug_src))
					return false;

				/* save binary for reuse */
				if(!save_binary(clbin))
					return false;
			}

			/* cache the program */
			OpenCLCache::store_program(cpPlatform, cdDevice, cpProgram, cache_locker);
		}

		/* find kernels */
		ckPathTraceKernel = clCreateKernel(cpProgram, "kernel_ocl_path_trace", &ciErr);
		if(opencl_error(ciErr))
			return false;

		ckFilmConvertKernel = clCreateKernel(cpProgram, "kernel_ocl_tonemap", &ciErr);
		if(opencl_error(ciErr))
			return false;

		ckShaderKernel = clCreateKernel(cpProgram, "kernel_ocl_shader", &ciErr);
		if(opencl_error(ciErr))
			return false;

		return true;
	}

	~OpenCLDevice()
	{
		task_pool.stop();

		if(null_mem)
			clReleaseMemObject(CL_MEM_PTR(null_mem));

		ConstMemMap::iterator mt;
		for(mt = const_mem_map.begin(); mt != const_mem_map.end(); mt++) {
			mem_free(*(mt->second));
			delete mt->second;
		}

		if(ckPathTraceKernel)
			clReleaseKernel(ckPathTraceKernel);  
		if(ckFilmConvertKernel)
			clReleaseKernel(ckFilmConvertKernel);  
		if(cpProgram)
			clReleaseProgram(cpProgram);
		if(cqCommandQueue)
			clReleaseCommandQueue(cqCommandQueue);
		if(cxContext)
			clReleaseContext(cxContext);
	}

	void mem_alloc(device_memory& mem, MemoryType type)
	{
		size_t size = mem.memory_size();

		cl_mem_flags mem_flag;
		void *mem_ptr = NULL;

		if(type == MEM_READ_ONLY)
			mem_flag = CL_MEM_READ_ONLY;
		else if(type == MEM_WRITE_ONLY)
			mem_flag = CL_MEM_WRITE_ONLY;
		else
			mem_flag = CL_MEM_READ_WRITE;

		if (use_unified_memory) {
			mem_flag |= CL_MEM_USE_HOST_PTR;
			mem_ptr = (void*)mem.data_pointer;
		}

		mem.device_pointer = (device_ptr)clCreateBuffer(cxContext, mem_flag, size, mem_ptr, &ciErr);

		opencl_assert(ciErr);

		stats.mem_alloc(size);
	}

	void mem_copy_to(device_memory& mem)
	{
		size_t size = mem.memory_size();

		/* this is blocking */
		if (!use_unified_memory) {
			ciErr = clEnqueueWriteBuffer(cqCommandQueue, CL_MEM_PTR(mem.device_pointer), CL_TRUE,
					0, size, (void*)mem.data_pointer, 0, NULL, NULL);
			opencl_assert(ciErr);
		}
		else {
			/* dummy map to enforce coherence */
			void *map_ptr = clEnqueueMapBuffer(cqCommandQueue,
				CL_MEM_PTR(mem.device_pointer), CL_TRUE, CL_MAP_WRITE, 0,
				size, 0, NULL, NULL, &ciErr);

			/* do actual copy if map pointer is not equal to data pointer */
			if (map_ptr != (void*)mem.data_pointer)
				memcpy(map_ptr, (void*)mem.data_pointer, size);

			clEnqueueUnmapMemObject(cqCommandQueue,
					CL_MEM_PTR(mem.device_pointer), map_ptr, 0, NULL, NULL);
		}
	}

	void mem_copy_from(device_memory& mem, int y, int w, int h, int elem)
	{
		size_t offset = elem*y*w;
		size_t size = elem*w*h;

		/* this is blocking */
		if (!use_unified_memory) {
			ciErr = clEnqueueReadBuffer(cqCommandQueue, CL_MEM_PTR(mem.device_pointer), CL_TRUE, offset, size, (uchar*)mem.data_pointer + offset, 0, NULL, NULL);
			opencl_assert(ciErr);
		}
		else {
			/* dummy map to enforce coherence */
			void *map_ptr = clEnqueueMapBuffer(cqCommandQueue,
				CL_MEM_PTR(mem.device_pointer), CL_TRUE, CL_MAP_READ, 0,
				size, 0, NULL, NULL, &ciErr);

			/* do actual copy if map pointer is not equal to data pointer */
			if (map_ptr != (void*)mem.data_pointer)
				memcpy((void*)mem.data_pointer, map_ptr, size);

			clEnqueueUnmapMemObject(cqCommandQueue,
					CL_MEM_PTR(mem.device_pointer), map_ptr, 0, NULL, NULL);
		}
	}

	void mem_zero(device_memory& mem)
	{
		if(mem.device_pointer) {
			size_t size = mem.memory_size();

			if (!use_unified_memory) {
				memset((void*)mem.data_pointer, 0, size);
				mem_copy_to(mem);
			}
			else {
				void *map_ptr = clEnqueueMapBuffer(cqCommandQueue,
					CL_MEM_PTR(mem.device_pointer), CL_TRUE, CL_MAP_WRITE, 0,
					size, 0, NULL, NULL, &ciErr);

				memset(map_ptr, 0, size);

				/* if map isn't our buffer, clear our buffer too */
				if (map_ptr != (void*)mem.data_pointer)
					memset((void*)mem.data_pointer, 0, size);

				clEnqueueUnmapMemObject(cqCommandQueue,
						CL_MEM_PTR(mem.device_pointer), map_ptr, 0, NULL, NULL);
			}
		}
	}

	void mem_free(device_memory& mem)
	{
		if(mem.device_pointer) {
			ciErr = clReleaseMemObject(CL_MEM_PTR(mem.device_pointer));
			mem.device_pointer = 0;
			opencl_assert(ciErr);

			stats.mem_free(mem.memory_size());
		}
	}

	void const_copy_to(const char *name, void *host, size_t size)
	{
		ConstMemMap::iterator i = const_mem_map.find(name);

		if(i == const_mem_map.end()) {
			device_vector<uchar> *data = new device_vector<uchar>();
			data->copy((uchar*)host, size);

			mem_alloc(*data, MEM_READ_ONLY);
			i = const_mem_map.insert(ConstMemMap::value_type(name, data)).first;
		}
		else {
			device_vector<uchar> *data = i->second;
			data->copy((uchar*)host, size);
		}

		mem_copy_to(*i->second);
	}

	void tex_alloc(const char *name, device_memory& mem, bool interpolation, bool periodic)
	{
		mem_alloc(mem, MEM_READ_ONLY);
		mem_copy_to(mem);
		assert(mem_map.find(name) == mem_map.end());
		mem_map.insert(MemMap::value_type(name, mem.device_pointer));
	}

	void tex_free(device_memory& mem)
	{
		if(mem.data_pointer)
			mem_free(mem);
	}

	size_t global_size_round_up(int group_size, int global_size)
	{
		int r = global_size % group_size;
		return global_size + ((r == 0)? 0: group_size - r);
	}

	void enqueue_kernel(cl_command_queue queue, cl_kernel kernel, size_t w, size_t h, cl_event *evt)
	{
		size_t workgroup_size, max_work_items[3];

		clGetKernelWorkGroupInfo(kernel, cdDevice,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, NULL);
		clGetDeviceInfo(cdDevice,
			CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_items), max_work_items, NULL);
	
#if 0
		/* try to divide evenly over 2 dimensions */
		size_t sqrt_workgroup_size = max(sqrt((double)workgroup_size), 1.0);
		size_t local_size[2] = {sqrt_workgroup_size, sqrt_workgroup_size};

		/* some implementations have max size 1 on 2nd dimension */
		if(local_size[1] > max_work_items[1]) {
			local_size[0] = workgroup_size/max_work_items[1];
			local_size[1] = max_work_items[1];
		}

		size_t global_size[2] = {global_size_round_up(local_size[0], w), global_size_round_up(local_size[1], h)};
#else
		/* let the device do as it pleases with local size */
		size_t global_size[2] = { w, h };
		size_t *local_size = NULL;
#endif

		/* if the passed event isn't null, release it */
		if (evt && *evt != NULL)
			clReleaseEvent(*evt);

		/* run kernel */
		ciErr = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, evt);
		opencl_assert(ciErr);
		//opencl_assert(clFlush(cqCommandQueue));
	}

	void path_trace(cl_command_queue queue, cl_kernel kernel, const RenderTile& rtile, int sample, cl_event *evt)
	{
		/* cast arguments to cl types */
		cl_mem d_data = CL_MEM_PTR(const_mem_map["__data"]->device_pointer);
		cl_mem d_buffer = CL_MEM_PTR(rtile.buffer);
		cl_mem d_rng_state = CL_MEM_PTR(rtile.rng_state);
		cl_int d_x = rtile.x;
		cl_int d_y = rtile.y;
		cl_int d_w = rtile.w;
		cl_int d_h = rtile.h;
		cl_int d_sample = sample;
		cl_int d_offset = rtile.offset;
		cl_int d_stride = rtile.stride;

		/* sample arguments */
		cl_uint narg = 0;
		ciErr = 0;

		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_data), (void*)&d_data);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_buffer), (void*)&d_buffer);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_rng_state), (void*)&d_rng_state);

#define KERNEL_TEX(type, ttype, name) \
	ciErr |= set_kernel_arg_mem(ckPathTraceKernel, &narg, #name);
#include "kernel_textures.h"

		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_sample), (void*)&d_sample);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_x), (void*)&d_x);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_y), (void*)&d_y);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_w), (void*)&d_w);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_h), (void*)&d_h);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_offset), (void*)&d_offset);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_stride), (void*)&d_stride);

		opencl_assert(ciErr);

		enqueue_kernel(queue, kernel, d_w, d_h, evt);
	}

	cl_int set_kernel_arg_mem(cl_kernel kernel, cl_uint *narg, const char *name)
	{
		cl_mem ptr;
		cl_int err = 0;

		MemMap::iterator i = mem_map.find(name);
		if(i != mem_map.end()) {
			ptr = CL_MEM_PTR(i->second);
		}
		else {
			/* work around NULL not working, even though the spec says otherwise */
			ptr = CL_MEM_PTR(null_mem);
		}
		
		err |= clSetKernelArg(kernel, (*narg)++, sizeof(ptr), (void*)&ptr);
		opencl_assert(err);

		return err;
	}

	void tonemap(cl_command_queue queue, cl_kernel kernel,
		DeviceTask& task, device_ptr buffer, device_ptr rgba)
	{
		/* cast arguments to cl types */
		cl_mem d_data = CL_MEM_PTR(const_mem_map["__data"]->device_pointer);
		cl_mem d_rgba = CL_MEM_PTR(rgba);
		cl_mem d_buffer = CL_MEM_PTR(buffer);
		cl_int d_x = task.x;
		cl_int d_y = task.y;
		cl_int d_w = task.w;
		cl_int d_h = task.h;
		cl_int d_sample = task.sample;
		cl_int d_offset = task.offset;
		cl_int d_stride = task.stride;

		/* sample arguments */
		cl_uint narg = 0;
		ciErr = 0;

		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_data), (void*)&d_data);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_rgba), (void*)&d_rgba);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_buffer), (void*)&d_buffer);

#define KERNEL_TEX(type, ttype, name) \
	ciErr |= set_kernel_arg_mem(kernel, &narg, #name);
#include "kernel_textures.h"

		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_sample), (void*)&d_sample);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_x), (void*)&d_x);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_y), (void*)&d_y);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_w), (void*)&d_w);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_h), (void*)&d_h);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_offset), (void*)&d_offset);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_stride), (void*)&d_stride);

		opencl_assert(ciErr);

		enqueue_kernel(queue, kernel, d_w, d_h, NULL);
	}

	void shader(cl_command_queue queue, cl_kernel kernel, DeviceTask& task)
	{
		/* cast arguments to cl types */
		cl_mem d_data = CL_MEM_PTR(const_mem_map["__data"]->device_pointer);
		cl_mem d_input = CL_MEM_PTR(task.shader_input);
		cl_mem d_output = CL_MEM_PTR(task.shader_output);
		cl_int d_shader_eval_type = task.shader_eval_type;
		cl_int d_shader_x = task.shader_x;
		cl_int d_shader_w = task.shader_w;

		/* sample arguments */
		cl_uint narg = 0;
		ciErr = 0;

		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_data), (void*)&d_data);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_input), (void*)&d_input);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_output), (void*)&d_output);

#define KERNEL_TEX(type, ttype, name) \
	ciErr |= set_kernel_arg_mem(ckShaderKernel, &narg, #name);
#include "kernel_textures.h"

		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_shader_eval_type), (void*)&d_shader_eval_type);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_shader_x), (void*)&d_shader_x);
		ciErr |= clSetKernelArg(kernel, narg++, sizeof(d_shader_w), (void*)&d_shader_w);

		opencl_assert(ciErr);

		enqueue_kernel(queue, kernel, task.shader_w, 1, NULL);
	}

	/* if we're not running on a subdevice, subdevice is NULL */
	void thread_run(DeviceTask *task)
	{
		if(task->type == DeviceTask::TONEMAP) {
			tonemap(cqCommandQueue, ckFilmConvertKernel, *task, task->buffer, task->rgba);
		}
		else if(task->type == DeviceTask::SHADER) {
			shader(cqCommandQueue, ckShaderKernel, *task);
		}
		else if(task->type == DeviceTask::PATH_TRACE) {
			RenderTile tile;

			/* keep rendering tiles until done */
			for(;;) {
				if (!task->acquire_tile(this, tile))
					break;

				int start_sample = tile.start_sample;
				int end_sample = tile.start_sample + tile.num_samples;

				for(int sample = start_sample; sample < end_sample; sample++) {
					if(task->get_cancel()) {
						if(task->need_finish_queue == false)
							break;
					}

					path_trace(cqCommandQueue, ckPathTraceKernel, tile, sample, NULL);

					tile.sample = sample + 1;

					task->update_progress(tile);
				}

				task->release_tile(tile);
			}
		}
	}

	/* simple unbounded producer/consumer queue */
	template<typename T>
	class ProducerConsumerQueue
	{
		std::queue<T> q;
		thread_mutex q_lock;
		thread_condition_variable q_not_empty;

	public:
		ProducerConsumerQueue() {}
		~ProducerConsumerQueue()
		{
			thread_scoped_lock lock(q_lock);
		}

		/* push one item into the queue */
		void enqueue(const T &item)
		{
			thread_scoped_lock lock(q_lock);
			bool was_empty = q.empty();
			q.push(item);
			if (was_empty)
				q_not_empty.notify_all();
		}

		/* get one item from the queue */
		T dequeue_one()
		{
			T result;

			thread_scoped_lock lock(q_lock);

			for (;;)
			{
				if (!q.empty()) {
					result = q.front();
					q.pop();
					return result;
				}

				q_not_empty.wait(lock);
			}
		}

		/* get all items from the queue */
		std::vector<T> dequeue_all()
		{
			std::vector<T> result;
			thread_scoped_lock lock(q_lock);
			if (!q.empty()) {
				result.reserve(q.size());
				while (!q.empty()) {
					result.push_back(q.front());
					q.pop();
				}
			}
		}
	};

	/* state machine to pump work into subdevice */
	class SubdevicePathTracer
	{
		RenderTile tile;
		cl_int ciErr;
		int start_sample, end_sample, sample;

		OpenCLDevice *parent;
		DeviceTask *task;

		cl_device_id work_device;
		cl_command_queue queue;
		cl_kernel kernel;

		cl_event trace_done_event;

		enum State {
			NEED_TILE,
			PATH_TRACING,
			DONE,
			FAILED
		};

		State state;

	public:
		SubdevicePathTracer()
			: ciErr(CL_SUCCESS)
			, start_sample(0), end_sample(0), sample(0)
			, work_device(NULL), queue(NULL)
			, kernel(NULL), trace_done_event(NULL)
			, state(NEED_TILE)
		{
		}

		~SubdevicePathTracer()
		{
			if (queue)
				clReleaseCommandQueue(queue);
			if (kernel)
				clReleaseKernel(kernel);
			if (work_device)
				parent->fission_ext.clReleaseDeviceEXT(work_device);
		}

		bool initialize(OpenCLDevice* parent_device, DeviceTask* device_task, cl_device_id sub_device)
		{
			parent = parent_device;
			task = device_task;
			work_device = sub_device;

			/* create our own command queue */
			queue = clCreateCommandQueue(parent->cxContext, sub_device, 0, &ciErr);
			if (ciErr != CL_SUCCESS) {
				state = FAILED;
				return false;
			}

			/* create our own kernel so argument setting won't conflict */
			kernel = clCreateKernel(parent->cpProgram, "kernel_ocl_path_trace", &ciErr);
			if (ciErr != CL_SUCCESS) {
				state = FAILED;
				return false;
			}

			return true;
		}

		/* called by opencl when a kernel invocation is complete */
		static void TraceDoneCallback_C(cl_event event, cl_int status, void *user_data)
		{
			reinterpret_cast<SubdevicePathTracer*>(user_data)->TraceDoneCallback(event, status);
		}

		void TraceDoneCallback(cl_event event, cl_int status)
		{

		}

		/* returns true when there is still work to do */
		bool do_work()
		{
			switch (state) {
			case NEED_TILE:
				/* acquire a tile and get range of samples */
				if (task->acquire_tile(parent, tile)) {
					start_sample = tile.start_sample;
					end_sample = start_sample + tile.num_samples;
					sample = start_sample;

					state = PATH_TRACING;
				}
				else {
					state = DONE;
				}
				break;

			case PATH_TRACING:
				/* enqueue a path_trace */
				parent->path_trace(queue, kernel, tile, sample, &trace_done_event);

				ciErr = clSetEventCallback(trace_done_event, CL_COMPLETE, TraceDoneCallback_C, this);
				if (ciErr != CL_SUCCESS) {
					state = FAILED;
					break;
				}

				ciErr = clFlush(queue);
				if (ciErr != CL_SUCCESS) {
					state = FAILED;
					break;
				}

				if (++sample == end_sample)
					state = NEED_TILE;
				break;

			case DONE:	/* fall thru */
			case FAILED:
				return false;

			}

			return true;
		}
	};

	void thread_run_subdevices(DeviceTask* task, std::vector<cl_device_id> subdevices)
	{
		assert(task->type == DeviceTask::PATH_TRACE);

		std::vector<SubdevicePathTracer> tracers;
		tracers.resize(subdevices.size());

		for (size_t i = 0; i < subdevices.size(); ++i)
			tracers[i].initialize(this, task, subdevices[i]);

		for (;;) {
			int done_count = 0;
			for (size_t i = 0; i < tracers.size(); ++i)
				if (tracers[i].do_work())
					++done_count;

			if (done_count == tracers.size())
				break;
		}
	}

	class OpenCLDeviceTask : public DeviceTask {
	public:
		OpenCLDeviceTask(OpenCLDevice *device, DeviceTask& task)
		: DeviceTask(task)
		{
			run = function_bind(&OpenCLDevice::thread_run, device, this);
		}

		OpenCLDeviceTask(OpenCLDevice *device, DeviceTask& task, std::vector<cl_device_id> &sub_devices)
		: DeviceTask(task)
		{
			run = function_bind(&OpenCLDevice::thread_run_subdevices, device, this, sub_devices);
		}
	};

	void task_add(DeviceTask& task)
	{
		/* see if the device supports device fission. If so, split it up
		 * into multiple devices */

		/* not really a loop */
		do {
			/* disabled for now */
			break;

			/* only try to use device fission for path_trace task */
			if (task.type != DeviceTask::PATH_TRACE)
				break;

			/* if device doesn't implement device fission, break */
			if (extensions.find("cl_ext_device_fission") == extensions.end())
				break;

			/* if device fission extension doesn't successfully initialize, break */
			if (!fission_ext.initialize())
				break;

			/* ask for the device to be split up as finely as possible */
			DeviceFissionExt::cl_device_partition_property_ext props[] = {
				DeviceFissionExt::CL_DEVICE_PARTITION_EQUALLY_EXT, 1,
				DeviceFissionExt::CL_PROPERTIES_LIST_END_EXT
			};

			cl_uint subdevice_count = 0;

			/* create the subdevices */
			ciErr = fission_ext.clCreateSubDevicesEXT(cdDevice,
				props, 0, NULL, &subdevice_count);

			/* if it failed or didn't create any subdevices, break */
			if (ciErr != CL_SUCCESS || subdevice_count < 1)
				break;

			std::vector<cl_device_id> subdevices;
			subdevices.resize(subdevice_count);

			/* create the subdevices */
			ciErr = fission_ext.clCreateSubDevicesEXT(cdDevice,
				props, subdevice_count, &subdevices[0], NULL);
			if (ciErr != CL_SUCCESS)
				break;

			printf("OpenCL: using device fission, %u devices\n", subdevice_count);

			task_pool.push(new OpenCLDeviceTask(this, task, subdevices));

			return;
		} while (false);

		/* discard errors that happened above */
		ciErr = CL_SUCCESS;

		/* no device fission available, just run task as usual */
		task_pool.push(new OpenCLDeviceTask(this, task));
	}

	void task_wait()
	{
		task_pool.wait_work();
	}

	void task_cancel()
	{
		task_pool.cancel();
	}
};

Device *device_opencl_create(DeviceInfo& info, Stats &stats, bool background)
{
	if (info.use_fission) {
		/* build a multi device */
		return OpenCLDevice::create_fission_device(info, stats, background);
	}

	return new OpenCLDevice(info, stats, background);
}

void device_opencl_info(vector<DeviceInfo>& devices)
{
	vector<cl_device_id> device_ids;
	cl_uint num_devices = 0;
	vector<cl_platform_id> platform_ids;
	cl_uint num_platforms = 0;

	/* get devices */
	if(clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0)
		return;
	
	platform_ids.resize(num_platforms);

	if(clGetPlatformIDs(num_platforms, &platform_ids[0], NULL) != CL_SUCCESS)
		return;

	/* devices are numbered consecutively across platforms */
	int num_base = 0;

	DeviceFissionExt fission_ext;

	for (int platform = 0; platform < num_platforms; platform++, num_base += num_devices) {
		num_devices = 0;
		if(clGetDeviceIDs(platform_ids[platform], opencl_device_type(), 0, NULL, &num_devices) != CL_SUCCESS || num_devices == 0)
			continue;

		device_ids.resize(num_devices);

		if(clGetDeviceIDs(platform_ids[platform], opencl_device_type(), num_devices, &device_ids[0], NULL) != CL_SUCCESS)
			continue;

		char pname[256];
		clGetPlatformInfo(platform_ids[platform], CL_PLATFORM_NAME, sizeof(pname), &pname, NULL);
		string platform_name = pname;

		/* add devices */
		for(int num = 0; num < num_devices; num++) {
			cl_device_id device_id = device_ids[num];
			char name[1024] = "\0";

			if(clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(name), &name, NULL) != CL_SUCCESS)
				continue;

			DeviceInfo info;

			info.type = DEVICE_OPENCL;
			info.description = string(name);
			info.num = num_base + num;
			info.id = string_printf("OPENCL_%d", info.num);
			/* we don't know if it's used for display, but assume it is */
			info.display_device = true;
			info.advanced_shading = opencl_kernel_use_advanced_shading(platform_name);
			info.pack_images = true;

			devices.push_back(info);

			/* See if the device supports fission */
			std::set<string> extensions;
			extensions = get_device_extensions(device_id, NULL);

			if (extensions.find("cl_ext_device_fission") != extensions.end()
					&& fission_ext.initialize()) {
				cl_uint subdevice_count;

				DeviceFissionExt::cl_device_partition_property_ext props[] = {
					DeviceFissionExt::CL_DEVICE_PARTITION_EQUALLY_EXT, 1,
					DeviceFissionExt::CL_PROPERTIES_LIST_END_EXT
				};

				fission_ext.clCreateSubDevicesEXT(device_id,
						props, 0, NULL, &subdevice_count);

				if (subdevice_count > 1) {
					std::stringstream ss;
					ss << name << " (x" << subdevice_count << ")";

					/* create a duplicate num, but set device fission flag */
					info.id += "_fission";
					info.description = ss.str();
					info.use_fission = true;
					devices.push_back(info);
				}
			}
		}
	}
}

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */

