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

#include <stdlib.h>
#include <string.h>

#include "device.h"
#include "device_intern.h"

#include "kernel.h"
#include "kernel_compat_cpu.h"
#include "kernel_types.h"
#include "kernel_globals.h"

#include "osl_shader.h"
#include "osl_globals.h"

#include "buffers.h"

#include "util_debug.h"
#include "util_foreach.h"
#include "util_function.h"
#include "util_opengl.h"
#include "util_progress.h"
#include "util_system.h"
#include "util_thread.h"

CCL_NAMESPACE_BEGIN

class CPUDevice : public Device
{
public:
	TaskPool task_pool;
	KernelGlobals kernel_globals;
#ifdef WITH_OSL
	OSLGlobals osl_globals;
#endif
	
	CPUDevice(Stats &stats) : Device(stats)
	{
#ifdef WITH_OSL
		kernel_globals.osl = &osl_globals;
#endif

		/* do now to avoid thread issues */
		system_cpu_support_sse2();
		system_cpu_support_sse3();
		system_cpu_support_sse4();
		system_cpu_support_avx1();
	}

	~CPUDevice()
	{
		task_pool.stop();
	}

	void mem_alloc(device_memory& mem, MemoryType type)
	{
		mem.device_pointer = mem.data_pointer;

		stats.mem_alloc(mem.memory_size());
	}

	void mem_copy_to(device_memory& mem)
	{
		/* no-op */
	}

	void mem_copy_from(device_memory& mem, int y, int w, int h, int elem)
	{
		/* no-op */
	}

	void mem_zero(device_memory& mem)
	{
		memset((void*)mem.device_pointer, 0, mem.memory_size());
	}

	void mem_free(device_memory& mem)
	{
		mem.device_pointer = 0;

		stats.mem_free(mem.memory_size());
	}

	void const_copy_to(const char *name, void *host, size_t size)
	{
		kernel_const_copy(&kernel_globals, name, host, size);
	}

	void tex_alloc(const char *name, device_memory& mem, bool interpolation, bool periodic)
	{
		kernel_tex_copy(&kernel_globals, name, mem.data_pointer, mem.data_width, mem.data_height);
		mem.device_pointer = mem.data_pointer;

		stats.mem_alloc(mem.memory_size());
	}

	void tex_free(device_memory& mem)
	{
		mem.device_pointer = 0;

		stats.mem_free(mem.memory_size());
	}

	void *osl_memory()
	{
#ifdef WITH_OSL
		return &osl_globals;
#else
		return NULL;
#endif
	}

	/* function pointer for optimized implementations */
	typedef void (*path_trace_impl_t)(KernelGlobals *kg, float *buffer, unsigned int *rng_state, int sample, int x, int y, int offset, int stride);
	typedef void (*tonemap_impl_t)(KernelGlobals *kg, uchar4 *rgba, float *buffer, int sample, int x, int y, int offset, int stride);
	typedef void (*shader_impl_t)(KernelGlobals *kg, uint4 *input, float4 *output, int type, int i);

	/* specializations of this function call the corresponding optimized
	 * implementation directly with no indirection */
	template<path_trace_impl_t path_trace_impl>
	void thread_path_trace(DeviceTask& task)
	{
		if(task_pool.cancelled()) {
			if(task.need_finish_queue == false)
				return;
		}

		KernelGlobals kg = kernel_globals;

#ifdef WITH_OSL
		OSLShader::thread_init(&kg, &kernel_globals, &osl_globals);
#endif

		RenderTile tile;

		while(task.acquire_tile(this, tile)) {
			float *render_buffer = (float*)tile.buffer;
			uint *rng_state = (uint*)tile.rng_state;
			int start_sample = tile.start_sample;
			int end_sample = tile.start_sample + tile.num_samples;

#if 0 // experiment: reorganize loop to maximize cache coherency. exactly same speed on my machine
			for(int y = tile.y; y < tile.y + tile.h; y++) {
				if (task.get_cancel() || task_pool.cancelled()) {
					if(task.need_finish_queue == false)
						break;
				}

				for(int x = tile.x; x < tile.x + tile.w; x++) {
					for(int sample = start_sample; sample < end_sample; sample++) {
						tile.sample = sample + 1;
						path_trace_impl(&kg, render_buffer, rng_state,
							sample, x, y, tile.offset, tile.stride);
					}
				}

				task.update_progress(tile);
			}
#else // original implementation here:
			for(int sample = start_sample; sample < end_sample; sample++) {
				if (task.get_cancel() || task_pool.cancelled()) {
					if(task.need_finish_queue == false)
						break;
				}

				for(int y = tile.y; y < tile.y + tile.h; y++) {
					for(int x = tile.x; x < tile.x + tile.w; x++) {
						path_trace_impl(&kg, render_buffer, rng_state,
							sample, x, y, tile.offset, tile.stride);
					}
				}

				tile.sample = sample + 1;

				task.update_progress(tile);
			}
#endif

			task.release_tile(tile);

			if(task_pool.cancelled()) {
				if(task.need_finish_queue == false)
					break;
			}
		}

#ifdef WITH_OSL
		OSLShader::thread_free(&kg);
#endif
	}

	template<tonemap_impl_t tonemap_impl>
	void thread_tonemap(DeviceTask& task)
	{
		for(int y = task.y; y < task.y + task.h; y++)
			for(int x = task.x; x < task.x + task.w; x++)
				tonemap_impl(&kernel_globals, (uchar4*)task.rgba, (float*)task.buffer,
					task.sample, x, y, task.offset, task.stride);
	}

	template<shader_impl_t shader_impl>
	void thread_shader(DeviceTask& task)
	{
		KernelGlobals kg = kernel_globals;

#ifdef WITH_OSL
		OSLShader::thread_init(&kg, &kernel_globals, &osl_globals);
#endif

		for(int x = task.shader_x; x < task.shader_x + task.shader_w; x++) {
			shader_impl(&kg, (uint4*)task.shader_input, (float4*)task.shader_output, task.shader_eval_type, x);

			if(task_pool.cancelled())
				break;
		}

#ifdef WITH_OSL
		OSLShader::thread_free(&kg);
#endif
	}

	void thread_run(DeviceTask *task)
	{
		unsigned old_csr = 0;
		if (system_cpu_support_sse2()) {
			/* configure SSE to flush denormals to zero and flush
			 * denormal results to zero. Denormals are VERY expensive,
			 * they cause hundreds of cycles of microcode "assists" */
			old_csr = _mm_getcsr();
			/* bit 15 is FTZ, bit 6 is DAZ */
			_mm_setcsr(old_csr | (1 << 15) | (1 << 6));
		}

		if(task->type == DeviceTask::PATH_TRACE) {
#ifdef WITH_OPTIMIZED_KERNEL
			if(system_cpu_support_avx1()) {
				thread_path_trace<kernel_cpu_avx1_path_trace>(*task);
			} else if(system_cpu_support_sse4()) {
				thread_path_trace<kernel_cpu_sse4_path_trace>(*task);
			} else if(system_cpu_support_sse3()) {
				thread_path_trace<kernel_cpu_sse3_path_trace>(*task);
			} else if(system_cpu_support_sse2()) {
				thread_path_trace<kernel_cpu_sse2_path_trace>(*task);
			} else {
				thread_path_trace<kernel_cpu_path_trace>(*task);
			}
#else
			thread_path_trace<kernel_cpu_path_trace>(*task);
#endif
		} else if(task->type == DeviceTask::TONEMAP) {
#ifdef WITH_OPTIMIZED_KERNEL
			if(system_cpu_support_avx1()) {
				thread_tonemap<kernel_cpu_avx1_tonemap>(*task);
			} else if(system_cpu_support_sse4()) {
				thread_tonemap<kernel_cpu_sse4_tonemap>(*task);
			} else if(system_cpu_support_sse3()) {
				thread_tonemap<kernel_cpu_sse3_tonemap>(*task);
			} else if(system_cpu_support_sse2()) {
				thread_tonemap<kernel_cpu_sse2_tonemap>(*task);
			} else {
				thread_tonemap<kernel_cpu_tonemap>(*task);
			}
#else
			thread_tonemap<kernel_cpu_tonemap>(*task);
#endif
		} else if(task->type == DeviceTask::SHADER) {
#ifdef WITH_OPTIMIZED_KERNEL
			/* configure SSE to flush denormals to zero and flush
			 * denormal results to zero */

			if(system_cpu_support_avx1()) {
				thread_shader<kernel_cpu_avx1_shader>(*task);
			} else if(system_cpu_support_sse4()) {
				thread_shader<kernel_cpu_sse4_shader>(*task);
			} else if(system_cpu_support_sse3()) {
				thread_shader<kernel_cpu_sse3_shader>(*task);
			} else if(system_cpu_support_sse2()) {
				thread_shader<kernel_cpu_sse2_shader>(*task);
			} else {
				thread_shader<kernel_cpu_shader>(*task);
			}
#else
			thread_shader<kernel_cpu_shader>(*task);
#endif
		}

		/* restore FPU config */
		if (system_cpu_support_sse2()) {
			_mm_setcsr(old_csr);
		}

	}

	class CPUDeviceTask : public DeviceTask {
	public:
		CPUDeviceTask(CPUDevice *device, DeviceTask& task)
		: DeviceTask(task)
		{
			run = function_bind(&CPUDevice::thread_run, device, this);
		}
	};

	void task_add(DeviceTask& task)
	{
		/* split task into smaller ones */
		list<DeviceTask> tasks;
		task.split(tasks, TaskScheduler::num_threads());

		foreach(DeviceTask& task, tasks)
			task_pool.push(new CPUDeviceTask(this, task));
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

Device *device_cpu_create(DeviceInfo& info, Stats &stats)
{
	return new CPUDevice(stats);
}

void device_cpu_info(vector<DeviceInfo>& devices)
{
	DeviceInfo info;

	info.type = DEVICE_CPU;
	info.description = system_cpu_brand_string();
	info.id = "CPU";
	info.num = 0;
	info.advanced_shading = true;
	info.pack_images = false;

	devices.insert(devices.begin(), info);
}

CCL_NAMESPACE_END

