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

#include "device.h"
#include "device_intern.h"
#include "device_network.h"

#include "util_foreach.h"
#include "util_debugtrace.h"

#if defined(WITH_NETWORK)

CCL_NAMESPACE_BEGIN

thread_mutex SyncOutputStream::stream_lock;

typedef map<device_ptr, device_ptr> PtrMap;
typedef vector<uint8_t> DataVector;
typedef map<device_ptr, DataVector> DataMap;

/* tile list */
typedef vector<RenderTile> TileList;

/* search a list of tiles and find the one that matches the passed render tile */
static TileList::iterator tile_list_find(TileList& tile_list, RenderTile& tile)
{
	for(TileList::iterator it = tile_list.begin(); it != tile_list.end(); ++it)
		if(tile.x == it->x && tile.y == it->y && tile.start_sample == it->start_sample)
			return it;
	return tile_list.end();
}

class NetworkDevice : public Device
{
public:
	boost::asio::io_service io_service;
	tcp::socket socket;
	device_ptr mem_counter;
	DeviceTask the_task; /* todo: handle multiple tasks */

	RPCStreamManager rpc_stream;

	NetworkDevice(DeviceInfo& info, Stats &stats, const char *address)
		: Device(info, stats, true)
		, socket(io_service)
	{
		mem_counter = 0;
		rpc_stream.connect_to_server(address);
	}

	~NetworkDevice()
	{
		CyclesRPCCallFactory::rpc_stop(rpc_stream);
	}

	void mem_alloc(device_memory& mem, MemoryType type)
	{
		mem.device_pointer = ++mem_counter;
		CyclesRPCCallFactory::rpc_mem_alloc(rpc_stream, mem, type);
	}

	void mem_copy_to(device_memory& mem)
	{
		CyclesRPCCallFactory::rpc_mem_copy_to(rpc_stream, mem);
	}

	void mem_copy_from(device_memory& mem, int y, int w, int h, int elem)
	{
		CyclesRPCCallFactory::rpc_mem_copy_from(rpc_stream,
				mem, y, w, h, elem, (void*)mem.data_pointer);
	}

	void mem_zero(device_memory& mem)
	{
		CyclesRPCCallFactory::rpc_mem_zero(rpc_stream, mem);
	}

	void mem_free(device_memory& mem)
	{
		if(mem.device_pointer) {
			CyclesRPCCallFactory::rpc_mem_free(rpc_stream, mem);
			mem.device_pointer = 0;
		}
	}

	void const_copy_to(const char *name, void *data, size_t size)
	{
		CyclesRPCCallFactory::rpc_const_copy_to(rpc_stream,
			std::string(name), data, size);
	}

	void tex_alloc(const char *name, device_memory& mem, bool interpolation, bool periodic)
	{
		mem.device_pointer = ++mem_counter;

		CyclesRPCCallFactory::rpc_tex_alloc(rpc_stream, name, mem, interpolation, periodic);
	}

	void tex_free(device_memory& mem)
	{
		if(mem.device_pointer) {
			CyclesRPCCallFactory::rpc_tex_free(rpc_stream, mem);
			mem.device_pointer = 0;
		}
	}

	bool load_kernels(bool experimental)
	{
		return CyclesRPCCallFactory::rpc_load_kernels(rpc_stream, experimental);
	}

	void task_add(DeviceTask& task)
	{
		the_task = task;
		CyclesRPCCallFactory::rpc_task_add(rpc_stream, task);
	}

	void task_wait()
	{
		CyclesRPCCallFactory::rpc_task_wait(rpc_stream);

		TileList the_tiles;

		/* todo: run this threaded for connecting to multiple clients */
		bool done = false;
		do {
			RenderTile tile;

			CyclesRPCCallBase *request = rpc_stream.wait_request();

			switch (request->get_call_id())
			{
			case CyclesRPCCallBase::acquire_tile_request:
				if(the_task.acquire_tile(this, tile)) { /* write return as bool */
					the_tiles.push_back(tile);

					CyclesRPCCallFactory::rpc_acquire_tile_response(rpc_stream, request, true, tile);
				}
				else {
					CyclesRPCCallFactory::rpc_acquire_tile_response(rpc_stream, request, false, tile);
				}
				break;

			case CyclesRPCCallBase::release_tile_request:
				request->read(tile);

				TileList::iterator it = tile_list_find(the_tiles, tile);
				if (it != the_tiles.end()) {
					tile.buffers = it->buffers;
					the_tiles.erase(it);
				}

				assert(tile.buffers != NULL);

				the_task.release_tile(tile);

				/* what's going on here? */

				RPCSend snd(socket, "release_tile");
				snd.write();
				lock.unlock();

				break;

			case CyclesRPCCallBase::task_wait_done_request:
				done = true;
				break;

			default:
				break;
			}
		} while (!done);
	}

	void task_cancel()
	{
		thread_scoped_lock lock(rpc_lock);
		RPCSend snd(socket, "task_cancel");
		snd.write();
	}
};

Device *device_network_create(DeviceInfo& info, Stats &stats, const char *address)
{
	return new NetworkDevice(info, stats, address);
}

void device_network_info(vector<DeviceInfo>& devices)
{
	DeviceInfo info;

	info.type = DEVICE_NETWORK;
	info.description = "Network Device";
	info.id = "NETWORK";
	info.num = 0;
	info.advanced_shading = true; /* todo: get this info from device */
	info.pack_images = false;

	devices.push_back(info);
}

class DeviceServer {
public:
	DeviceServer(Device *device_, tcp::socket& socket_)
		: device(device_), rpc_stream(socket_)
	{
	}

	void listen()
	{
		/* receive remote function calls */
		for(;;) {
			thread_scoped_lock lock(rpc_lock);
			RPCReceive rcv(socket);

			if(rcv.name == "stop")
				break;

			process(rcv, lock);
		}
	}

protected:
	/* create a memory buffer for a device buffer and insert it into mem_data */
	DataVector &data_vector_insert(device_ptr client_pointer, size_t data_size)
	{
		/* create a new DataVector and insert it into mem_data */
		pair<DataMap::iterator,bool> data_ins = mem_data.insert(
				DataMap::value_type(client_pointer, DataVector()));

		/* make sure it was a unique insertion */
		assert(data_ins.second);

		/* get a reference to the inserted vector */
		DataVector &data_v = data_ins.first->second;

		/* size the vector */
		data_v.resize(data_size);

		return data_v;
	}

	DataVector &data_vector_find(device_ptr client_pointer)
	{
		DataMap::iterator i = mem_data.find(client_pointer);
		assert(i != mem_data.end());
		return i->second;
	}

	/* setup mapping and reverse mapping of client_pointer<->real_pointer */
	void pointer_mapping_insert(device_ptr client_pointer, device_ptr real_pointer)
	{
		pair<PtrMap::iterator,bool> mapins;

		/* insert mapping from client pointer to our real device pointer */
		mapins = ptr_map.insert(PtrMap::value_type(client_pointer, real_pointer));
		assert(mapins.second);

		/* insert reverse mapping from real our device pointer to client pointer */
		mapins = ptr_imap.insert(PtrMap::value_type(real_pointer, client_pointer));
		assert(mapins.second);
	}

	device_ptr device_ptr_from_client_pointer(device_ptr client_pointer)
	{
		PtrMap::iterator i = ptr_map.find(client_pointer);
		assert(i != ptr_map.end());
		return i->second;
	}

	device_ptr device_ptr_from_client_pointer_erase(device_ptr client_pointer)
	{
		PtrMap::iterator i = ptr_map.find(client_pointer);
		assert(i != ptr_map.end());

		device_ptr result = i->second;

		/* erase the mapping */
		ptr_map.erase(i);

		/* erase the reverse mapping */
		PtrMap::iterator irev = ptr_imap.find(result);
		assert(irev != ptr_imap.end());
		ptr_imap.erase(irev);

		/* erase the data vector */
		DataMap::iterator idata = mem_data.find(client_pointer);
		assert(idata != mem_data.end());
		mem_data.erase(idata);

		return result;
	}

	/* note that the lock must be already acquired upon entry.
	 * This is necessary because the caller often peeks at
	 * the header and delegates control to here when it doesn't
	 * specifically handle the current RPC.
	 * The lock must be unlocked before returning */
	void process(CyclesRPCCallBase& rcv, thread_scoped_lock &lock)
	{
		fprintf(stderr, "receive process %s\n", rcv.name.c_str());

		switch (rcv.get_call_id()) {
		case CyclesRPCCallBase::mem_alloc_request:
		{
			MemoryType type;
			network_device_memory mem;
			device_ptr client_pointer;

			rcv.read(mem);
			rcv.read(type);

			lock.unlock();

			client_pointer = mem.device_pointer;

			/* create a memory buffer for the device buffer */
			size_t data_size = mem.memory_size();
			DataVector &data_v = data_vector_insert(client_pointer, data_size);

			if(data_size)
				mem.data_pointer = (device_ptr)&(data_v[0]);
			else
				mem.data_pointer = 0;

			/* perform the allocation on the actual device */
			device->mem_alloc(mem, type);

			/* store a mapping to/from client_pointer and real device pointer */
			pointer_mapping_insert(client_pointer, mem.device_pointer);
			break;
		}
		case CyclesRPCCallBase::mem_mem_copy_to_request:
		{
			network_device_memory mem;

			rcv.read(mem);
			lock.unlock();

			device_ptr client_pointer = mem.device_pointer;

			DataVector &data_v = data_vector_find(client_pointer);

			size_t data_size = mem.memory_size();

			/* get pointer to memory buffer	for device buffer */
			mem.data_pointer = (device_ptr)&data_v[0];

			/* copy data from network into memory buffer */
			rcv.read_buffer((uint8_t*)mem.data_pointer, data_size);

			/* translate the client pointer to a real device pointer */
			mem.device_pointer = device_ptr_from_client_pointer(client_pointer);

			/* copy the data from the memory buffer to the device buffer */
			device->mem_copy_to(mem);
			break;
		}
		case CyclesRPCCallBase::mem_copy_from_request:
		{
			network_device_memory mem;
			int y, w, h, elem;

			rcv.read(mem);
			rcv.read(y);
			rcv.read(w);
			rcv.read(h);
			rcv.read(elem);

			device_ptr client_pointer = mem.device_pointer;
			mem.device_pointer = device_ptr_from_client_pointer(client_pointer);

			DataVector &data_v = data_vector_find(client_pointer);

			mem.data_pointer = (device_ptr)&(data_v[0]);

			device->mem_copy_from(mem, y, w, h, elem);

			size_t data_size = mem.memory_size();
			SyncOutputStream() << "Responding to mem_copy_from size=" << data_size;

			RPCSend snd(socket);
			snd.write();
			snd.write_buffer((uint8_t*)mem.data_pointer, data_size);
			lock.unlock();
			break;
		}
		case CyclesRPCCallBase::mem_zero_request:
		{
			network_device_memory mem;
			
			rcv.read(mem);
			lock.unlock();

			device_ptr client_pointer = mem.device_pointer;
			mem.device_pointer = device_ptr_from_client_pointer(client_pointer);

			DataVector &data_v = data_vector_find(client_pointer);

			mem.data_pointer = (device_ptr)&(data_v[0]);

			device->mem_zero(mem);
		}
		case CyclesRPCCallBase::mem_free_request:
		{
			network_device_memory mem;
			device_ptr client_pointer;

			rcv.read(mem);
			lock.unlock();

			client_pointer = mem.device_pointer;

			mem.device_pointer = device_ptr_from_client_pointer_erase(client_pointer);

			device->mem_free(mem);
		}
		case CyclesRPCCallBase::const_copy_to_request:
		{
			string name_string;
			size_t size;

			rcv.read(name_string);
			rcv.read(size);

			vector<char> host_vector(size);
			rcv.read_buffer(&host_vector[0], size);
			lock.unlock();

			device->const_copy_to(name_string.c_str(), &host_vector[0], size);
		}
		case CyclesRPCCallBase::tex_alloc_request:
		{
			network_device_memory mem;
			string name;
			bool interpolation;
			bool periodic;
			device_ptr client_pointer;

			rcv.read(name);
			rcv.read(mem);
			rcv.read(interpolation);
			rcv.read(periodic);
			lock.unlock();

			client_pointer = mem.device_pointer;

			size_t data_size = mem.memory_size();

			DataVector &data_v = data_vector_insert(client_pointer, data_size);

			if(data_size)
				mem.data_pointer = (device_ptr)&(data_v[0]);
			else
				mem.data_pointer = 0;

			rcv.read_buffer((uint8_t*)mem.data_pointer, data_size);

			device->tex_alloc(name.c_str(), mem, interpolation, periodic);

			pointer_mapping_insert(client_pointer, mem.device_pointer);
		}
		case CyclesRPCCallBase::tex_free_request:
		{
			network_device_memory mem;
			device_ptr client_pointer;

			rcv.read(mem);
			lock.unlock();

			client_pointer = mem.device_pointer;

			mem.device_pointer = device_ptr_from_client_pointer_erase(client_pointer);

			device->tex_free(mem);
		}
		case CyclesRPCCallBase::load_kernels_request:
		{
			bool experimental;
			rcv.read(experimental);

			bool result;
			result = device->load_kernels(experimental);
			RPCSend snd(socket);
			snd.add(result);
			snd.write();
			lock.unlock();
		}
		case CyclesRPCCallBase::task_add_request:
		{
			DeviceTask task;

			rcv.read(task);
			lock.unlock();

			if(task.buffer)
				task.buffer = device_ptr_from_client_pointer(task.buffer);

			if(task.rgba)
				task.rgba = device_ptr_from_client_pointer(task.rgba);

			if(task.shader_input)
				task.shader_input = device_ptr_from_client_pointer(task.shader_input);

			if(task.shader_output)
				task.shader_output = device_ptr_from_client_pointer(task.shader_output);

			task.acquire_tile = function_bind(&DeviceServer::task_acquire_tile, this, _1, _2);
			task.release_tile = function_bind(&DeviceServer::task_release_tile, this, _1);
			task.update_progress_sample = function_bind(&DeviceServer::task_update_progress_sample, this);
			task.update_tile_sample = function_bind(&DeviceServer::task_update_tile_sample, this, _1);
			task.get_cancel = function_bind(&DeviceServer::task_get_cancel, this);

			device->task_add(task);
		}
		case CyclesRPCCallBase::task_wait_request:
		{
			lock.unlock();
			device->task_wait();

			lock.lock();
			RPCSend snd(socket, "task_wait_done");
			snd.write();
			lock.unlock();
		}
		case CyclesRPCCallBase::task_cancel_request:
		{
			lock.unlock();
			device->task_cancel();
		}
		case CyclesRPCCallBase::acquire_tile_request:
		{
			RenderTile tile;
			rcv.read(tile);
			lock.unlock();
			//
		}
		default:
			SyncOutputStream() << "Unhandled op in CyclesServer::process" << rcv.name;
			raise(SIGTRAP);
		}
	}

	bool task_acquire_tile(Device *device, RenderTile& tile)
	{
		thread_scoped_lock acquire_lock(acquire_mutex);

		bool result = false;

		RPCSend snd(socket, "acquire_tile");
		snd.write();

		while(1) {
			thread_scoped_lock lock(rpc_lock);
			RPCReceive rcv(socket);

			if(rcv.name == "acquire_tile") {
				rcv.read(tile);

				if(tile.buffer) tile.buffer = ptr_map[tile.buffer];
				if(tile.rng_state) tile.rng_state = ptr_map[tile.rng_state];
				if(tile.rgba) tile.rgba = ptr_map[tile.rgba];

				result = true;
				break;
			}
			else if(rcv.name == "acquire_tile_none")
				break;
			else
				process(rcv, lock);
		}

		return result;
	}

	void task_update_progress_sample()
	{
		; /* skip */
	}

	void task_update_tile_sample(RenderTile&)
	{
		; /* skip */
	}

	void task_release_tile(RenderTile& tile)
	{
		thread_scoped_lock acquire_lock(acquire_mutex);

		if(tile.buffer) tile.buffer = ptr_imap[tile.buffer];
		if(tile.rng_state) tile.rng_state = ptr_imap[tile.rng_state];
		if(tile.rgba) tile.rgba = ptr_imap[tile.rgba];

		thread_scoped_lock lock(rpc_lock);
		RPCSend snd(socket, "release_tile");
		snd.add(tile);
		snd.write();
		lock.unlock();

		while(1) {
			lock.lock();
			RPCReceive rcv(socket);

			if(rcv.name == "release_tile")
				break;
			else
				process(rcv, lock);
		}
	}

	bool task_get_cancel()
	{
		return false;
	}

	/* properties */
	Device *device;
	RPCStreamManager rpc_stream;

	/* mapping of remote to local pointer */
	PtrMap ptr_map;
	PtrMap ptr_imap;
	DataMap mem_data;

	thread_mutex acquire_mutex;

	/* todo: free memory and device (osl) on network error */
};

void Device::server_run()
{
	try {
		/* starts thread that responds to discovery requests */
		ServerDiscovery discovery;

		for(;;) {
			/* accept connection */
			boost::asio::io_service io_service;
			tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), SERVER_PORT));

			tcp::socket socket(io_service);
			acceptor.accept(socket);

			string remote_address = socket.remote_endpoint().address().to_string();
			printf("Connected to remote client at: %s\n", remote_address.c_str());

			DeviceServer server(this, socket);
			server.listen();

			printf("Disconnected.\n");
		}
	}
	catch(exception& e) {
		fprintf(stderr, "Network server exception: %s\n", e.what());
	}
}

CCL_NAMESPACE_END

#endif


