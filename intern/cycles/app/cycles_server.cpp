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

#include <stdio.h>

#include "device.h"

#include "util_args.h"
#include "util_foreach.h"
#include "util_path.h"
#include "util_stats.h"
#include "util_string.h"
#include "util_task.h"

using namespace ccl;

int main(int argc, const char **argv)
{
#ifndef NDEBUG
	/* force stdout to line buffered when debugging so if the dev is
	 * using an IDE that pipes stdout, it will not be fully buffered.
	 * Note that this *must* be done before stdout is used, so do it
	 * as soon as possible. */
	setvbuf(stdout, NULL, _IOLBF, 4096);
	setvbuf(stderr, NULL, _IOLBF, 4096);
	std::cout.sync_with_stdio();
	std::cerr.sync_with_stdio();
#endif

	path_init();

	/* device types */
	string devicelist = "";
	string devicename = "cpu";
	bool list = false;
	int threads = 0;

	vector<DeviceType>& types = Device::available_types();

	foreach(DeviceType type, types) {
		if(devicelist != "")
			devicelist += ", ";

		devicelist += Device::string_from_type(type);
	}

	/* parse options */
	ArgParse ap;

	ap.options ("Usage: cycles_server [options]",
		"--device %s", &devicename, ("Devices to use: " + devicelist).c_str(),
		"--list-devices", &list, "List information about all available devices",
		"--threads %d", &threads, "Number of threads to use",
		NULL);

	if(ap.parse(argc, argv) < 0) {
		fprintf(stderr, "%s\n", ap.geterror().c_str());
		ap.usage();
		exit(EXIT_FAILURE);
	}
	else if(list) {
		vector<DeviceInfo>& devices = Device::available_devices();

		printf("Devices:\n");

		foreach(DeviceInfo& info, devices) {
			printf("    %s%s\n",
				info.description.c_str(),
				(info.display_device)? " (display)": "");
		}

		exit(EXIT_SUCCESS);
	}

	/* find matching device */
	DeviceType device_type = Device::type_from_string(devicename.c_str());
	vector<DeviceInfo>& devices = Device::available_devices();
	DeviceInfo device_info;

	foreach(DeviceInfo& device, devices) {
		if(device_type == device.type) {
			device_info = device;
			break;
		}
	}

	TaskScheduler::init(threads);

	while(1) {
		Stats stats;
		Device *device = Device::create(device_info, stats, true);
		printf("Cycles Server with device: %s\n", device->info.description.c_str());
		device->server_run();
		delete device;
	}

	TaskScheduler::exit();

	return 0;
}

