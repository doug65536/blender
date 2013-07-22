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

#ifndef __UTIL_DEBUGTRACE_H__
#define __UTIL_DEBUGTRACE_H__

#include <iostream>
#include <sstream>
#include "util_thread.h"

CCL_NAMESPACE_BEGIN

class SyncOutputStream
{
	static ccl::thread_mutex stream_lock;
	mutable std::stringstream ss;

public:
	SyncOutputStream()
	{
	}

	~SyncOutputStream()
	{
		ccl::thread_scoped_lock lock(stream_lock);
		std::cout << std::hex << std::setw(sizeof(void*)/4) << thread::id() <<
				": " << ss.str() << std::endl << std::flush;
	}

	template<typename T>
	friend const SyncOutputStream &operator<<(const SyncOutputStream &s, const T &value)
	{
		s.ss << value;
		return s;
	}
};

CCL_NAMESPACE_END

#endif	// __UTIL_DEBUGTRACE_H__
