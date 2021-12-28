#pragma once

#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>

auto 取得时间()
{
	auto 时间 = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	return std::put_time(std::localtime(&时间), "%H:%M:%S");
}
