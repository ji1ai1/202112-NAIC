#pragma once

#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>

auto 取得時間()
{
	auto 時間 = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	return std::put_time(std::localtime(&時間), "%H:%M:%S");
}
