#pragma once

#include <cstdlib>
#include <string>

namespace 番荔枝::特征編碼
{
	class 類別_樣本
	{
	public:
		std::string 檔案名;
		float 特征[2048];

		類別_樣本(const std::string& 路徑, const std::string& 檔案名) : 檔案名(檔案名)
		{
			auto 檔案 = fopen((路徑 + 檔案名).c_str(), "rb");
			fread(特征, 4, 2048, 檔案);
			fclose(檔案);
		}
	};
}
