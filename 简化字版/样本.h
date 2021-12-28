#pragma once

#include <cstdlib>
#include <string>

namespace 番荔枝::特征编码
{
	class 类别_样本
	{
	public:
		std::string 文件名;
		float 特征[2048];

		类别_样本(const std::string& 路径, const std::string& 文件名) : 文件名(文件名)
		{
			auto 档案 = fopen((路径 + 文件名).c_str(), "rb");
			fread(特征, 4, 2048, 档案);
			fclose(档案);
		}
	};
}

