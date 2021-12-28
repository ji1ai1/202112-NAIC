#define _CRT_SECURE_NO_WARNINGS

#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "基础.h"
#include "样本.h"
#include "预测器.h"

using namespace 番荔枝::特征编码;

void 读测试数据(
	std::vector<std::shared_ptr<类别_样本>>& 查询样本向量
	, std::vector<std::shared_ptr<类别_样本>>& 画廊样本向量
	, const std::string& 查询路径
	, const std::string& 画廊路径
)
{
	for (const auto& 档案 : std::filesystem::directory_iterator(查询路径))
		查询样本向量.push_back(std::make_shared<类别_样本>(查询路径, 档案.path().filename().string()));

	for (const auto& 档案 : std::filesystem::directory_iterator(画廊路径))
	{
		画廊样本向量.push_back(std::make_shared<类别_样本>(画廊路径, 档案.path().filename().string()));
		if (画廊样本向量.size() % 16384 == 0)
			std::cout << 取得时间() << "\t已读取" << 画廊样本向量.size() << "个画廊样本......" << std::endl;
	}
}

void 写预测至JSON(
	const std::string& 路径
	, const std::vector<std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>>& 预测向量
)
{
	std::ofstream 流(路径);
	流 << "{";
	auto 标记 = 0;
	for (const auto& 预测元组 : 预测向量)
	{
		if (标记 == 0)
			标记 = 1;
		else
			流 << ", ";
		流 << "\"" << std::get<0>(预测元组) << "\": [";

		auto 第二标记 = 0;
		for (auto& 预测文件名 : *std::get<1>(预测元组).get())
		{
			if (第二标记 == 0)
				第二标记 = 1;
			else
				流 << ", ";
			流 << "\"" << 预测文件名 << "\"";
		}
		流 << "]";
	}
	流 << "}";
	流.close();
}

int main(int 参数数量, char* 参数数组[])
{
	std::ios_base::sync_with_stdio(false);

	std::cout << 取得时间() << "\t开始......" << std::endl;

	std::vector<std::shared_ptr<类别_样本>> 测试查询样本向量;
	std::vector<std::shared_ptr<类别_样本>> 测试画廊样本向量;
	读测试数据(测试查询样本向量, 测试画廊样本向量, "C:/2021.12 特征编码/test_A/query_feature_A\\", "C:/2021.12 特征编码/test_A/gallery_feature_A\\");
	std::cout << 取得时间() << "\t已读取" << 测试查询样本向量.size() << "个画廊样本" << ", " << 测试画廊样本向量.size() << "个画廊样本......" << std::endl;

	std::vector<std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>> 预测向量;
	类别_预测器::预测(预测向量, 测试查询样本向量, 测试画廊样本向量);
	写预测至JSON("C:/2021.12 特征编码/20211228M.json", 预测向量);

	return 0;
}
