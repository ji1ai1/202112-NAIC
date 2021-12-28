#define _CRT_SECURE_NO_WARNINGS

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
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
	读测试数据(测试查询样本向量, 测试画廊样本向量, "test_A/query_feature_A/", "test_A/gallery_feature_A/");
	std::cout << 取得时间() << "\t已读取" << 测试查询样本向量.size() << "个画廊样本" << ", " << 测试画廊样本向量.size() << "个画廊样本......" << std::endl;

	double 特征最小值数组[2048];
	double 特征最大值数组[2048];
	for (auto 子 = 0; 子 < 2048; 子++)
	{
		特征最小值数组[子] = INFINITY;
		特征最大值数组[子] = -INFINITY;
	}
	for (const auto& 样本 : 测试查询样本向量)
	{
		for (auto 子 = 0; 子 < 2048; 子++)
		{
			if (样本->特征[子] < 特征最小值数组[子])
				特征最小值数组[子] = 样本->特征[子];
			if (样本->特征[子] > 特征最大值数组[子])
				特征最大值数组[子] = 样本->特征[子];
		}
	}
	for (const auto& 样本 : 测试画廊样本向量)
	{
		for (auto 子 = 0; 子 < 2048; 子++)
		{
			if (样本->特征[子] < 特征最小值数组[子])
				特征最小值数组[子] = 样本->特征[子];
			if (样本->特征[子] > 特征最大值数组[子])
				特征最大值数组[子] = 样本->特征[子];
		}
	}
	std::vector<int> 列向量;
	for (auto 子 = 0; 子 < 2048; 子++)
	{
		if (特征最小值数组[子] != 特征最大值数组[子])
			列向量.push_back(子);
	}

	std::vector<std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>> 预测向量;
	类别_预测器::预测(预测向量, 测试查询样本向量, 测试画廊样本向量, 列向量);
	写预测至JSON("result.json", 预测向量);

	return 0;
}
