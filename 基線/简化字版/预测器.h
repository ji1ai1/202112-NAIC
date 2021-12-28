#pragma once

#include <iostream>
#include <thread>
#include <vector>

#include "基础.h"
#include "样本.h"

namespace 番荔枝::特征编码
{
	class 类别_预测器
	{
	public:
		static void 预测(
			std::vector<std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>>& 预测向量
			, const std::vector<std::shared_ptr<类别_样本>>& 查询样本向量
			, const std::vector<std::shared_ptr<类别_样本>>& 画廊样本向量
			, const std::vector<int>& 列向量
		)
		{
			const auto 线程数 = 28;
			std::thread 线程数组[线程数];
			std::vector<std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>> 预测向量数组[线程数];
			for (auto 子 = 0; 子 < 线程数; 子++)
			{
				线程数组[子] = std::thread([查询样本向量, 画廊样本向量, 列向量](auto 线程序号, auto 预测向量数组)
					{
						for (auto 子 = size_t(线程序号); 子 < 查询样本向量.size(); 子 += 线程数)
						{
							if (子 % 1024 == 0)
								std::cout << 取得时间() << "\t已预测" << 子 << "个样本......" << std::endl;

							auto& 样本 = *查询样本向量[子].get();

							std::vector<std::tuple<std::string, double>> 文件名相似度元组向量;
							for (auto 丑 = 0; 丑 < 画廊样本向量.size(); 丑++)
							{
								auto& 画廊样本 = *画廊样本向量[丑].get();
								auto 相似度 = 计算相似度(样本, 画廊样本, 列向量);

								文件名相似度元组向量.push_back(std::tuple<std::string, double>(画廊样本.文件名, 相似度));
							}
							std::sort(文件名相似度元组向量.begin(), 文件名相似度元组向量.end(), [](const auto& 子元组, const auto& 丑元组) { return std::get<1>(子元组) > std::get<1>(丑元组); });

							auto 预测向量指针 = std::make_shared<std::vector<std::string>>();
							for (auto 丑 = 0; 丑 < 100; 丑++)
								预测向量指针->push_back(std::get<0>(文件名相似度元组向量[丑]));
							预测向量数组[线程序号].push_back(std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>(样本.文件名, 预测向量指针));
						}
					}, 子, 预测向量数组);
			}

			for (auto 子 = 0; 子 < 线程数; 子++)
			{
				线程数组[子].join();
				for (auto 丑 : 预测向量数组[子])
					预测向量.push_back(丑);
			}
		}

	private:
		static double 计算相似度(const 类别_样本& 子样本, const 类别_样本& 丑样本, const std::vector<int>& 列向量)
		{
			auto 相似度分子 = 0.0;
			for (auto 子 : 列向量)
				相似度分子 += 子样本.特征[子] * 丑样本.特征[子];

			return 相似度分子 / sqrt(子样本.特征平方和 * 丑样本.特征平方和);
		}

		/*static double 计算相似度(const 类别_样本& 子样本, const 类别_样本& 丑样本, const std::vector<int>& 列向量)
		{
			auto 距离 = 0.0;
			for (auto 子 : 列向量)
			{
				auto 差 = 子样本.特征[子] - 丑样本.特征[子];
				距离 += 差 * 差;
			}
			return 1 / (1 + 距离);
		}*/
	};
}
