#define _CRT_SECURE_NO_WARNINGS

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "基礎.h"
#include "樣本.h"
#include "預測器.h"

using namespace 番荔枝::特征編碼;

void 讀測試資料(
	std::vector<std::shared_ptr<類別_樣本>>& 查詢樣本向量
	, std::vector<std::shared_ptr<類別_樣本>>& 畫廊樣本向量
	, const std::string& 查詢路徑
	, const std::string& 畫廊路徑
)
{
	for (const auto& 檔案 : std::filesystem::directory_iterator(查詢路徑))
		查詢樣本向量.push_back(std::make_shared<類別_樣本>(查詢路徑, 檔案.path().filename().string()));

	for (const auto& 檔案 : std::filesystem::directory_iterator(畫廊路徑))
	{
		畫廊樣本向量.push_back(std::make_shared<類別_樣本>(畫廊路徑, 檔案.path().filename().string()));
		if (畫廊樣本向量.size() % 16384 == 0)
			std::cout << 取得時間() << "\t已讀取" << 畫廊樣本向量.size() << "箇畫廊樣本......" << std::endl;
	}
}

void 寫預測至JSON(
	const std::string& 路徑
	, const std::vector<std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>>& 預測向量
)
{
	std::ofstream 流(路徑);
	流 << "{";
	auto 標記 = 0;
	for(const auto& 預測元組 : 預測向量)
	{
		if (標記 == 0)
			標記 = 1;
		else
			流 << ", ";
		流 << "\"" << std::get<0>(預測元組) << "\": [";

		auto 第二標記 = 0;
		for (auto& 預測檔案名 : *std::get<1>(預測元組).get())
		{
			if (第二標記 == 0)
				第二標記 = 1;
			else
				流 << ", ";
			流 << "\"" << 預測檔案名 << "\"";
		}
		流 << "]";
	}
	流 << "}";
	流.close();
}


int main(int 引數數量, char* 引數陣列[])
{
	std::ios_base::sync_with_stdio(false);

	std::cout << 取得時間() << "\t開始......" << std::endl;

	std::vector<std::shared_ptr<類別_樣本>> 測試查詢樣本向量;
	std::vector<std::shared_ptr<類別_樣本>> 測試畫廊樣本向量;
	讀測試資料(測試查詢樣本向量, 測試畫廊樣本向量, "test_A/query_feature_A/", "test_A/gallery_feature_A/");
	std::cout << 取得時間() << "\t已讀取" << 測試查詢樣本向量.size() << "箇畫廊樣本" << ", " << 測試畫廊樣本向量.size()  << "箇畫廊樣本......" << std::endl;

	std::vector<std::tuple<std::string, std::shared_ptr<std::vector<std::string>>>> 預測向量;
	類別_預測器::預測(預測向量, 測試查詢樣本向量, 測試畫廊樣本向量);
	寫預測至JSON("result.json", 預測向量);

	return 0;
}
