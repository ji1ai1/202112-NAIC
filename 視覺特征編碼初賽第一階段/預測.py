import json
import pandas

# 預先用任意方式計算出餘弦相似度，儲存在「相似度.csv」档案中。毎箇query保留相似度最大的200箇gallery即可。
預測表 = pandas.read_csv("相似度.csv", header=None, names=["档案名", "預測档案名", "相似度"])
預測表["排名"] = 預測表.groupby("filename")["相似度"].rank(ascending=False)
預測表 = 預測表.loc[預測表.排名 <= 200]
預測表["相似度_16"] = 預測表.相似度 ** 16
預測表 = 預測表.merge(預測表.groupby("預測档案名").aggregate(預測档案相似度_16和=("相似度_16", "sum")), on="預測档案名")
預測表["打分"] = 預測表.相似度 / (1 + 預測表.預測档案相似度_16和 ** 0.03125) ** 4
預測表["排名"] = 預測表.groupby("filename").打分.rank(ascending=False, method="first")
預測表 = 預測表.loc[預測表.排名 <= 100].reset_index(drop=True)
預測表 = 預測表.sort_values(["档案名", "排名"])
with open("result.json", "w") as 档案:
	json.dump(預測表.groupby("档案名").apply(lambda 子: list(子.預測档案名)).to_dict(), 档案)
