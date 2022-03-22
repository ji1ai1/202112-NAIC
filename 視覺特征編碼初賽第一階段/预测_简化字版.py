import json
import pandas

# 预先用任意方式计算出余弦相似度，储存在「相似度.csv」档案中。毎个query保留相似度最大的200个gallery即可。
预测表 = pandas.read_csv("相似度.csv", header=None, names=["档案名", "预测档案名", "相似度"])
预测表["排名"] = 预测表.groupby("filename")["相似度"].rank(ascending=False)
预测表 = 预测表.loc[预测表.排名 <= 200]
预测表["相似度_16"] = 预测表.相似度 ** 16
预测表 = 预测表.merge(预测表.groupby("预测档案名").aggregate(预测档案相似度_16和=("相似度_16", "sum")), on="预测档案名")
预测表["打分"] = 预测表.相似度 / (1 + 预测表.预测档案相似度_16和 ** 0.03125) ** 4
预测表["排名"] = 预测表.groupby("filename").打分.rank(ascending=False, method="first")
预测表 = 预测表.loc[预测表.排名 <= 100].reset_index(drop=True)
预测表 = 预测表.sort_values(["档案名", "排名"])
with open("result.json", "w") as 档案:
   json.dump(预测表.groupby("档案名").apply(lambda 子: list(子.预测档案名)).to_dict(), 档案)

