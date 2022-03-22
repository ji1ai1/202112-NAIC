### [AI+視覺特征編碼](https://www.heywhale.com/org/2021NAIC/competition/area/61b81042902a13001708eb17/content)
<br/><br/>
在余弦相似度的基础上，做了一个简单的后处理。每个query和gallery的打分，等于它们间的余弦相似度，除以该gallery的余弦相似度的16范数的平方加1的4次方。其中16和4都是经过线下测试选择的。
