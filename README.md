# (k,inf)-VCS的模拟退火实验
## 运行环境
- python 3.9
- 各种测试脚本(`*.ipynb`)需要安装`Jupyter Notebook`才能运行，执行以下命令安装。
```shell
pip install jupyter
```
- 依赖的其他库版本位于`requirements.txt`，执行以下命令安装。
```shell
pip install -r requirements.txt
```
## 文件说明

### optimize.py
实现了核心算法的文件，算法分别对应以下4个函数。
- optimize_nonliner. 调用`scipy.optimze`库的非线性规划函数实现，只能算出`k=2`时的解
- optimize_sa1. 一开始的模拟退火，只有一层循环进行迭代。已弃用。
- optimize_sa2. 模拟退火2，添加了马尔科夫链对应的第二层循环。
- optimize_sa3. 模拟退火3，在每次马尔可夫链循环结束后，更新部分超参数。

计算结果由对象`OptimizedResult`保存。

### opt_sa.ipynb
能看到生成图像效果的脚本

### *_stat.ipynb
对各种结果进行统计的脚本