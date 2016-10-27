# DC-猜你喜欢比赛
基于用户行为的推荐算法大赛---第四名（临兵斗列）

比赛平台：DataCastle

软件运行环境：Python 2.x

使用说明：文件夹包括数据以及代码，其中数据包括train和test两个数据集，其中test是用于提交到DC平台的测试集。代码包括三种方法测试的结果，最终第四名的成果采用的是深度学习。train数据集太大，请自行去官网下载【address：http://www.pkbigdata.com/】

代码详细说明：average_precision.py是利用用户的平均分进行的预测；
lfm.py采用的是LFM算法测试的结果；
DeepLearn.py采用的是深度学习方法，最终结果达到7.83398（10分为满分），取得第四名。
