import pandas as pd
import chardet
df = pd.read_csv("C:\\Users\\renxi\\Documents\\Datasets\\2018.csv",encoding='gb18030',low_memory=False)
# 对数据集进行删剪(删除店铺编号，店铺简称，店铺地址，颜色说明和尺码列)
df = df.drop(df.columns[[0,1,3,8,9]],axis = 1)
# 对异常值进行处理
df = df[(df.数量>0)&(df.销售价格>0)]
df = df.dropna(axis = 0)
# 将销售日期改为所对应的周数
df['销售日期'] = pd.to_datetime(df['销售日期']).apply(lambda x:x.strftime("%W"))
df.rename(columns={'销售日期':'销售周数'},inplace=True)
# 将吊牌价和销售价格的信息提取为折扣百分比
df['折扣百分比'] = df.apply(lambda x: (x[6]/x[5])*100, axis=1)
df = df.drop(df.columns[[6]],axis = 1)
In [5]:
# 删除总销量个位数的货号
group = df.groupby('货号')
df = group.filter(lambda x: len(x) > 9)
In [6]:
# 对于吊牌价和折扣百分比根据分布划分等地
Q1_price = df['吊牌价'].quantile(0.25)
Q2_price = df['吊牌价'].quantile(0.5)
Q3_price = df['吊牌价'].quantile(0.75)
def price_level (x):
    if x < Q1_price:
        pl = 0
    elif x < Q2_price:
        pl = 1
    elif x < Q3_price:
        pl = 2
    else:
        pl = 3
    return pl
df['价格等地'] = df.apply(lambda x: price_level(x[5]),axis=1)
Q1_disc = df['折扣百分比'].quantile(0.25)
Q2_disc = df['折扣百分比'].quantile(0.5)
Q3_disc = df['折扣百分比'].quantile(0.75)
def discount_level (x):
    if x < Q1_disc:
        dl = 0
    elif x < Q2_disc:
        dl = 1
    elif x < Q3_disc:
        dl = 2
    else:
        dl = 3
    return dl
df['折扣力度'] = df.apply(lambda x: discount_level(x[7]),axis=1)
# 尝试建立预测某一个品类在某一个城市的周销量的模型
df_1 = df[['店铺省市','货品名称','销售周数','颜色编号','数量']]
# 统计销量
df_1 = df_1.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_1 = df_1.reset_index()
In [8]:
# 用神经网络建立简单模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
X = pd.get_dummies(df_1[['店铺省市','货品名称','颜色编号']])
X[['销售周数']] =  df_1[['销售周数']]
y =  df_1['数量']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 30)
reg = MLPRegressor().fit(X_train,y_train)
score = reg.score(X_test, y_test)
# 依据价格等地将数据集分为两类（高价商品和低价商品）进行尝试
df_2 = df[['店铺省市','货品名称','销售周数','颜色编号','数量','价格等地']]
df_exp = df_2[df.价格等地 > 1]
df_cheap = df_2[df.价格等地 < 2]
df_exp = df_exp.drop(['价格等地'],axis = 1)
df_cheap = df_cheap.drop(['价格等地'],axis = 1)
# 统计销量
df_exp = df_exp.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_exp = df_exp.reset_index()
df_cheap = df_cheap.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_cheap = df_cheap.reset_index()
# 建立神经网络模型
X_exp = pd.get_dummies(df_exp[['店铺省市','货品名称','颜色编号']])
X_exp[['销售周数']] =  df_exp[['销售周数']]
y_exp =  df_exp['数量']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_exp,y_exp,test_size = 0.2, random_state = 30)
reg2 = MLPRegressor().fit(X_train2,y_train2)
score2 = reg2.score(X_test2, y_test2)
print("高价商品模型R^2 score:{}".format(score2))
X_cheap = pd.get_dummies(df_cheap[['店铺省市','货品名称','颜色编号']])
X_cheap[['销售周数']] =  df_cheap[['销售周数']]
y_cheap =  df_cheap['数量']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_cheap,y_cheap,test_size = 0.2, random_state = 30)
reg3 = MLPRegressor().fit(X_train3,y_train3)
score3 = reg3.score(X_test3, y_test3)
# 依据折扣力度将数据集分为两类进行尝试
df_3 = df[['店铺省市','货品名称','销售周数','颜色编号','数量','折扣力度']]
df_disbig = df_3[df.折扣力度 > 1]
df_dissma = df_3[df.折扣力度 < 2]
df_disbig = df_disbig.drop(['折扣力度'],axis = 1)
df_dissma = df_dissma.drop(['折扣力度'],axis = 1)
# 统计销量
df_disbig = df_disbig.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_disbig = df_disbig.reset_index()
df_dissma = df_dissma.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_dissma = df_dissma.reset_index()
# 建立神经网络模型
X_disbig = pd.get_dummies(df_disbig[['店铺省市','货品名称','颜色编号']])
X_disbig[['销售周数']] =  df_disbig[['销售周数']]
y_disbig =  df_disbig['数量']
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_disbig,y_disbig,test_size = 0.2, random_state = 30)
reg4 = MLPRegressor().fit(X_train4,y_train4)
score4 = reg4.score(X_test4, y_test4)
print("折扣力度小商品模型R^2 score:{}".format(score4))
X_dissma = pd.get_dummies(df_dissma[['店铺省市','货品名称','颜色编号']])
X_dissma[['销售周数']] =  df_dissma[['销售周数']]
y_dissma =  df_dissma['数量']
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_dissma,y_dissma,test_size = 0.2, random_state = 30)
reg5 = MLPRegressor().fit(X_train5,y_train5)
score5 = reg5.score(X_test5, y_test5)
# 折扣力度大商品模型score小的原因可能在于数据集总量太小，调整分类比例再次进行尝试
df_3 = df[['店铺省市','货品名称','销售周数','颜色编号','数量','折扣力度']]
df_disbig = df_3[df.折扣力度 > 2]
df_dissma = df_3[df.折扣力度 < 3]
df_disbig = df_disbig.drop(['折扣力度'],axis = 1)
df_dissma = df_dissma.drop(['折扣力度'],axis = 1)
# 统计销量
df_disbig = df_disbig.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_disbig = df_disbig.reset_index()
df_dissma = df_dissma.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_dissma = df_dissma.reset_index()
# 建立神经网络模型
X_disbig = pd.get_dummies(df_disbig[['店铺省市','货品名称','颜色编号']])
X_disbig[['销售周数']] =  df_disbig[['销售周数']]
y_disbig =  df_disbig['数量']
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_disbig,y_disbig,test_size = 0.2, random_state = 30)
reg4 = MLPRegressor().fit(X_train4,y_train4)
score4 = reg4.score(X_test4, y_test4)
print("折扣力度小商品模型R^2 score:{}".format(score4))
X_dissma = pd.get_dummies(df_dissma[['店铺省市','货品名称','颜色编号']])
X_dissma[['销售周数']] =  df_dissma[['销售周数']]
y_dissma =  df_dissma['数量']
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_dissma,y_dissma,test_size = 0.2, random_state = 30)
reg5 = MLPRegressor().fit(X_train5,y_train5)
score5 = reg5.score(X_test5, y_test5)
# 将价格等级和折扣力度作为参数考虑进去进行尝试
df_4 = df[['店铺省市','货品名称','销售周数','颜色编号','数量','价格等地','折扣力度']]
# 统计销量
df_4['价格等地'] = df_4.apply(lambda x : x[4]*float(x[5]),axis = 1)
df_4['折扣力度'] = df_4.apply(lambda x : x[4]*float(x[6]),axis = 1)
df_4 = df_4.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_4 = df_4.reset_index()
df_4['价格等地'] = df_4.apply(lambda x : x[5]/x[4],axis = 1)
df_4['折扣力度'] = df_4.apply(lambda x : x[6]/x[4],axis = 1)
X_all = pd.get_dummies(df_4[['店铺省市','货品名称','颜色编号']])
X_all[['销售周数','价格等地','折扣力度']] =  df_4[['销售周数','价格等地','折扣力度']]
y_all =  df_4['数量']
X_train6, X_test6, y_train6, y_test6 = train_test_split(X_all,y_all,test_size = 0.2, random_state = 30)
reg6 = MLPRegressor().fit(X_train6,y_train6)
score6 = reg6.score(X_test6, y_test6)
# 从上述实验来看，依据高价和低价将数据集分割成两部分对销售量的预测比较有益，
# 折扣力度小的商品在销售量上有一定规律可寻，但折扣力度大的商品销量不稳定性很大，对预测模型造成了干扰
In [45]:
# 依据同一颜色的销量定义颜色流行度，考虑进去进行尝试
df_5 = df[['店铺省市','货品名称','销售周数','颜色编号','数量']]
color_sum = df_5.groupby(['颜色编号']).sum()
fashion_avg = color_sum['数量'].quantile(0.7)
color_sum['流行与否'] = color_sum.apply(lambda x : x[0] > fashion_avg, axis = 1)
color_sum = color_sum.transpose()
# 依旧颜色流行与否将数据集分成两类进行尝试
df_6 = df[['店铺省市','货品名称','销售周数','颜色编号','数量']]
df_6['流行与否'] = df_6.apply(lambda x : color_sum[x[3]].流行与否, axis = 1)
df_fashion = df_6[df_6.流行与否]
df_notfash = df_6[~df_6.流行与否]
df_fashion = df_fashion.drop(['流行与否'],axis = 1)
df_notfash = df_notfash.drop(['流行与否'],axis = 1)
# 统计销量
df_fashion = df_fashion.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_fashion = df_fashion.reset_index()
df_notfash = df_notfash.groupby(['货品名称','店铺省市','销售周数','颜色编号']).sum()
df_notfash = df_notfash.reset_index()
# 建立神经网络模型
X_fashion = pd.get_dummies(df_fashion[['店铺省市','货品名称','颜色编号']])
X_fashion[['销售周数']] =  df_fashion[['销售周数']]
y_fashion =  df_fashion['数量']
X_train7, X_test7, y_train7, y_test7 = train_test_split(X_fashion,y_fashion,test_size = 0.2, random_state = 30)
reg7 = MLPRegressor().fit(X_train7,y_train7)
score7 = reg7.score(X_test7, y_test7)
print("颜色流行商品模型R^2 score:{}".format(score7))
X_notfash = pd.get_dummies(df_notfash[['店铺省市','货品名称','颜色编号']])
X_notfash[['销售周数']] =  df_notfash[['销售周数']]
y_notfash =  df_notfash['数量']
X_train8, X_test8, y_train8, y_test8 = train_test_split(X_notfash,y_notfash,test_size = 0.2, random_state = 30)
reg8 = MLPRegressor().fit(X_train8,y_train8)
score8 = reg8.score(X_test8, y_test8)
# 总结：
# 1. 将商品按照高价低价分割后对销售量预测有帮助
# 2. 折扣力度小和流行颜色的商品占大多数，销售量较稳定
# 3. 将价格赋予数值的等级作为vairable对销售量不是很有帮助，一个学习得到的weight不能很好地表现各等级间的差异，分割更有效一些
