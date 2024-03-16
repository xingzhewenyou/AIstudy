import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# https://blog.csdn.net/u012474716/article/details/80417909

# 'ex1.csv'的内容如下：
# a,b,c,d,message
# 1,2,3,4,hello
# 5,6,7,8,world
# 9,10,11,12,foo
df = pd.read_csv('ex1.csv')
print(df)
# 输出结果如下：
#    a   b   c   d message
# 0  1   2   3   4   hello
# 1  5   6   7   8   world
# 2  9  10  11  12     foo


# (3)读入文件可以让pandas为其分配默认的列名，也可以自己定义列名：
print(pd.read_csv('ex1.csv', header=None))
# 输出结果如下：
#    0   1   2   3        4
# 0  a   b   c   d  message
# 1  1   2   3   4    hello
# 2  5   6   7   8    world
# 3  9  10  11  12      foo
print(pd.read_csv('ex1.csv', names=['a', 'b', 'c', 'd', 'message']))
# 输出结果如下：
#    a   b   c   d  message
# 0  a   b   c   d  message
# 1  1   2   3   4    hello
# 2  5   6   7   8    world
# 3  9  10  11  12      foo

# （7）skiprows跳过文件的一些行，可以帮助处理各种各样的异形文件格式
print("7-------------7----------------------------------")
# 'ex4.csv'的内容如下：
##hey!
# a,b,c,d,message
# #just wanted to make thins more difficult for u
# # who reads CSV files with computers,anyway?
# 1,2,3,4,hello
# 5,6,7,8,world
# 9,10,11,12,foo
print(pd.read_csv('ex4.csv', skiprows=[0, 2, 3]))
# 输出结果如下：
#    a   b   c   d message
# 0  1   2   3   4   hello
# 1  5   6   7   8   world
# 2  9  10  11  12     foo


# 逐块读取文本文件:
# 在处理很大文件时，或找出大文件中的参数集以便于后续处理时，你可能只想读取文件的一小部分或逐块对文件进行迭代。
print("8-----8-------------------------")

# 'ex6.csv'的内容如下：
# <class 'pandas.core.frame.DataFrame'>
# Int64Index:10000 entries, 0 to 9999
# Data columns:
# one     10000    non-null       values
# two     10000    non-null       values
# three     10000    non-null       values
# four     10000    non-null       values
# key     10000    non-null       values
# dtypes: float64(4),object(1)
print(pd.read_csv('ex6.csv', nrows=5))  # nrows=5取前6行，下标从0开始

# 要逐块读取文件，需要设置chunksize(行数)
chunker = pd.read_csv('ex6.csv', chunksize=1000)
print(chunker)
# 输出结果如下：
# <pandas.io.parsers.TextFileReader object at 0x102ebb5d0>

# read_csv所返回的这个TextParser对象使你可以根据chunksize对文件进行逐块迭代。比如说：
# 我们可以迭代处理ex6.csv,将值计数聚合到"key"列中。
# tot = Series([])
# for piece in chunker:
#     tot = tot.add(piece['key'].value_counts(), fill_value=0)  # value_counts计算个数，fill_value为空时填充0
# tot = tot.order(ascending=False)  # 此版本Series没有有order,可以换成sort_value
# # tot=tot.sort_value(ascending=False)
# print(tot)  # 报key错误


# （1）Series的to_csv方法，将Series写入到.csv文件中
dates = pd.date_range('1/1/2000', periods=7)  # date_range可以生成时间序列，periods=7表示可以生成7个时间序列，从2000/1/1开始
print(dates)
# 输出结果如下：
# DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03', '2000-01-04',
#                '2000-01-05', '2000-01-06', '2000-01-07'],
#               dtype='datetime64[ns]', freq='D')
ts = Series(np.arange(7), index=dates)  # index行索引用dates
ts.to_csv('tseries.csv')
# tseries.csv的内容如下:
# 2000-01-01,0
# 2000-01-02,1
# 2000-01-03,2
# 2000-01-04,3
# 2000-01-05,4
# 2000-01-06,5
# 2000-01-07,6


# csv 参数选项如下：
# 参数             说明
# delimiter	    用于分隔字段的单字符字符串。默认为“，”
# lineterminator	用于写操作的行结束符，默认为“\r\n”
# quotechar		用于带有特殊字符(如分隔符)的字段的引用符号。默认为“"”
# quoting			引用约定。可选值包括csv.quote_all(引用所有字段)，
# 				csv.quote_minimal(只引用带有诸如分隔符之类特殊字符的字段)默认为quote_minimal
# skipinitialspace 忽略分隔符后面的空白符。默认False
# doublequote		 如何处理字段内的引用符号。如果为True,则双写。
# escapechar		 用于对分隔符进行转义的字符串。默认禁用
#
# 总结
#  （1）对于那些使用复杂分隔符或多字符分隔符的文件，csv模块就无能为力了。在这种情况下，就只能用字符串split方法或正则表达式方法re.split
# 进行拆分和其它整理工作了。
# （2）最后，如果你读取CSV数据的目的是做数据分析和统计的话，
# 你可能需要看一看 Pandas 包。Pandas 包含了一个非常方便的函数叫 pandas.read_csv() ，
# 它可以加载CSV数据到一个 DataFrame 对象中去。 然后利用这个对象你就可以生成各种形式的统计、过滤数据以及执行其他高级操作了



