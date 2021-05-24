__author__ = 'zoulida'



import datetime
import numpy as np
import scipy.stats as st
import baostock as bs
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

import mplfinance as mpf

# 登录系统
lg = bs.login()
# 显示登录返回信息
print(lg.error_code)
print(lg.error_msg)

# 平安银行股票代码
bs_code = '000001.SZ'
# 过去x days的数据
days = 50
end_date = datetime.date.today().strftime('%Y-%m-%d')
start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')

rs = bs.query_history_k_data(code=bs_code,
                             fields='code,date,open,high,low,close,peTTM,pbMRQ',  # 代码 日期 etc
                             start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")

# 获取具体的股价信息
stock_list = []
while (rs.error_code == '0') & rs.next():
    # 分页查询，将每页信息合并在一起
    stock_list.append(rs.get_row_data())
stock_data = pd.DataFrame(stock_list, columns=rs.fields)

# 如果写入文档可以启用下边这行
# stock_data.to_csv("stock_temp.csv", index=False)

# 获取原始数据中的部分数据，注意一定用loc，不要用iloc
show_data = stock_data.loc[:, ['date', 'open', 'high', 'low', 'close']]
# 原数据集中的日期强制修改为日期格式
show_data['date'] = pd.to_datetime(show_data['date'])
# 选取所有行的股票代码 开收盘价 以及最高低价，并转化为float类型（默认为string）
show_data.loc[:, ['open', 'high', 'low', 'close']] = show_data[['open', 'high', 'low', 'close']].astype(float)
# 将日期设定为index主键
show_data = show_data.set_index(['date'], drop=True)

print(show_data)

# 如果上文用到写入文档，这里就可以读取文档了 读取文档就不用上边这一堆罗里吧嗦的转换了
# show_data2 = pd.read_csv('stock_temp.csv', index_col=0, parse_dates=True, usecols=[1, 2, 3, 4, 5])

# 画K线图
mpf.plot(show_data, type='candle', style='yahoo')

