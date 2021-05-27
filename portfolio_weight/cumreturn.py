__author__ = 'zoulida'
# -*- coding: utf-8 -*-


import pandas as pd
import baostock as bs
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

#获取给定时间段的股票交易信息
def get_stock_data(t1,t2,stock_name):
    lg = bs.login()
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见baostock
    rs = bs.query_history_k_data(stock_name,
                                 "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                 start_date=t1, end_date=t2,
                                 frequency="d", adjustflag="3")
    print('query_history_k_data respond error_code:' + rs.error_code)
    print('query_history_k_data respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    print(result)

    #### 结果集输出到csv文件 ####
    result.to_csv("history_A_stock_k_data.csv", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()
    result['date'] = pd.to_datetime(result['date'])
    result.set_index("date", inplace=True)
    return result

byjc = get_stock_data('2014-1-1','2015-1-1','sh.600004')
hxyh = get_stock_data('2014-1-1','2015-1-1','sh.600015')
zndl = get_stock_data('2014-1-1','2015-1-1','sh.600023')
fjgs = get_stock_data('2014-1-1','2015-1-1','sh.600033')
sykj = get_stock_data('2014-1-1','2015-1-1','sh.600183')


by = byjc['pctChg']
by.name = 'byjc'
by = pd.DataFrame(by,dtype=np.float)/100


hx = hxyh['pctChg']
hx.name = 'hxyh'
hx = pd.DataFrame(hx,dtype=np.float)/100

zn = zndl['pctChg']
zn.name = 'zndl'
zn = pd.DataFrame(zn,dtype=np.float)/100

fj = fjgs['pctChg']
fj.name = 'fjgs'
fj = pd.DataFrame(fj,dtype=np.float)/100

sy = sykj['pctChg']
sy.name = 'sykj'
sy = pd.DataFrame(sy,dtype=np.float)/100

sh_return = pd.concat([by,fj,hx,sy,zn],axis=1)

#了解各股票的收益率状况,并作图
sh_return = sh_return.dropna()
cumreturn = (1+ sh_return).cumprod()
sh_return.plot()
plt.title('Daily Return of 5 Stocks(2014-2015)')
plt.legend(bbox_to_anchor = (0.5,-0.3),ncol = 5,fancybox = True,shadow = True)
cumreturn.plot()
plt.title('Cumulative Return of 5 stocks(2014-2015)')
plt.show()