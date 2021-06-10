__author__ = 'zoulida'

import time
import akshare as ak
cffex_text = ak.match_main_contract(exchange="cffex")
while True:
    time.sleep(3)
    data = ak.futures_zh_spot(subscribe_list=cffex_text, market="FF")
    print(data)