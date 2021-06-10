__author__ = 'zoulida'
import akshare as ak
get_futures_daily_df = ak.get_futures_daily(start_date="20200701", end_date="20200716", market="cffex", index_bar=True)
print(get_futures_daily_df)