import re
import os
import math
import time
import json
import warnings
import finnhub
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict
import nltk

nltk.download('vader_lexicon')
warnings.filterwarnings("ignore")


class StockInfoCrawler():
    def __init__(self):
        self.finnhub_client = finnhub.Client(api_key="clipok1r01qvsg59ere0clipok1r01qvsg59ereg")
        self.id2company_info = {}

    def get_company_profile(self, symbol):
        company_df = pd.read_csv(fr"./data/profile/symbol_to_profile.tsv", sep='\t',
                                 header=0, encoding='utf-8-sig')
        company2profile = dict(zip(company_df["symbol"], company_df["profile"]))
        company2profile = {k: json.loads(company2profile[k]) for k in company2profile}

        if symbol in company2profile:
            profile = company2profile[symbol]
        else:
            time.sleep(0.5)
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            detail  = {"symbol":  [symbol], "profile": [json.dumps(profile)]}
            pd.DataFrame(detail).to_csv(fr"./data/profile/symbol_to_profile.tsv", mode='a', sep = '\t',header=False ,index=False,encoding='utf-8-sig')
        return profile


class StockNewsFactorCrawler(StockInfoCrawler):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.finnhub_client = finnhub.Client(api_key="clipok1r01qvsg59ere0clipok1r01qvsg59ereg")
        self.DATA_DIR = f"./{self.start_date}_{self.end_date}"
        self.id2company_info = {}

    def prepare_data_for_company(self, symbol, with_basics=True):
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR, exist_ok=True)

        data = self.get_returns(symbol)
        data = self.get_news(symbol, data)
        data = self.get_company_profile(symbol, data)

        if with_basics:
            data = self.get_basics(symbol, data)
            data.to_csv(f"{self.DATA_DIR}/{symbol}_{self.start_date}_{self.end_date}.csv")
        else:
            data['Basics'] = [json.dumps({})] * len(data)
            data.to_csv(f"{self.DATA_DIR}/{symbol}_{self.start_date}_{self.end_date}_nobasics.csv")
        return data

    def get_company_profile(self, symbol, data):
        if symbol in self.id2company_info:
            profile = self.id2company_info[symbol]
        else:
            profile = self.finnhub_client.company_profile2(symbol=symbol)
        data["Profile"] = [json.dumps(profile)] * len(data)
        return data

    def get_returns(self, stock_symbol):
        def bin_mapping(ret):
            up_down = 'U' if ret >= 0 else 'D'
            integer = math.ceil(abs(100 * ret))
            return up_down + (str(integer) if integer <= 5 else '5+')

        # Download historical stock data
        stock_data = yf.download(stock_symbol, start=self.start_date, end=self.end_date)

        weekly_data = stock_data['Adj Close'].resample('W').ffill()
        weekly_returns = weekly_data.pct_change()[1:]
        weekly_start_prices = weekly_data[:-1]
        weekly_end_prices = weekly_data[1:]

        weekly_data = pd.DataFrame({
            'Start Date': weekly_start_prices.index,
            'Start Price': weekly_start_prices.values,
            'End Date': weekly_end_prices.index,
            'End Price': weekly_end_prices.values,
            'Weekly Returns': weekly_returns.values
        })

        weekly_data['Bin Label'] = weekly_data['Weekly Returns'].map(bin_mapping)
        return weekly_data

    def get_news(self, symbol, data):
        news_list = []
        for end_date, row in data.iterrows():
            start_date = row['Start Date'].strftime('%Y-%m-%d')
            end_date = row['End Date'].strftime('%Y-%m-%d')
            print(symbol, ': ', start_date, ' - ', end_date)
            time.sleep(1)  # control qpm
            weekly_news = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            weekly_news = [
                {
                    "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                    "headline": n['headline'],
                    "summary": n['summary'],
                } for n in weekly_news
            ]
            weekly_news.sort(key=lambda x: x['date'])
            news_list.append(json.dumps(weekly_news))

        data['News'] = news_list
        return data

    def get_basics(self, symbol, data, always=False):
        basic_financials = self.finnhub_client.company_basic_financials(symbol, 'all')

        final_basics, basic_list, basic_dict = [], [], defaultdict(dict)

        for metric, value_list in basic_financials['series']['quarterly'].items():
            for value in value_list:
                basic_dict[value['period']].update({metric: value['v']})

        for k, v in basic_dict.items():
            v.update({'period': k})
            basic_list.append(v)

        basic_list.sort(key=lambda x: x['period'])

        for i, row in data.iterrows():

            start_date = row['End Date'].strftime('%Y-%m-%d')
            last_start_date = self.start_date if i < 2 else data.loc[i - 2, 'Start Date'].strftime('%Y-%m-%d')

            used_basic = {}
            for basic in basic_list[::-1]:
                if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                    used_basic = basic
                    break
            final_basics.append(json.dumps(used_basic))

        data['Basics'] = final_basics

        return data


if __name__ == '__main__':
    pass
