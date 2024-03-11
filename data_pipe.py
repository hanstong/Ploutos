# encoding=utf-8
# !/usr/local/bin/python
import os
import io
import json
import numpy as np
from datetime import datetime, timedelta, date
import random
import time
import pandas as pd
from data_util import bin_mapping, map_bin_label
from crawler import StockInfoCrawler
from multiprocessing import Pool


class DataPipe:

    def __init__(self, config):
        # load path
        self.config = config
        self.movement_path = config.movement
        self.tweet_path = config.preprocessed
        self.vocab_path = config.vocab
        self.glove_path = config.glove

        # load dates
        self.train_start_date = config.dates['train_start_date']
        self.train_end_date = config.dates['train_end_date']
        self.dev_start_date = config.dates['dev_start_date']
        self.dev_end_date = config.dates['dev_end_date']
        self.test_start_date = config.dates['test_start_date']
        self.test_end_date = config.dates['test_end_date']

        # load model config
        self.batch_size = config.config_model['batch_size']
        self.shuffle = config.config_model['shuffle']

        self.max_n_days = config.config_model['max_n_days']
        self.max_n_words = config.config_model['max_n_words']
        self.max_n_msgs = config.config_model['max_n_msgs']

        self.word_embed_type = config.config_model['word_embed_type']
        self.word_embed_size = config.config_model['word_embed_size']
        self.init_stock_with_word = config.config_model['init_stock_with_word']
        self.price_embed_size = config.config_model['word_embed_size']
        self.y_size = config.config_model['y_size']

        assert self.word_embed_type in ('rand', 'glove')

    @staticmethod
    def _convert_token_to_id(token, token_id_dict):
        if token not in token_id_dict:
            token = 'UNK'
        return token_id_dict[token]

    def _get_start_end_date(self, phase):
        """
            phase: train, dev, test, unit_test
            => start_date & end_date
        """
        assert phase in {'train', 'dev', 'test', 'whole', 'unit_test'}
        if phase == 'train':
            return self.train_start_date, self.train_end_date
        elif phase == 'dev':
            return self.dev_start_date, self.dev_end_date
        elif phase == 'test':
            return self.test_start_date, self.test_end_date
        elif phase == 'whole':
            return self.train_start_date, self.test_end_date
        else:
            return '2012-07-23', '2012-08-05'  # '2014-07-23', '2014-08-05'

    def _get_batch_size(self, phase):
        """
            phase: train, dev, test, unit_test
        """
        if phase == 'train':
            return self.batch_size
        elif phase == 'unit_test':
            return 5
        else:
            return 1

    def index_token(self, token_list, key='id', type='word'):
        assert key in ('id', 'token')
        assert type in ('word', 'stock')
        indexed_token_dict = dict()

        if type == 'word':
            token_list_cp = list(token_list)  # un-change the original input
            token_list_cp.insert(0, 'UNK')  # for unknown tokens
        else:
            token_list_cp = token_list

        if key == 'id':
            for id in range(len(token_list_cp)):
                indexed_token_dict[id] = token_list_cp[id]
        else:
            for id in range(len(token_list_cp)):
                indexed_token_dict[token_list_cp[id]] = id

        # id_token_dict = dict(zip(token_id_dict.values(), token_id_dict.keys()))
        return indexed_token_dict

    def build_stock_id_word_id_dict(self):
        # load vocab, user, stock list
        stock_id_word_id_dict = dict()

        vocab_id_dict = self.index_token(self.config.vocab, key='token')
        id_stock_dict = self.index_token(self.config.stock_symbols, type='stock')

        for (stock_id, stock_symbol) in id_stock_dict.items():
            stock_symbol = stock_symbol.lower()
            if stock_symbol in vocab_id_dict:
                stock_id_word_id_dict[stock_id] = vocab_id_dict[stock_symbol]
            else:
                stock_id_word_id_dict[stock_id] = None
        return stock_id_word_id_dict

    def _convert_words_to_ids(self, words, vocab_id_dict):
        """
            Replace each word in the data set with its index in the dictionary

        :param words: words in tweet
        :param vocab_id_dict: dict, vocab-id
        :return:
        """
        return [self._convert_token_to_id(w, vocab_id_dict) for w in words]

    def _get_prices_and_ts(self, ss, main_target_date):

        def _get_mv_class(data, use_one_hot=False):
            mv = float(data[1])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0
                else:
                    return [0.0, 1.0] if use_one_hot else 1

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2

        def _get_y(data):
            return _get_mv_class(data, use_one_hot=True)

        def _get_prices(data):
            return [float(p) for p in data[3:6]]

        def _get_mv_percents(data):
            return _get_mv_class(data)

        ts, ys, prices, mv_percents, main_mv_percent = list(), list(), list(), list(), 0.0
        d_t_min = main_target_date - timedelta(days=self.max_n_days - 1)

        stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(ss))
        with io.open(stock_movement_path, 'r', encoding='utf8') as movement_f:
            for line in movement_f:  # descend
                data = line.split('\t')
                t = datetime.strptime(data[0], '%Y-%m-%d').date()
                # logger.info(t)
                if t == main_target_date:
                    # logger.info(t) 
                    ts.append(t)
                    ys.append(_get_y(data))
                    main_mv_percent = data[1]
                    if -0.005 <= float(main_mv_percent) < 0.0055:  # discard sample with low movement percent
                        return None
                if d_t_min <= t < main_target_date:
                    ts.append(t)
                    ys.append(_get_y(data))
                    prices.append(_get_prices(data))  # high, low, close
                    mv_percents.append(_get_mv_percents(data))
                if t < d_t_min:  # one additional line for x_1_prices. not a referred trading day
                    prices.append(_get_prices(data))
                    mv_percents.append(_get_mv_percents(data))
                    break

        T = len(ts)
        if len(ys) != T or len(prices) != T or len(mv_percents) != T:  # ensure data legibility
            return None

        # ascend
        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        prices_and_ts = {
            'T': T,  # 天数
            'ts': ts,  # 具体的日期
            'ys': ys,  # 数据的label, high,low,close
            'main_mv_percent': main_mv_percent,  # target 日期的上升下降比例
            'mv_percents': mv_percents,  # 其他日期的上升下降情况
            'prices': prices,  # 价格
        }

        return prices_and_ts

    def _get_unaligned_corpora(self, ss, main_target_date, vocab_id_dict):

        def get_ss_index(word_seq, ss):
            ss = ss.lower()
            ss_index = len(word_seq) - 1  # init
            if ss in word_seq:  # the shape of word_seq?
                ss_index = word_seq.index(ss)
            else:
                if '$' in word_seq:
                    dollar_index = word_seq.index('$')
                    if dollar_index is not len(word_seq) - 1 and ss in word_seq[dollar_index + 1]:  # 这里在else后面，能满足？
                        ss_index = dollar_index + 1
                    else:
                        for index in range(dollar_index + 1, len(word_seq)):
                            if ss in word_seq[index]:
                                ss_index = index
                                break
            return ss_index

        unaligned_corpora = list()  # list of sets: (d, msgs, ss_indices)
        stock_tweet_path = os.path.join(str(self.tweet_path), ss)

        d_d_max = main_target_date - timedelta(days=1)
        d_d_min = main_target_date - timedelta(days=self.max_n_days)

        d = d_d_max  # descend
        while d >= d_d_min:
            msg_fp = os.path.join(stock_tweet_path, d.isoformat())
            if os.path.exists(msg_fp):
                word_mat = np.zeros([self.max_n_msgs, self.max_n_words], dtype=np.int32)
                n_word_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                ss_index_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                msg_id = 0
                with open(msg_fp, 'r') as tweet_f:
                    for line in tweet_f:
                        msg_dict = json.loads(line)
                        text = msg_dict['text']
                        if not text:
                            continue

                        words = text[:self.max_n_words]
                        word_ids = self._convert_words_to_ids(words, vocab_id_dict)
                        n_words = len(word_ids)

                        n_word_vec[msg_id] = n_words
                        word_mat[msg_id, :n_words] = word_ids
                        ss_index_vec[msg_id] = get_ss_index(words, ss)  # important indicator

                        msg_id += 1
                        if msg_id == self.max_n_msgs:
                            break
                corpus = [d, word_mat[:msg_id], ss_index_vec[:msg_id], n_word_vec[:msg_id],
                          msg_id]  # need check on why :msg_id
                unaligned_corpora.append(corpus)
            d -= timedelta(days=1)

        unaligned_corpora.reverse()  # ascend
        return unaligned_corpora

    def _trading_day_alignment(self, ts, T, unaligned_corpora):
        aligned_word_tensor = np.zeros([T, self.max_n_msgs, self.max_n_words], dtype=np.int32)
        aligned_ss_index_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)
        aligned_n_words_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)
        aligned_n_msgs_vec = np.zeros([T, ], dtype=np.int32)

        # list for gathering
        aligned_msgs = [[] for _ in range(T)]
        aligned_ss_indices = [[] for _ in range(T)]
        aligned_n_words = [[] for _ in range(T)]
        aligned_n_msgs = [[] for _ in range(T)]

        corpus_t_indices = []
        max_threshold = 0

        for corpus in unaligned_corpora:
            d = corpus[0]
            for t in range(T):
                if d < ts[t]:
                    corpus_t_indices.append(t)
                    break

        assert len(corpus_t_indices) == len(unaligned_corpora)

        for i in range(len(unaligned_corpora)):
            corpus, t = unaligned_corpora[i], corpus_t_indices[i]
            word_mat, ss_index_vec, n_word_vec, n_msgs = corpus[1:]
            aligned_msgs[t].extend(word_mat)
            aligned_ss_indices[t].extend(ss_index_vec)
            aligned_n_words[t].append(n_word_vec)
            aligned_n_msgs[t].append(n_msgs)

        def is_eligible():
            n_fails = len([0 for n_msgs in aligned_n_msgs if sum(n_msgs) == 0])
            return n_fails <= max_threshold

        if not is_eligible():
            return None

        # gather into nd_array and clip exceeded part
        for t in range(T):
            n_msgs = sum(aligned_n_msgs[t])

            if aligned_msgs[t] and aligned_ss_indices[t] and aligned_n_words[t]:
                msgs, ss_indices, n_word = np.vstack(aligned_msgs[t]), np.hstack(aligned_ss_indices[t]), np.hstack(
                    aligned_n_words[t])
                assert len(msgs) == len(ss_indices) == len(n_word)
                n_msgs = min(n_msgs, self.max_n_msgs)  # clip length
                aligned_n_msgs_vec[t] = n_msgs
                aligned_word_tensor[t, :n_msgs] = msgs[:n_msgs]
                aligned_ss_index_mat[t, :n_msgs] = ss_indices[:n_msgs]
                aligned_n_words_mat[t, :n_msgs] = n_word[:n_msgs]

        aligned_info_dict = {
            'msgs': aligned_word_tensor,
            'ss_indices': aligned_ss_index_mat,
            'n_words': aligned_n_words_mat,
            'n_msgs': aligned_n_msgs_vec,
        }

        return aligned_info_dict

    def sample_gen_from_one_stock(self, vocab_id_dict, stock_id_dict, s, phase):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """
        start_date, end_date = self._get_start_end_date(phase)
        stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(s))
        main_target_dates = []

        with open(stock_movement_path, 'r') as movement_f:
            for line in movement_f:
                data = line.split('\t')
                main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()
                main_target_date_str = main_target_date.isoformat()

                if start_date <= main_target_date_str < end_date:
                    main_target_dates.append(main_target_date)

        if self.shuffle:  # shuffle data
            random.shuffle(main_target_dates)

        for main_target_date in main_target_dates:
            # logger.info('start _get_unaligned_corpora')
            unaligned_corpora = self._get_unaligned_corpora(s, main_target_date, vocab_id_dict)
            # logger.info('start _get_prices_and_ts')
            prices_and_ts = self._get_prices_and_ts(s, main_target_date)
            if not prices_and_ts:
                continue

            # logger.info('start _trading_day_alignment')
            aligned_info_dict = self._trading_day_alignment(prices_and_ts['ts'], prices_and_ts['T'], unaligned_corpora)
            if not aligned_info_dict:
                continue

            sample_dict = {
                # meta info
                'stock': self._convert_token_to_id(s, stock_id_dict),  # 股票代码
                'main_target_date': main_target_date.isoformat(),  # 主要的预测日期
                'T': prices_and_ts['T'],  # 周期天数，由处理的数据生成，但貌似也是固定的？
                # target
                'ys': prices_and_ts['ys'],  # 数据的label
                'main_mv_percent': prices_and_ts['main_mv_percent'],  # 主要日期的涨幅
                'mv_percents': prices_and_ts['mv_percents'],  # 其他辅助日期的涨幅 需要搞清楚这里的涨幅是单日的吗？
                # source
                'prices': prices_and_ts['prices'],  # 周期里每天的价格
                'msgs': aligned_info_dict['msgs'],  # 周天里每天的新闻
                'ss_indices': aligned_info_dict['ss_indices'],  # 关键词的位置
                'n_words': aligned_info_dict['n_words'],  # word 和message为什么要分开？
                'n_msgs': aligned_info_dict['n_msgs'],
            }

            yield sample_dict

    def batch_gen(self, phase):
        batch_size = self._get_batch_size(phase)
        # prepare vocab, user, stock dict
        vocab_id_dict = self.index_token(self.config.vocab, key='token')
        stock_id_dict = self.index_token(self.config.stock_symbols, key='token', type='stock')
        generators = [self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase) for s in self.config.stock_symbols]
        # logger.info('{0} Generators prepared...'.format(len(generators)))

        while True:
            # start_time = time.time()

            stock_batch = np.zeros([batch_size, ], dtype=np.int32)  # ?
            T_batch = np.zeros([batch_size, ], dtype=np.int32)  # ?
            y_batch = np.zeros([batch_size, self.max_n_days, self.y_size], dtype=np.float32)  # ?

            main_mv_percent_batch = np.zeros([batch_size, ], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size, self.max_n_days], dtype=np.float32)
            price_batch = np.zeros([batch_size, self.max_n_days, 3], dtype=np.float32)

            word_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs, self.max_n_words], dtype=np.int32)
            ss_index_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs],
                                      dtype=np.int32)  # what's ss_index for?

            n_msgs_batch = np.zeros([batch_size, self.max_n_days], dtype=np.int32)
            n_words_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)

            sample_id = 0
            while sample_id < batch_size:
                gen_id = random.randint(0, len(generators) - 1)
                try:
                    sample_dict = next(generators[gen_id])
                    T = sample_dict['T']
                    # meta
                    stock_batch[sample_id] = sample_dict['stock']
                    T_batch[sample_id] = T
                    # target
                    y_batch[sample_id, :T] = sample_dict['ys']
                    main_mv_percent_batch[sample_id] = sample_dict['main_mv_percent']
                    mv_percent_batch[sample_id, :T] = sample_dict['mv_percents']
                    # source
                    price_batch[sample_id, :T] = sample_dict['prices']
                    word_batch[sample_id, :T] = sample_dict['msgs']
                    ss_index_batch[sample_id, :T] = sample_dict['ss_indices']
                    n_msgs_batch[sample_id, :T] = sample_dict['n_msgs']
                    n_words_batch[sample_id, :T] = sample_dict['n_words']

                    sample_id += 1
                except StopIteration:
                    del generators[gen_id]
                    if generators:
                        continue
                    else:
                        return

            batch_dict = {
                # meta
                'batch_size': sample_id,
                'stock_batch': stock_batch,
                'T_batch': T_batch,
                # target
                'y_batch': y_batch,  # label: up, down or close?
                'main_mv_percent_batch': main_mv_percent_batch,  # only the final day?
                'mv_percent_batch': mv_percent_batch,  # everyday
                # source
                'price_batch': price_batch,
                'word_batch': word_batch,
                'ss_index_batch': ss_index_batch,
                'n_msgs_batch': n_msgs_batch,
                'n_words_batch': n_words_batch,
            }
            yield batch_dict

    def batch_gen_by_stocks(self, phase):
        batch_size = 2000
        vocab_id_dict = self.index_token(self.config.vocab, key='token')
        stock_id_dict = self.index_token(self.config.stock_symbols, key='token', type='stock')

        for s in self.config.stock_symbols:
            gen = self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase)

            stock_batch = np.zeros([batch_size, ], dtype=np.int32)
            T_batch = np.zeros([batch_size, ], dtype=np.int32)
            n_msgs_batch = np.zeros([batch_size, self.max_n_days], dtype=np.int32)
            n_words_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            y_batch = np.zeros([batch_size, self.max_n_days, self.y_size], dtype=np.float32)
            price_batch = np.zeros([batch_size, self.max_n_days, 3], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size, self.max_n_days], dtype=np.float32)
            word_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs, self.max_n_words], dtype=np.int32)
            ss_index_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            main_mv_percent_batch = np.zeros([batch_size, ], dtype=np.float32)

            sample_id = 0
            while True:
                try:
                    sample_info_dict = next(gen)
                    T = sample_info_dict['T']

                    # meta
                    stock_batch[sample_id] = sample_info_dict['stock']
                    T_batch[sample_id] = sample_info_dict['T']
                    # target
                    y_batch[sample_id, :T] = sample_info_dict['ys']
                    main_mv_percent_batch[sample_id] = sample_info_dict['main_mv_percent']
                    mv_percent_batch[sample_id, :T] = sample_info_dict['mv_percents']
                    # source
                    price_batch[sample_id, :T] = sample_info_dict['prices']
                    word_batch[sample_id, :T] = sample_info_dict['msgs']
                    ss_index_batch[sample_id, :T] = sample_info_dict['ss_indices']
                    n_msgs_batch[sample_id, :T] = sample_info_dict['n_msgs']
                    n_words_batch[sample_id, :T] = sample_info_dict['n_words']

                    sample_id += 1
                except StopIteration:
                    break

            n_sample_threshold = 1
            if sample_id < n_sample_threshold:
                continue

            batch_dict = {
                # meta
                's': s,
                'batch_size': sample_id,
                'stock_batch': stock_batch[:sample_id],
                'T_batch': T_batch[:sample_id],
                # target
                'y_batch': y_batch[:sample_id],
                'main_mv_percent_batch': main_mv_percent_batch[:sample_id],
                'mv_percent_batch': mv_percent_batch[:sample_id],
                # source
                'price_batch': price_batch[:sample_id],
                'word_batch': word_batch[:sample_id],
                'ss_index_batch': ss_index_batch[:sample_id],
                'n_msgs_batch': n_msgs_batch[:sample_id],
                'n_words_batch': n_words_batch[:sample_id],
            }

            yield batch_dict

    def sample_mv_percents(self, phase):
        main_mv_percents = []
        for s in self.config.stock_symbols:
            start_date, end_date = self._get_start_end_date(phase)
            stock_mv_path = os.path.join(str(self.movement_path), '{}.txt'.format(s))
            main_target_dates = []

            with open(stock_mv_path, 'r') as movement_f:
                for line in movement_f:
                    data = line.split('\t')
                    main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()
                    main_target_date_str = main_target_date.isoformat()

                    if start_date <= main_target_date_str < end_date:
                        main_target_dates.append(main_target_date)

            for main_target_date in main_target_dates:
                prices_and_ts = self._get_prices_and_ts(s, main_target_date)
                if not prices_and_ts:
                    continue
                main_mv_percents.append(prices_and_ts['main_mv_percent'])

        return main_mv_percents

    def init_word_table(self):
        t1 = time.time()
        word_table_init = np.random.random((self.config.vocab_size, self.word_embed_size)) * 2 - 1  # [-1.0, 1.0]

        if self.word_embed_type is not 'rand':
            n_replacement = 0
            vocab_id_dict = self.index_token(self.config.vocab, key='token')

            with io.open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tuples = line.split()
                    word, embed = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
                    if word in ['<unk>', 'unk']:  # unify UNK
                        word = 'UNK'
                    if word in vocab_id_dict:
                        n_replacement += 1
                        word_id = vocab_id_dict[word]
                        word_table_init[word_id] = embed

        print("cost time for loading word embedding:{:.2f} secs".format(time.time() - t1))

        return word_table_init


class DataPipePrompt(DataPipe):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.st_crawler = StockInfoCrawler()

    def sample_gen_from_one_stock(self, vocab_id_dict, stock_id_dict, s, phase):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """
        start_date, end_date = self._get_start_end_date(phase)
        stock_movement_path = os.path.join(str(self.movement_path), '{}.csv'.format(s))
        stock_df = pd.read_csv(stock_movement_path, sep=',', header=0)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y-%m-%d')
        stock_df = stock_df.sort_values(by='Date', ascending=False)
        stock_df['Date'] = stock_df['Date'].apply(lambda x: x.strftime("%Y-%m-%d"))
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        filtered_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] < end_date)]
        main_target_dates = filtered_df['Date'].dt.date.tolist()

        if self.shuffle:  # shuffle data
            random.shuffle(main_target_dates)

        sample_list = []
        for main_target_date in main_target_dates:
            # logger.info('start ')
            unaligned_corpora_text = self._get_unaligned_corpora_text(s, main_target_date, vocab_id_dict)
            prices_and_ts = self._get_prices_and_ts(s, main_target_date)
            if not prices_and_ts:
                continue

            if self.args.alignment == 1:
                unaligned_corpora = self._get_unaligned_corpora(s, main_target_date, vocab_id_dict)
                aligned_info_dict = self._trading_day_alignment(prices_and_ts['ts'], prices_and_ts['T'], unaligned_corpora)
                if not aligned_info_dict:
                    continue
            try:
                symbol_map = {'RDS-B': 'SHEL', 'PTR': 'PRCH', 'TOT': 'TTE', 'SNP': '', 'BBL': '', 'UN': '',
                              'CELG': 'BMY', 'PCLN': 'BKNG', 'HRG': 'SPB', 'PICO': 'VWTR', 'UTX': 'RTX', 'ABB': 'ABBNY',
                              'FB': 'META', 'CHL': ''}
                if s in symbol_map and symbol_map[s] == "":
                    company_prompt = "No Company Introduction"
                else:
                    company_symbol = symbol_map[s] if s in symbol_map else s
                    cm_info = self.st_crawler.get_company_profile(company_symbol)
                    company_template = "{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
                                       "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."
                    company_prompt = company_template.format(**cm_info)
            except:
                print("No company info", s)
                company_prompt = "No Company Introduction"

            news = ""
            news_list = []

            if self.args.dedup_news==1:
                existed_news = set()
                for d in unaligned_corpora_text:
                    for m in d:
                        if m[1] not in existed_news:
                            news += f"[{m[0]}]:{m[1]}"
                            news += "\n"
                            news_list.append(m)
                            existed_news.add(m[1])
            else:
                for d in unaligned_corpora_text:
                    for m in d:
                        news += f"[{m[0]}]:{m[1]}"
                        news += "\n"
                        news_list.append(m)

            if len(news_list) == 0:
                news = "No important company related news in this period"

            sample_dict = {
                # meta info
                'STOCK_NAME': s,
                'TARGET_DAY': main_target_date,
            }
            sample_dict.update(prices_and_ts)
            sample_dict.update({'COMPANY_NEWS': news,
                                'COMPANY_NEWS_LIST': news_list,
                                'COMPANY_INFO': company_prompt, })
            sample_list.append(sample_dict)
        return sample_list

    def batch_gen(self, phase):
        # prepare vocab, user, stock dict
        vocab_id_dict = self.index_token(self.config.vocab, key='token')
        stock_id_dict = self.index_token(self.config.stock_symbols, key='token', type='stock')
        generators = [self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase) for s in self.config.stock_symbols]
        # logger.info('{0} Generators prepared...'.format(len(generators)))
        while True:
            gen_id = random.randint(0, len(generators) - 1)
            try:
                sample_dict = next(generators[gen_id])
            except StopIteration:
                del generators[gen_id]
                if generators:
                    continue
                else:
                    return
            yield sample_dict

    def batch_gen_by_stocks(self, phase):
        vocab_id_dict = self.index_token(self.config.vocab, key='token')
        stock_id_dict = self.index_token(self.config.stock_symbols, key='token', type='stock')

        for s in self.config.stock_symbols:
            gen = self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase)
            sample_id = 0
            while True:
                try:
                    sample_info_dict = next(gen)
                    sample_id += 1
                except StopIteration:
                    break
                yield sample_info_dict

    def _get_unaligned_corpora_text(self, ss, main_target_date, vocab_id_dict):
        unaligned_corpora = list()  # list of sets: (d, msgs, ss_indices)
        stock_tweet_path = os.path.join(str(self.tweet_path), ss)

        d_d_max = main_target_date - timedelta(days=1)
        d_d_min = main_target_date - timedelta(days=self.max_n_days)

        d = d_d_max  # descend
        while d >= d_d_min:
            msg_fp = os.path.join(stock_tweet_path, d.isoformat())
            if os.path.exists(msg_fp):
                msg_id = 0
                msg_list = []
                with open(msg_fp, 'r') as tweet_f:
                    try:
                        for line in tweet_f:
                            msg_dict = json.loads(line)
                            text = msg_dict['text']
                            if not text:
                                continue
                            words = text[:self.max_n_words]
                            words = " ".join(words)
                            msg_list.append((d.isoformat(), words))

                            msg_id += 1
                            if msg_id == self.max_n_msgs:
                                break
                    except:
                        continue
                unaligned_corpora.append(msg_list)
            d -= timedelta(days=1)
        unaligned_corpora.reverse()  # ascend

        return unaligned_corpora

    def _get_prices_and_ts(self, ss, main_target_date):
        def _get_mv_class(data, use_one_hot=False):
            mv = float(data["movement"])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0
                else:
                    return [0.0, 1.0] if use_one_hot else 1

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2

        def _get_y(data):
            return _get_mv_class(data, use_one_hot=True)

        def _get_prices(data):
            return [float(p) for p in [data["High"], data["Low"], data["Close"]]]

        def _get_mv_percents(data):
            return _get_mv_class(data)

        ts, ys, prices, mv_percents, mv_percents_ori, main_mv_percent, target_price = list(), list(), list(), list(), list(), 0.0, 0.0
        d_t_min = main_target_date - timedelta(days=self.max_n_days - 1)
        d_t_min_pre = main_target_date - timedelta(days=self.max_n_days)
        d_t_max = main_target_date - timedelta(days=1)
        stock_movement_path = os.path.join(str(self.movement_path), '{}.csv'.format(ss))
        stock_df = pd.read_csv(stock_movement_path, sep=',', header=0)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y-%m-%d')  # convert to datatime
        stock_df = stock_df.sort_values(by='Date', ascending=True)
        stock_df['movement'] = stock_df['Adj Close'].pct_change()  # date ascending
        tech_df = get_tech_ind(stock_df, self.args)  # generate ind in date ascending format
        tech_ind_list = tech_df.columns
        tech_dict = {k: [] for k in tech_ind_list}
        stock_df = pd.concat([stock_df, tech_df],axis=1)
        stock_df = stock_df.sort_values(by='Date', ascending=False)
        stock_df['Date'] = stock_df['Date'].apply(lambda x: x.strftime("%Y-%m-%d"))  # convert to string
        for i in range(6):
            stock_df.iloc[:, i + 1] = stock_df.iloc[:, i + 1].astype(float)

        stock_df = stock_df.reset_index()

        for index, data in stock_df.iterrows():
            t = datetime.strptime(data["Date"], '%Y-%m-%d').date()
            if t == main_target_date:
                ts.append(t)
                ys.append(_get_y(data))
                main_mv_percent = data["movement"]
                if -0.005 <= float(main_mv_percent) < 0.0055:  # discard sample with low movement percent
                    return None
            if d_t_min <= t < main_target_date:
                ts.append(t)
                ys.append(_get_y(data))
                prices.append(_get_prices(data))  # high, low, close
                mv_percents.append(_get_mv_percents(data))
                mv_percents_ori.append(str(round(data["movement"], 6)))
                for k in tech_ind_list:
                    tech_dict[k].append(data[k])
            if t < d_t_min:  # one additional line for x_1_prices. not a referred trading day
                prices.append(_get_prices(data))
                mv_percents.append(_get_mv_percents(data))
                mv_percents_ori.append(str(round(data["movement"], 6)))
                for k in tech_ind_list:
                    tech_dict[k].append(data[k])
                break

        T = len(ts)
        if len(ys) != T or len(prices) != T or len(mv_percents) != T:  # ensure data legibility
            return None

        # ascend
        for item in (ts, ys, mv_percents, mv_percents_ori, prices):
            item.reverse()

        for k in tech_ind_list:
            tech_dict[k].reverse()

        if float(prices[1][-1]) > (prices[-1][-1]):
            direction = "decreased"
        else:
            direction = "increased"

        prices_and_ts = {
            "ID": f"{ss}_{main_target_date}_{T}",
            "Prediction": map_bin_label(bin_mapping(main_mv_percent)),
            "PredictionMovement": "Up" if main_mv_percent > 0 else "Down",
            'PERIOD_START_DATE': d_t_min,  # 天数
            'PERIOD_END_DATE': d_t_max,  # 天数
            'T': T,  # 天数
            'ts': ts,  # 具体的日期
            'ys': ys,  # 数据的label, high,low,close
            'main_mv_percent': main_mv_percent,  # target 日期的上升下降比例
            'PERIOD_START_DATE_PRE': d_t_min_pre,  # 天数
            'TARGETSINGLEPRICE': tech_dict['TARGETPRICE'][-1],
            'PMV': ",".join(mv_percents_ori),  # 其他日期的上升下降情况
            'PRICE_PREDICTION': str(round(main_mv_percent * 100, 2)) + "%",  # 价格
            'PRICE_PREDICTION_FLOAT': str(round(main_mv_percent, 6)),  # 价格
            "DIRECTION": direction,
            'PERIOD_START_PRICE': tech_dict['PRICE'][0],  # 价格R
            'PERIOD_END_PRICE': tech_dict['PRICE'][-1],  # 价格
            "SYMBOL": ss,
            "TARGETDATE": main_target_date,
            'PRICE_PREDICTION_HISTORY': ",".join(list(map(str, tech_dict['TARGETPRICE'][:-1]))),  # 价格
        }
        tech_dict = {k: ",".join(list(map(str, tech_dict[k]))) for k in tech_dict}
        prices_and_ts.update(tech_dict)
        return prices_and_ts

    def sample_gen_multiprocess(self, phase, num_processors=None):
        vocab_id_dict = self.index_token(self.config.vocab, key='token')
        stock_id_dict = self.index_token(self.config.stock_symbols, key='token', type='stock')

        # Determine the number of processors to use
        if num_processors is None:
            samples = []
            for s in self.config.stock_symbols:
                samples.extend(self.sample_gen_from_one_stock(vocab_id_dict, stock_id_dict, s, phase))
        else:
            pool = Pool(processes=num_processors)  # Create a Pool with the specified number of processors

            # Prepare arguments for the method
            args = [(vocab_id_dict, stock_id_dict, s, phase) for s in self.config.stock_symbols]

            # Use the pool's map method to execute the sample generation in parallel
            results = pool.starmap(self.sample_gen_from_one_stock, args)

            # Close the pool and wait for the work to finish
            pool.close()
            pool.join()

            # Flatten the list of lists into a single list of samples
            samples = [sample for sublist in results for sample in sublist]
        return samples


def get_tech_ind(data, args):
    tech_df = pd.DataFrame()

    # basic info
    tech_df['OPEN'] = data['Open']
    tech_df['HIGH'] = data['High']
    tech_df['LOW'] = data['Low']
    tech_df['CLOSE'] = data['Close']
    tech_df['ADJCLOSE'] = data['Adj Close']
    tech_df['VOLUME'] = data['Volume']
    tech_df['PRICE'] = data['Adj Close']
    tech_df['TARGETPRICE'] = data['Adj Close'].shift(periods=-1)

    # common used alpha
    tech_df['MV7'] = data['Adj Close'].rolling(window=7).mean()  # Close column
    tech_df['MV10'] = data['Adj Close'].rolling(window=10).mean()
    tech_df['MV20'] = data['Adj Close'].rolling(window=20).mean()
    tech_df['MV30'] = data['Adj Close'].rolling(window=30).mean()
    tech_df['MV60'] = data['Adj Close'].rolling(window=60).mean()
    tech_df['MACD'] = data['Adj Close'].ewm(span=26).mean() - data.iloc[:, 1].ewm(span=12, adjust=False).mean()
    tech_df['BBM'] = data['Adj Close'].rolling(window=20).std()
    tech_df['BBU'] = tech_df['MV20'] + (tech_df['BBM'] * 2)
    tech_df['BBL'] = tech_df['MV20'] - (tech_df['BBM'] * 2)
    tech_df['EMA'] = data['Adj Close'].ewm(com=0.5).mean()
    tech_df['LOGM'] = np.log(data['Adj Close'] - 1)
    tech_df['KUP2'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / (data['High'] - data['Low'] + 1e-12)
    tech_df['KSFT'] = (2 * data['Close'] - data['High'] - data['Low']) / data['Open']
    tech_df['QTLU20'] = tech_df['MV20'].quantile(0.8) / data['Close']
    price_change_volume = abs(data['Close'].pct_change() * data['Volume'])
    tech_df['WVMA10'] = price_change_volume.rolling(window=10).std() / (price_change_volume.rolling(window=10).mean() + 1e-12)
    volume_change = data['Volume'].diff()
    tech_df['VSUMP5'] = volume_change.rolling(window=5).apply(lambda x: x[x > 0].sum(), raw=False) / (abs(volume_change).rolling(window=5).sum() + 1e-12)
    tech_df['VSTD60'] = data['Volume'].rolling(window=60).std() / (data['Volume'] + 1e-12)
    tech_df['MIN20'] = data['Low'].rolling(window=20).min() / data['Close']
    tech_df['VMA60'] = data['Volume'].rolling(window=60).mean() / (data['Volume'] + 1e-12)
    tech_df['MIN10'] = data['Low'].rolling(window=10).min() / data['Close']
    close_change = data['Close'].pct_change()
    log_volume_change = np.log(data['Volume'].diff() + 1)
    tech_df['CORD20'] = close_change.rolling(window=20).apply(
        lambda x: x.corr(log_volume_change[x.index]), raw=False
    )
    tech_df['MAX60'] = data['High'].rolling(window=60).max() / data['Close']
    tech_df['VSUMP20'] = volume_change.rolling(window=20).apply(lambda x: x[x > 0].sum(), raw=False) / (abs(volume_change).rolling(window=20).sum() + 1e-12)
    tech_df['SUMN10'] = volume_change.rolling(window=10).apply(lambda x: x[x < 0].sum(), raw=False) / (abs(volume_change).rolling(window=10).sum() + 1e-12)
    tech_df['QTLU10'] = data['Close'].rolling(window=10).quantile(0.8) / data['Close']
    tech_df['QTLU5'] = data['Close'].rolling(window=5).quantile(0.8) / data['Close']
    tech_df['STD30'] = data['Close'].rolling(window=30).std() / data['Close']
    tech_df['LOW0'] = data['Low'] / data['Close']
    tech_df['VSTD30'] = data['Volume'].rolling(window=30).std() / (data['Volume'] + 1e-12)
    tech_df['SUMN5'] = volume_change.rolling(window=5).apply(lambda x: x[x < 0].sum(), raw=False) / (abs(volume_change).rolling(window=5).sum() + 1e-12)
    low_min = data['Low'].rolling(window=30).min()
    high_max = data['High'].rolling(window=30).max()
    tech_df['RSV30'] = (data['Close'] - low_min) / (high_max - low_min + 1e-12)
    tech_df['KSFT2'] = (2 * data['Close'] - data['High'] - data['Low']) / (data['High'] - data['Low'] + 1e-12)
    tech_df['VSUMN60'] = volume_change.rolling(window=60).apply(lambda x: x[x < 0].sum(), raw=False) / (abs(volume_change).rolling(window=60).sum() + 1e-12)
    tech_df['KLOW2'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / (data['High'] - data['Low'] + 1e-12)
    tech_df['VMA5'] = data['Volume'].rolling(window=5).mean() / (data['Volume'] + 1e-12)
    tech_df['VSTD10'] = data['Volume'].rolling(window=10).std() / (data['Volume'] + 1e-12)
    tech_df['KLOW'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Open']
    positive_change_sum = data['Close'].diff().rolling(window=30).apply(lambda x: x[x > 0].sum(), raw=False)
    negative_change_sum = (-data['Close'].diff()).rolling(window=30).apply(lambda x: x[x > 0].sum(), raw=False)
    tech_df['SUMD30'] = (positive_change_sum - negative_change_sum) / (abs(data['Close'].diff()).rolling(window=30).sum() + 1e-12)
    tech_df['QTLD5'] = data['Close'].rolling(window=5).quantile(0.2) / data['Close']
    tech_df['QTLU30'] = data['Close'].rolling(window=30).quantile(0.8) / data['Close']
    tech_df['ROC60'] = data['Close'].shift(60) / data['Close']
    tech_df['SUMP60'] = data['Close'].diff().rolling(window=60).apply(lambda x: x[x > 0].sum(), raw=False) / (abs(data['Close'].diff()).rolling(window=60).sum() + 1e-12)
    low_min_5 = data['Low'].rolling(window=5).min()
    high_max_5 = data['High'].rolling(window=5).max()
    tech_df['RSV5'] = (data['Close'] - low_min_5) / (high_max_5 - low_min_5 + 1e-12)
    tech_df['MA10'] = data['Close'].rolling(window=10).mean() / data['Close']

    tech_df.fillna(0, inplace=True)
    for k in tech_df.columns:
        if k in ["VOLUME"]:
            tech_df[k] = tech_df[k].apply(lambda x: int(x/100))
        else:
            tech_df[k] = tech_df[k].apply(lambda x: int(round(x, 2) * 100))
    return tech_df
