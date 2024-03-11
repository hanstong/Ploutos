#!/usr/local/bin/python
import logging
import logging.config
import yaml
import itertools
import os
import io
import json
import sys

# from constants import PROJECT_PATH

import os, sys, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from constants import config_loader

class PathParser:
    def __init__(self, config_name = "config.yml"):
        config_fp = os.path.join(os.path.dirname(__file__), config_name)
        config = yaml.load(open(config_fp, 'r'),Loader=yaml.FullLoader)
        self.config = config
        self.config_model = config['model']
        self.dates = config['dates']
        self.config_stocks = config['stocks']  # a list of lists
        self.config_stocks = self.config_stocks
        self.stock_symbols = list(itertools.chain.from_iterable([self.config_stocks[key] for key in self.config_stocks]))

        config_path = config['paths']
        self.root = config_loader
        self.data = config_path['data']
        self.res = os.path.join(self.root, config_path['res'])
        self.graphs = os.path.join(self.root, config_path['graphs'])
        self.checkpoints = os.path.join(self.root, config_path['checkpoints'])

        self.glove = os.path.join(self.res, config_path['glove'])

        self.retrieved = os.path.join(self.data, config_path['tweet_retrieved'])
        self.preprocessed = os.path.join(self.data, config_path['tweet_preprocessed'])
        self.movement = os.path.join(self.data, config_path['price'])
        self.vocab = os.path.join(self.res, config_path['vocab_tweet'])
        self.prompt_path = os.path.join(self.root, "data/prompt")

        with io.open(str(self.vocab), 'r', encoding='utf-8') as vocab_f:
            vocab = json.load(vocab_f)
            self.vocab_size = len(vocab) + 1  # for unk

