from abc import abstractmethod
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
from sklearn.metrics import accuracy_score, f1_score
import ast
import re
import os
import metrics
from constants import config_loader
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
except:
    print("vllm not installed")
import datasets

class QuantEval():
    def __init__(self, model, tokenizer, batch_size, max_length, dataset_key="", temperature=0, model_args =None,  *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset_key = dataset_key
        self.temperature = temperature
        self.model_args = model_args

    @abstractmethod
    def load_dataset(self):
        pass

    def format_input(self,x):
        pass
        return x

    def parse_llm_result(self,x):
        return x

    def convert_llm_response_to_label(self, i, o):
        return o

    def predict(self, test_list):
        if self.tokenizer:
            res_sentences = self.model_infer(test_list)
        else:
            res_sentences = self.vllm_model_infer(test_list)
        return res_sentences

    def model_infer(self, test_list):
        if len(test_list) == 0:
            return []
        tokens = self.tokenizer(test_list, return_tensors='pt', padding=True, max_length=self.max_length)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()

        if self.max_length < 0:
            max_token_len = tokens['input_ids'].shape[1]
            predict_max_len = max_token_len - self.max_length
        else:
            predict_max_len = self.max_length

        res = self.model.generate(**tokens, max_length=predict_max_len, )
        res_sentences = [self.tokenizer.decode(i) for i in res]
        try:
            res_sentences = [o.split(i)[1] for i, o in zip(test_list, res_sentences)]
        except:
            print("fail to parse llm res")
        return res_sentences

    def vllm_model_infer(self, test_list):
        if len(test_list) == 0:
            return []
        predict_len = self.max_length if self.max_length > 0 else -self.max_length

        if self.model_args.lora_path:
            outputs = self.model.generate(test_list, SamplingParams(temperature=self.temperature, top_p=1, max_tokens=predict_len),lora_request=LoRARequest("sql_adapter", 1, self.model_args.lora_path))
        else:
            outputs = self.model.generate(test_list, SamplingParams(temperature=self.temperature, top_p=1, max_tokens=predict_len))
        res_sentences = [output.outputs[0].text for output in outputs]
        return res_sentences

    def eval(self):
        prediction_list = []
        if len(self.dataset_key) > 0:
            input_list, label_list,id_list = self.load_dataset(dataset_key=self.dataset_key)
        else:
            input_list, label_list,id_list = self.load_dataset()
        total_steps = len(input_list) // self.batch_size + 1
        print(f"Input example: {input_list[0]}, \nOutput example:{label_list[0]}")
        print(f"Total len: {len(input_list)}. Batch size: {self.batch_size}. Total steps: {total_steps}. ")
        ori_input_list = []
        ori_output_list = []
        for i in tqdm(range(total_steps)):
            cur_input = input_list[i * self.batch_size:(i + 1) * self.batch_size]
            res_sentences = self.predict(cur_input)
            ori_input_list.extend(input_list[i * self.batch_size:(i + 1) * self.batch_size])
            ori_output_list.extend(res_sentences)
            prediction = [self.parse_llm_result(o) for o in res_sentences]
            prediction = [self.convert_llm_response_to_label(i, o) for i, o in zip(cur_input, prediction)]
            prediction_list += prediction
            torch.cuda.empty_cache()

        acc = accuracy_score(label_list, prediction_list)
        f1_macro = f1_score(label_list, prediction_list, average="macro")
        f1_micro = f1_score(label_list, prediction_list, average="micro")
        f1_weighted = f1_score(label_list, prediction_list, average="weighted")

        line_list = [json.dumps({"input": i, "output": o, "parsed": p, "label": l, "id": id}) for i, o, p, l,id in
                     zip(ori_input_list, ori_output_list, prediction_list, label_list,id_list)]

        valid_label_list = [l for l, p in zip(label_list, prediction_list) if p in [0, 1]]
        valid_prediction_list = [p for l, p in zip(label_list, prediction_list) if p in [0, 1]]

        if len(valid_prediction_list) > 0:
            acc = accuracy_score(valid_label_list, valid_prediction_list)
            f1_macro = f1_score(valid_label_list, valid_prediction_list, average="macro")
            f1_micro = f1_score(valid_label_list, valid_prediction_list, average="micro")
            f1_weighted = f1_score(valid_label_list, valid_prediction_list, average="weighted")
            print(
                f"Test set has {len(prediction_list)} samples, while {len(valid_prediction_list)} could be predicted.")
            tp, fp, tn, fn = metrics.create_confusion_matrix(np.array(valid_label_list),
                                                             np.array(valid_prediction_list), False)
            print("eval tp = {}, fp = {}, tn = {}, fn = {}".format(tp, fp, tn, fn))
            mcc = metrics.eval_mcc(tp, fp, tn, fn)
        else:
            print("no valid prediction")
            mcc = -1
        print(
            f"Acc: {acc}. Mcc: {mcc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")
        return {"acc": acc, "mcc": mcc, "f1_macro": f1_macro, "f1_micro": f1_micro,
                "f1_weighted": f1_weighted}, line_list

class SentimentEval(QuantEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "prompt_version" in kwargs:
            self.prompt_version = kwargs["prompt_version"]

    def parse_sentiment_from_text(self, x):
        if 'positive' in x or 'Positive' in x:
            return 'positive'
        elif 'negative' in x or 'Negative' in x:
            return 'negative'
        else:
            return 'neutral'

    def parse_llm_result(self, x):
        return self.parse_sentiment_from_text(x)

    def format_example(self, example: dict) -> dict:
        context = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            context += f"Input: {example['input']}\n"
        context += "Answer: "
        target = example["output"]
        return {"input": context, "output": target}

    def map_sentiment(self, x):
        dic = {
            0: "negative",
            1: 'neutral',
            2: 'positive',
        }
        return dic[x]

class FPBSentimentEval(SentimentEval):
    def load_dataset(self):
        dataset = load_dataset("financial_phrasebank", "sentences_50agree")["train"]
        dataset = dataset.train_test_split(seed=42)['test']
        dataset = dataset.to_pandas()
        dataset.columns = ["input", "output"]
        dataset["output"] = dataset["output"].apply(lambda x: self.map_sentiment(x))
        if self.prompt_version=="0":
            dataset["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
        elif self.prompt_version=="1":
            dataset["instruction"] = "What is the sentiment of this sentence? Please choose an answer from {negative/neutral/positive}."
        dataset[["input", "output"]] = dataset.apply(self.format_example, axis=1, result_type="expand")
        input_list, label_list = dataset["input"].tolist(), dataset["output"].tolist()
        label_list = [self.parse_sentiment_from_text(s) for s in label_list]
        return input_list, label_list

class FIQASentimentEval(SentimentEval):
    def load_dataset(self):
        def make_label(x):
            if x < - 0.1:
                return "negative"
            elif x >= -0.1 and x < 0.1:
                return "neutral"
            elif x >= 0.1:
                return "positive"

        dataset = load_dataset('pauri32/fiqa-2018')
        dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        dataset = dataset.train_test_split(0.226, seed=42)['test']
        dataset = dataset.to_pandas()
        dataset["output"] = dataset.sentiment_score.apply(make_label)
        dataset = dataset[['sentence', 'output']]
        dataset.columns = ["input", "output"]
        if self.prompt_version=="0":
            dataset[
                "instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
        elif self.prompt_version=="1":
            dataset["instruction"] = "What is the sentiment of this sentence? Please choose an answer from {negative/neutral/positive}."


        dataset[["input", "output"]] = dataset.apply(self.format_example, axis=1, result_type="expand")
        input_list, label_list = dataset["input"].tolist(), dataset["output"].tolist()
        label_list = [self.parse_sentiment_from_text(s) for s in label_list]
        return input_list, label_list

class TFNSSentimentEval(SentimentEval):
    def load_dataset(self):
        # dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
        dataset = load_dataset('csv', data_files={
            'train': os.path.join(config_loader, fr"data/evaluation/twitter-financial-news-sentiment/sent_train.csv"),
            'test': os.path.join(config_loader, fr"data/evaluation/twitter-financial-news-sentiment/sent_valid.csv")})
        dataset = dataset['test']
        dataset = dataset.to_pandas()
        dataset['label'] = dataset['label'].apply(lambda x: self.map_sentiment(x))
        dataset.columns = ['input', 'output']
        if self.prompt_version=="0":
            dataset[
                "instruction"] = "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
        elif self.prompt_version=="1":
            dataset["instruction"] = "What is the sentiment of this sentence? Please choose an answer from {negative/neutral/positive}."

        dataset[["input", "output"]] = dataset.apply(self.format_example, axis=1, result_type="expand")
        input_list, label_list = dataset["input"].tolist(), dataset["output"].tolist()
        label_list = [self.parse_sentiment_from_text(s) for s in label_list]
        return input_list, label_list

    def map_sentiment(self, x):
        dic = {
            0:"negative",
            1:'positive',
            2:'neutral',
        }
        return dic[x]

class NWGISentimentEval(SentimentEval):
    def load_dataset(self):
        # dataset = datasets.load_dataset('oliverwang15/news_with_gpt_instructions')
        dataset = load_dataset('parquet', data_files={
            'train': os.path.join(config_loader,
                                  fr"data/evaluation/news_with_gpt_instructions/train-00000-of-00001-dd971e407aecb39b.parquet"),
            'test': os.path.join(config_loader,
                                 fr"data/evaluation/news_with_gpt_instructions/test-00000-of-00001-c551483ebf365496.parquet")})
        dataset = dataset['test'].to_pandas()
        dataset['output'] = dataset['label'].apply(lambda x: self.map_sentiment(x))
        dataset = dataset[['news', 'output']]
        dataset.columns = ['input', 'output']
        if self.prompt_version=="0":
            dataset[
                "instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
        elif self.prompt_version=="1":
            dataset["instruction"] = "What is the sentiment of this sentence? Please choose an answer from {negative/neutral/positive}."

        dataset[["input", "output"]] = dataset.apply(self.format_example, axis=1, result_type="expand")
        input_list, label_list = dataset["input"].tolist(), dataset["output"].tolist()
        label_list = [self.parse_sentiment_from_text(s) for s in label_list]
        return input_list, label_list

    def map_sentiment(self, x):
        dic = {
            'strong negative': "negative",
            'moderately negative': "negative",
            'mildly negative': "neutral",
            'strong positive': "positive",
            'moderately positive': "positive",
            'mildly positive': 'neutral',
            'neutral': 'neutral',
        }
        return dic[x]

class BTSASentimentEval(SentimentEval):
    def load_dataset(self):
        dataset = load_dataset("ckandemir/bitcoin_tweets_sentiment_kaggle")['test']
        dataset = dataset.to_pandas()
        dataset = dataset[['text', 'Sentiment']]
        dataset.columns = ["input", "output"]
        dataset['output'] = dataset['output'].apply(lambda x: self.map_sentiment(x))
        if self.prompt_version=="0":
            dataset[
                "instruction"] = "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
        elif self.prompt_version=="1":
            dataset["instruction"] = "What is the sentiment of this sentence? Please choose an answer from {negative/neutral/positive}."

        dataset[["input", "output"]] = dataset.apply(self.format_example, axis=1, result_type="expand")
        input_list, label_list = dataset["input"].tolist(), dataset["output"].tolist()
        label_list = [self.parse_sentiment_from_text(s) for s in label_list]
        return input_list, label_list

    def map_sentiment(self, x):
        dic = {
            'Negative': "negative",
            'Neutral': 'neutral',
            'Positive': 'positive',
        }
        return dic[x]

class SentimentPricePredictionLabeling(SentimentEval):
    def load_dataset(self, dataset_key):
        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            # print(os.path.join(PROJECT_PATH, fr"data\gpt_label\{dataset_key}.json"))
            line_list = file.readlines()
            line_list = [l.strip() for l in line_list]

        news_list = [json.loads(line_list[i])['COMPANY_NEWS_LIST'] for i in range(len(line_list))]
        news_list = [s[1] for n in news_list for s in n]
        input_list = [self.format_input(news) for news in news_list]
        return news_list, input_list

    def label(self):
        prediction_list = []
        news_list, input_list = self.load_dataset(dataset_key=self.dataset_key)
        total_steps = len(input_list) // self.batch_size + 1
        print(f"News example: {news_list[0]}, Input example:{input_list[0]}")
        print(f"Total len: {len(input_list)}. Batch size: {self.batch_size}. Total steps: {total_steps}. ")
        for i in tqdm(range(total_steps)):
            res_sentences = self.predict(input_list[i * self.batch_size:(i + 1) * self.batch_size])
            res_sentences = [self.parse_llm_result(i) for i in res_sentences]
            prediction_list += res_sentences
            torch.cuda.empty_cache()

        return news_list, prediction_list

    def parse_llm_result(self, x):
        if 'positive' in x or 'Positive' in x:
            return 'positive'
        elif 'negative' in x or 'Negative' in x:
            return 'negative'
        else:
            return 'neutral'

    def format_input(self, news):
        if self.prompt_version=="0":
            context = "Instruction: What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.\n"
        elif self.prompt_version=="1":
            context = "Instruction: What is the sentiment of this sentence? Please choose an answer from {negative/neutral/positive}.\n"
        context += f"Input: {news}\n"
        context += "Answer: "
        return context

    def parse_llm_list_result(self, x_list):
        res_list = []
        for x in x_list:
            if 'positive' in x or 'Positive' in x:
                res_list.append('positive')
            elif 'negative' in x or 'Negative' in x:
                res_list.append('negative')
            else:
                res_list.append('neutral')

        map_dict = {'positive': 1, 'neutral': 0, 'negative': -1}
        res = sum([map_dict[r] for r in res_list])
        if res > 0:
            return 1
        elif res < 0:
            return 0
        else:
            return -1

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class LLMFileLabeling(QuantEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     self.model = nn.DataParallel(self.model)

    def load_dataset(self, dataset_key=""):
        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            news_list = [json.loads(l.strip())['input'] for l in line_list]
        input_list = [(news, self.format_input(news)) for news in news_list]
        inference_dataset = MyDataset(input_list)
        inference_loader = DataLoader(inference_dataset, batch_size=self.batch_size, shuffle=False)
        return inference_loader

    def label(self):
        res_list = []
        inference_loader = self.load_dataset(dataset_key=self.dataset_key)
        print(
            f"News example: {inference_loader.dataset.data[0][0]}, Input example:{inference_loader.dataset.data[0][1]}")
        print(f"Total len: {len(inference_loader)}. Batch size: {self.batch_size}. ")

        with torch.no_grad():
            for batch in inference_loader:
                news_list, input_list = list(batch[0]), list(batch[1])
                res_sentences = self.predict(input_list)
                prediction = [self.parse_llm_result(o) for o in res_sentences]
                prediction = [self.convert_llm_response_to_label(i, o) for i, o in zip(input_list, prediction)]
                torch.cuda.empty_cache()
                res_list.extend([json.dumps({"news": n, "input": i, "output": o, "prediction": p}) for n, i, o, p in
                                 zip(news_list, input_list, res_sentences, prediction)])
        return res_list

    def format_input(self, input):
        return input

class GeneralLLMFileLabeling(QuantEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def load_dataset(self, dataset_key=""):
        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            info_list = [json.loads(l.strip()) for l in line_list]
        format_input_list = [(json.dumps(info), self.format_input(info['input'])) for info in info_list]
        inference_dataset = MyDataset(format_input_list)
        inference_loader = DataLoader(inference_dataset, batch_size=self.batch_size, shuffle=False)
        return inference_loader

    def label(self):
        res_list = []
        inference_loader = self.load_dataset(dataset_key=self.dataset_key)
        print(
            f"News example: {inference_loader.dataset.data[0][0]}, Input example:{inference_loader.dataset.data[0][1]}")
        print(f"Total len: {len(inference_loader)}. Batch size: {self.batch_size}. ")

        with torch.no_grad():
            for batch in inference_loader:
                info_list, format_input_list = list(batch[0]), list(batch[1])
                res_sentences = self.predict(format_input_list)
                prediction = [self.parse_llm_result(o) for o in res_sentences]
                prediction = [self.convert_llm_response_to_label(i, o) for i, o in zip(format_input_list, prediction)]
                torch.cuda.empty_cache()
                batch_list = []
                for idx in range(len(info_list)):
                    n,i,o,p = json.loads(info_list[idx]),format_input_list[idx],res_sentences[idx],prediction[idx]
                    cur_dict = {"RES_input": i, "RES_output": o, "RES_prediction": p}
                    cur_dict.update(n)
                    batch_list.append(json.dumps(cur_dict))
                res_list.extend(batch_list)
        return res_list

    def format_input(self, input):
        return input

class SentimentAnalysisFileLabeling(LLMFileLabeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "prompt_version" in kwargs:
            self.prompt_version = kwargs["prompt_version"]

    def format_input(self, news):
        if self.prompt_version=="0":
            context = "Instruction: What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.\n"
        elif self.prompt_version=="1":
            context = "Instruction: What is the sentiment of this sentence? Please choose an answer from {negative/neutral/positive}.\n"

        context += f"Input: {news}\n"
        context += "Answer: "
        return context

    def parse_llm_result(self, x):
        if 'positive' in x or 'Positive' in x:
            return 'positive'
        elif 'negative' in x or 'Negative' in x:
            return 'negative'
        else:
            return 'neutral'

class TechnicalAnalysisFileLabeling(GeneralLLMFileLabeling):

    def parse_llm_result(self, text):
        pattern = r"(-?\d+(\.\d+)?)"
        match = re.search(pattern, text)
        if match:
            number_str = match.group(1)
            number = float(number_str)
            return number
        else:
            number = -1
        return number

    def convert_llm_response_to_label(self, input, current_num):
        if current_num == -1:
            return -1
        input = input.split("Please predict the ")[0]
        last_num = self.parse_last_numbers_from_text(input)
        if last_num:
            try:
                if float(last_num) < float(current_num):
                    return 1
                else:
                    return 0
            except:
                print(input, "####", current_num)
                return -1
        else:
            return -1

    def format_input(self, input):
        return input

    def parse_last_numbers_from_text(self, text):
        pattern = r"(-?\d+(\.\d+)?)"
        match = re.findall(pattern, text)
        if match:
            return [s[0] for s in match][-1]
        else:
            return None

class SentimentPricePredictionEval(QuantEval):
    def load_input2label_dict(self, input_dict):
        self.news2result_dict = input_dict

    def load_dataset(self, dataset_key):
        def map_label(text):
            text = text.replace("%","")
            mv = float(text)
            threshold_1, threshold_2 = -0.005, 0.0055
            if mv < threshold_1:
                return 0
            elif mv < threshold_2:
                return -1
            else:
                return 1

        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            line_list = [l.strip() for l in line_list]
        input_list = [json.loads(line_list[i])["COMPANY_NEWS_LIST"] for i in range(len(line_list))]
        label_list = [{"Up":1,"Down":0}[json.loads(line_list[i])["PredictionMovement"]] for i in range(len(line_list))]
        return input_list, label_list

    def format_input(self, news_list):
        formatted_news_list = []
        for n in news_list:
            context = "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\n"
            context += f"Input: {n}\n"
            context += "Answer: "
            formatted_news_list.append(context)
        return formatted_news_list

    def predict(self, test_list_list):
        res_sentences_list = [[self.news2result_dict[r[1]] for r in test_list] for test_list in test_list_list]
        return res_sentences_list

    def parse_llm_result(self, x_list):
        res_list = []
        for x in x_list:
            if 'positive' in x or 'Positive' in x:
                res_list.append('positive')
            elif 'negative' in x or 'Negative' in x:
                res_list.append('negative')
            else:
                res_list.append('neutral')

        map_dict = {'positive': 1, 'neutral': 0, 'negative': -1}
        res = sum([map_dict[r] for r in res_list])
        if res > 0:
            return 1
        elif res < 0:
            return 0
        else:
            return -1

class TimeSeriesPricePredictionEval(QuantEval):
    def load_dataset(self, dataset_key="technical_input"):
        def map_label(text):
            text = text.replace("%","")
            mv = float(text)
            threshold_1, threshold_2 = -0.005, 0.0055
            if mv < threshold_1:
                return 0
            elif mv < threshold_2:
                return -1
            else:
                return 1

        with open(os.path.join(config_loader,
                               r"data\overall\finance-overall-analysis-test-v003.json"), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            line_list = [l.strip() for l in line_list]
        input_list = [json.loads(line_list[i])[dataset_key] for i in range(len(line_list))]
        label_list = [map_label(json.loads(line_list[i])['price_prediction']) for i in range(len(line_list))]
        return input_list, label_list

    def parse_llm_result(self, text):
        pattern = r"(-?\d+(\.\d+)?)"
        match = re.search(pattern, text)
        if match:
            number_str = match.group(1)
            number = float(number_str)
            if number > 0:
                return 1
            else:
                return 0
        else:
            number = -1
        return number

class NoAgentPricePredictionEval(QuantEval):
    def load_dataset(self, dataset_key):
        def map_label(text):
            try:
                pattern = r"Prediction:\s*(Down|Up)"
                match = re.search(pattern, text).group(1)
            except:
                print(text, "map label error")
                raise RuntimeError
            if match == "Up":
                res = 1
            elif match == "Down":
                res = 0
            else:
                raise RuntimeError(text)
            return res

        # df = pd.read_csv(os.path.join(PROJECT_PATH, r"data\gpt_label\output\W3_hanstong_ploutosnotasv003test_ScrapeV2WP.tsv"), sep='\t',header=0)
        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            line_list = [l.strip() for l in line_list]
        input_list = [json.loads(line_list[k])['input'] for k in range(len(line_list))]
        id_list = [json.loads(line_list[k])['ID'] for k in range(len(line_list))]
        label_list = [map_label(json.loads(line_list[k])['output']) for k in range(len(line_list))]
        return input_list, label_list,id_list

    def parse_llm_result(self, text):
        return detect_trend_from_sent(text)
        # pattern = r'Prediction:\s*(\w+)'
        # match = re.search(pattern, text)
        # number = -1
        # trend = detect_trend_from_sent(text)
        # if trend is not None:
        #     return trend
        # else:
        #
        #     number_str = match.group(1).lower()
        #     if "down" in number_str:
        #         return 0
        #     elif "up" in number_str:
        #         return 1
        #     else:
        #         print("can not parse:",number_str, text)
        # return number

class TimeSeriesNumPricePredictionCompareEval(QuantEval):

    def load_dataset(self, dataset_key):
        def map_label(text):
            text = text.replace("%","")
            mv = float(text)
            threshold_1, threshold_2 = -0.005, 0.0055
            if mv < threshold_1:
                return 0
            elif mv < threshold_2:
                return -1
            else:
                return 1

        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            line_list = [l.strip() for l in line_list]
        input_list = [json.loads(line_list[i])["technical_num_input"] for i in range(len(line_list))]
        label_list = [map_label(json.loads(line_list[i])['price_prediction']) for i in range(len(line_list))]
        return input_list, label_list

    def parse_last_numbers_from_text(self, text):
        pattern = r"(-?\d+(\.\d+)?)"
        match = re.findall(pattern, text)
        if match:
            return [s[0] for s in match][-1]
        else:
            return None

    def parse_llm_result(self, text):
        pattern = r"(-?\d+(\.\d+)?)"
        match = re.search(pattern, text)
        if match:
            number_str = match.group(1)
            number = float(number_str)
            return number
        else:
            number = -1
        return number

    def convert_llm_response_to_label(self, input, current_num):
        if current_num == -1:
            return -1
        last_num = self.parse_last_numbers_from_text(input)
        if last_num:
            try:
                if float(last_num) < float(current_num):
                    return 1
                else:
                    return 0
            except:
                print(input, "####", current_num)
                return -1
        else:
            return -1

def detect_trend_from_sent(sent):
    sent = sent.lower()
    wordsmap = {"rebound":1,"rise":1, "rise by":1,"move up":1,"up":1,"down":0, "drop":0,
                "up":1,"decline":0,"downward":0,"uptrend":1,"bullish":1,"bearish":0,"increase":1,
                "decrease":0,"rally":1,"close do":0,"pullback":0,"flat or slightly down":0}
    wordstr = "|".join(list(wordsmap.keys()))

    matches = re.compile(fr"Prediction:\s*(?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"I predict that (?P<stock_symbol>.*) will continue to (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"The prediction is (?P<action>{wordstr}) by 0-1% for (?P<stock_symbol>.*)").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"predict that (?P<stock_symbol>.*) will experience a slight (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"The prediction is (?P<action>{wordstr}) by").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"predict that (?P<stock_symbol>.*) will continue its (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"predict that (?P<stock_symbol>.*) will (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"predict that (?P<stock_symbol>.*) will have a slight (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"The price may (?P<action>{wordstr}) by").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"predict that (?P<stock_symbol>.*)'s stock price will go (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"expect (?P<stock_symbol>.*) to (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"The stock price of (?P<stock_symbol>.*) is expected to (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    matches = re.compile(fr"predict that (?P<stock_symbol>.*)'s stock price will remain (?P<action>{wordstr})").search(sent)
    if matches:
        return wordsmap[matches.group("action")]
    return -1

class TimeSeriesPricePredictionCompareEval(TimeSeriesNumPricePredictionCompareEval):

    def load_dataset(self, dataset_key):
        def map_label(text):
            text = text.replace("%","")
            mv = float(text)
            threshold_1, threshold_2 = -0.005, 0.0055
            if mv < threshold_1:
                return 0
            elif mv < threshold_2:
                return -1
            else:
                return 1

        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            line_list = [l.strip() for l in line_list]
        input_list = [json.loads(line_list[i])["input"] for i in range(len(line_list))]
        label_list = [json.loads(line_list[i])['label'] for i in range(len(line_list))]
        id_list = [json.loads(line_list[i])['ID'] for i in range(len(line_list))]
        return input_list, label_list, id_list

    def convert_llm_response_to_label(self, input, current_num):
        if current_num == -1:
            return -1
        input = input.split("Please predict the ")[0]
        last_num = self.parse_last_numbers_from_text(input)
        if last_num:
            try:
                if float(last_num) < float(current_num):
                    return 1
                else:
                    return 0
            except:
                print(input, "####", current_num)
                return -1
        else:
            return -1

class SentimentInstrucPricePredictionEval(QuantEval):

    def load_dataset(self, dataset_key):

        with open(os.path.join(config_loader, dataset_key), "r",
                  encoding="utf-8") as file:
            line_list = file.readlines()
            line_list = [l.strip() for l in line_list]
        input_list = [json.loads(line_list[i])["input"] for i in range(len(line_list))]
        label_list = [json.loads(line_list[i])['label'] for i in range(len(line_list))]
        id_list = [json.loads(line_list[i])['ID'] for i in range(len(line_list))]
        return input_list, label_list,id_list

    def parse_llm_result(self, x):
        x = x.lower()
        if 'up' in x or 'positive' in x:
            return 1
        elif 'down' in x or 'negative' in x:
            return 0
        else:
            return -1
