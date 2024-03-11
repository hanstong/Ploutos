import json
import pickle
import os
from collections import defaultdict
from data_util import split_file
import pandas as pd

class PromptGen():
    """
    The PromptGen class is designed to automate the generation of prompts for ploutos
    based on provided configurations, data versions, and phase lists.
    Attributes:
        args: Command-line arguments or other configurations passed to the class.
        config: Configuration settings for the project.
        project_path: The root path of the project directory.
        dataset: The name of the dataset being used.
        phase_list: List of phases or stages in data processing (e.g., train, dev, test).
        data_version: The version of the data being processed.
        phase2record_dict: Dictionary mapping each phase to its corresponding data records.
    """
    def __init__(self, args, config, data_version, phase_list=["train", "dev", "test"]):

        self.args = args
        self.config = config
        self.project_path = self.config.root
        self.dataset = self.args.dataset
        self.phase_list = phase_list
        self.data_version = data_version
        self.phase2record_dict = self.load_info(data_version, phase_list)

    def load_info(self, version, phase_list):
        phase2info_dict = {}
        for phase in phase_list:
            with open(os.path.join(self.project_path, f'data\info\info-{self.dataset}-{phase}-{version}.pkl'),
                      'rb') as handle:
                phase2info_dict[phase] = pickle.load(handle)
        return phase2info_dict

    def gen_faithfulness_prompt(self, rationale_text, record_dict, pos_neg_prompt=None, groundscore_prompt=None):
        pos_rationales, neg_rationale = self.split_rationale(rationale_text)
        prompt_list = []
        movement_accuracy_prompt_list = []
        for rationale in pos_rationales:
            record_dict["SINGLE_RATIONALE"] = rationale
            if groundscore_prompt:
                prompt_list.append(groundscore_prompt.format(**record_dict))
            if pos_neg_prompt:
                record_dict["SINGLE_Movement"] = "bullish"
                movement_accuracy_prompt_list.append((pos_neg_prompt.format(**record_dict), "bullish"))
        for rationale in neg_rationale:
            record_dict["SINGLE_RATIONALE"] = rationale
            if groundscore_prompt:
                prompt_list.append(groundscore_prompt.format(**record_dict))
            if pos_neg_prompt:
                record_dict["SINGLE_Movement"] = "bearish"
                movement_accuracy_prompt_list.append((pos_neg_prompt.format(**record_dict), "bearish"))
        return prompt_list, movement_accuracy_prompt_list

    def split_rationale(self, rationales):
        def clean_text(x):
            x = x.split("\n")
            x = [x.replace("  - ", "") for x in x if
                 len(x) > 20 and 'bullish rationale' not in x.lower() and 'bearish rationale' not in x.lower()]
            return x

        try:
            if "[Bearish Rationales]:" not in rationales:
                return clean_text(rationales), []
            elif "[Bullish Rationales]:" not in rationales:
                return [], clean_text(rationales)
            elif "[Bullish Rationales]:" in rationales and "[Bearish Rationales]:" in rationales:
                pos_ra = rationales.split("[Bearish Rationales]:")[0]
                neg_ra = rationales.split("[Bearish Rationales]:")[1]
                return clean_text(pos_ra), clean_text(neg_ra)
            else:
                print("invalid rationale", rationales)
                return [], []
        except:
            print("aaaa")
            raise RuntimeError

    def gen_ground_prompt(self, ration_path, ground_prompt_version, ground_name=""):
        ground_score_prompt = self.load_prompt(os.path.join(self.project_path,
                                                            f'data\prompt\ground_{ground_prompt_version}.txt'))
        for phase in ["test"]:
            record_dict = self.phase2record_dict[phase]
            record_dict = {r["ID"]: r for r in record_dict}
            with open(ration_path, "r", encoding="utf-8") as f:
                line_list = f.readlines()
                line_list = [json.loads(l) for l in line_list]
                id2ration = {l["id"]: l["output"] for l in line_list}
            faithfulness_eval_data_list = []
            for record_id in record_dict:
                if record_id in id2ration:
                    cur_ration = self.parse_rationale(id2ration[record_id], analysis_key="Analysis:")["Rationale"]
                    faithfulness_eval_data, movement_prompt = self.gen_faithfulness_prompt(cur_ration,
                                                                                           record_dict[record_id],
                                                                                           groundscore_prompt=ground_score_prompt)
                    faithfulness_eval_data_list.extend(
                        [{"id": record_id, "Prompt": d, "Max Length": 3000} for d in faithfulness_eval_data])
            self.save_prompt_list(faithfulness_eval_data_list, os.path.join(self.project_path,
                                                                            rf'data\gpt_label\input\W3_hanstong_n{ground_name}d{self.dataset.replace("_", "")}truthploutos{self.data_version}p{ground_prompt_version}_ScrapeV2WP.tsv'))

    def parse_rationale(self, x, analysis_key="[Prediction]:"):
        x = x.replace("|", "\n").replace("\n\n", "\n").strip("\n")
        start_symbol = ["bot", "system"]
        for sy in start_symbol:
            if x.startswith(sy):
                x = x[len(sy):]
        if analysis_key in x:
            rationale = x.split(analysis_key)[0]
            analysis = x.split(analysis_key)[1]
            return {"Rationale": rationale, "Analysis": analysis}
        else:
            print("########", x)
            return {"Rationale": x, "Analysis": ""}

    def parse_llm_result(self, x):
        if ('positive' in x) or ('Positive' in x):
            return 'Up'
        elif ('negative' in x) or ('Negative' in x):
            return 'Down'
        else:
            return 'Unknown'

    def gen_prompt(self, phase, input_prompt_path, output_prompt_path, customized_dict={}):
        input_prompt = self.load_prompt(input_prompt_path)
        ouput_prompt = self.load_prompt(output_prompt_path)
        record_list = self.phase2record_dict[phase]
        data_list = []
        for record in record_list:
            record.update(customized_dict)
            data = {"input": input_prompt.format(**record),
                    "output": ouput_prompt.format(**record),
                    "instruction": "", "ID": record["ID"], "label": self.parse_label(record)}
            data_list.append(data)
        print(f"Saving {len(data_list)} data to path")
        return data_list

    def parse_label(self, record):
        if record["PredictionMovement"] == "Up":
            label = 1
        elif record["PredictionMovement"] == "Down":
            label = 0
        else:
            raise RuntimeError
        return label

    def gen_sentiment_prompt(self, version="v1.0"):
        news2ids_dict = defaultdict(list)
        instruct_list = []
        # prompt for training
        for phase in self.phase_list:
            sup_sentiment_list = self.gen_prompt(phase,
                                                 f'./data\prompt\sentiment_analysis\ploutosgpt_sentiment_analysis_expert_{version}_input.txt',
                                                 f'./data/prompt/sentiment_analysis/ploutosgpt_sentiment_analysis_expert_{version}_output.txt',
                                                 customized_dict={"SA_OPTION": "{positive/negative}"})
            # filter empty news
            record_list = self.phase2record_dict[phase]
            valid_news_id_set = [r["ID"] for r in record_list if len(r["COMPANY_NEWS_LIST"]) > 0]
            sup_sentiment_list = [n for n in sup_sentiment_list if n["ID"] in valid_news_id_set]
            print(f"phase:{phase}, ori len:{len(record_list)}, valid len:{len(sup_sentiment_list)}")

            # add customize result
            for i in range(len(sup_sentiment_list)):
                sup_sentiment_list[i]["output"] = {"Up": "positive", "Down": "negative"}[
                    sup_sentiment_list[i]["output"]]
            self.save_data_list(sup_sentiment_list, os.path.join(self.project_path,
                                                                 rf'data\training\{self.dataset}-SA-{phase}-d{self.data_version}-p{version}.json'))

            # add news
            for nl in self.phase2record_dict[phase]:
                for n in nl["COMPANY_NEWS_LIST"]:
                    news2ids_dict[n[1]].append(nl["ID"])
            instruct_list.extend(sup_sentiment_list)

            # add sup&unsup
            if phase == "train":
                with open(os.path.join(self.project_path,
                                       fr"data\training\finance-sentiment-analysis-{phase}-v0.0.5.json"),
                          "r",
                          encoding="utf-8") as file:
                    unsup_sentiment_list = [json.loads(l.strip()) for l in file.readlines()]
                    unsup_sentiment_list = [{"input": dic["input"],
                                             "output": dic["output"],
                                             "instruction": dic["instruction"],
                                             "ID": dic["input"],
                                             "label": {"positive": 1, "negative": 0, "neutral": -1}[dic["output"]]} for
                                            dic in unsup_sentiment_list]
                self.save_data_list(unsup_sentiment_list + sup_sentiment_list, os.path.join(self.project_path,
                                                                                            rf'data\training\{self.dataset}-SAM-{phase}-d{self.data_version}-p{version}.json'))

        with open(
                fr"./data/agent_label/input/{self.dataset}-SA-news-d{self.data_version}-p{version}.tsv",
                "w",
                encoding="utf-8") as f:
            # Write the strings to the file
            print(f"Saving {len(news2ids_dict)} news")
            for news in news2ids_dict:
                f.write(json.dumps({"input": news, "id_list": news2ids_dict[news]}) + "\n")

        with open(
                fr"./data/agent_label/input/{self.dataset}-SA-expert-d{self.data_version}-p{version}.tsv",
                "w",
                encoding="utf-8") as f:
            print(f"Saving {len(instruct_list)} SA")
            for instruct in instruct_list:
                f.write(json.dumps(instruct) + "\n")

    def gen_lrr_prompt(self, sa_expert_path, ta_expert_path, pre_prompt_version, ground_prompt_version, mv_prompt_version, lrr_version):
        ground_score_prompt = self.load_prompt( os.path.join(self.project_path,
                                                             f'data\prompt\ground_{ground_prompt_version}.txt'))
        mv_prompt = self.load_prompt( os.path.join(self.project_path,
                                                   f'data\prompt\mv_{mv_prompt_version}.txt'))
        for phase in ["train","dev","test"]:
            record_dict = self.phase2record_dict[phase]
            record_dict = {r["ID"]: r for r in record_dict}
            # prompt_version = "v3"
            # move_prediction_to_end = False
            data_version = self.args.data_version
            dataset = self.args.dataset
            # 1. load rationale
            scrape_df = pd.read_csv(os.path.join(self.project_path, f'data\gpt_label\output\{pre_prompt_version}\W3_hanstong_d{dataset.replace("_","")}sataall{data_version}{phase}p{pre_prompt_version}_ScrapeV2WP.tsv'), sep='\t', header=0)
            id2answer_dict = {json.loads(scrape_df["PROMPT"][i])[0]["id"]: scrape_df["Result.res"][i] for i in range(len(scrape_df)) }
            invalid_list = []
            for id in id2answer_dict:
                try:
                    record_dict[id].update(self.parse_rationale(id2answer_dict[id]))
                except:
                    invalid_list.append(id)
                    if phase=="test":
                        record_dict[id].update({"Rationale":"","Analysis":""})

            # regen_invalid_list
            with open(os.path.join(self.project_path, rf'data\gpt_label\input\{pre_prompt_version}\W3_hanstong_d{dataset.replace("_","")}sataall{data_version}{phase}p{pre_prompt_version}_ScrapeV2WP.tsv'), "r",
                      encoding="utf-8") as file:
                line_list = file.readlines()
                line_list = [(json.loads(l.split("\t")[0])[0],l) for l in line_list]
            line_list = [l[1] for l in line_list if l[0]['id'] in invalid_list]
            with open(os.path.join(self.project_path, rf'data\gpt_label\input\{pre_prompt_version}\W3_hanstong_d{dataset.replace("_","")}sataall{data_version}{phase}p{pre_prompt_version}invalid_ScrapeV2WP.tsv'), 'w') as file:
                for entry in line_list:
                    file.write(entry)
            print("invalid list:", phase, invalid_list)

            # 2. load SA & TA result
            with open(os.path.join(self.project_path, rf'data\agent_label\output\{sa_expert_path}.tsv'), "r", encoding="utf-8") as f:
                line_list = f.readlines()
                line_list = [json.loads(l) for l in line_list]
                sa_expert_dict = {l["ID"]: l["RES_prediction"] for l in line_list}

            invalid_sent_list = []
            for id in record_dict:
                if id in sa_expert_dict:
                    record_dict[id]["sa_expert_output"] = sa_expert_dict[id]
                else:
                    record_dict[id]["sa_expert_output"] = "Unknown"
                    invalid_sent_list.append(id)
            print("invalid sentiment list:", phase, invalid_sent_list)

            with open(os.path.join(self.project_path, rf'data\agent_label\output\{ta_expert_path}.tsv'), "r", encoding="utf-8") as f:
                line_list = f.readlines()
                line_list = [json.loads(l) for l in line_list]
                ta_expert_dict = {l["ID"]: l["RES_prediction"] for l in line_list}

            for id in record_dict:
                record_dict[id]["ta_expert_output"] = ta_expert_dict[id]

            # 3. merge result with lrr result
            input_prompt_path = os.path.join(self.project_path, f'data\prompt\ploutosgpt_lrr_{lrr_version}_input.txt')
            output_prompt_path = os.path.join(self.project_path, f'data\prompt\ploutosgpt_lrr_{lrr_version}_output.txt')
            input_prompt = self.load_prompt(input_prompt_path)
            output_prompt = self.load_prompt(output_prompt_path)
            data_list = []
            faithfulness_eval_data_list = []
            movement_prompt_list= []

            for record_id in record_dict:
                if "Rationale" not in record_dict:
                    print("no valid rationale")
                    faithfulness_eval_data,movement_prompt =  [],[]
                else:
                    faithfulness_eval_data,movement_prompt = self.gen_faithfulness_prompt(record_dict["Rationale"], record_dict[record_id], mv_prompt , ground_score_prompt )
                faithfulness_eval_data_list.extend([{"id":record_id,"Prompt":d,"Max Length": 3000} for d in faithfulness_eval_data])
                movement_prompt_list.extend([{"id":record_id,"Prompt":d[0],"label":d[1],"Max Length": 300} for d in movement_prompt])

                if record_id in invalid_list and phase != "test":
                    continue

                record = record_dict[record_id]
                record["label"] = {"Up":1,"Down":0}[record["PredictionMovement"]]
                record["SENTIMENT_PREDICTION"] = self.parse_llm_result(record["sa_expert_output"])
                record["TECHNICAL_PREDICTION"] =  {1:"Up",0:"Down"}[record["ta_expert_output"]]

                # cut off news list
                record['COMPANY_NEWS'] = "\n".join(list(set(record['COMPANY_NEWS'].split("\n")))[:30])

                data = {"input": input_prompt.format(**record),
                        "output": output_prompt.format(**record),
                        "instruction": "", "ID": record["ID"], "label": record["label"]}
                data_list.append(data)
            self.save_data_list(data_list, os.path.join(self.project_path,
                                                        rf'data\training\{self.dataset}-lrr{lrr_version}-{phase}-d{self.data_version}-p{pre_prompt_version}.json'))
            if phase =="test":
                self.save_prompt_list(faithfulness_eval_data_list,os.path.join(self.project_path,
                                                                               rf'data\gpt_label\input\W3_hanstong_d{self.dataset.replace("_","")}truthgpt4{self.data_version}p{ground_prompt_version}_ScrapeV2WP.tsv'))
                self.save_prompt_list(movement_prompt_list,os.path.join(self.project_path,
                                                                        rf'data\gpt_label\input\W3_hanstong_d{self.dataset.replace("_","")}mv{self.data_version}p{mv_prompt_version}_ScrapeV2WP.tsv'))

    def gen_technical_prompt(self, version="v2.0"):
        print(f"generating technical prompt version:{version}")
        instruct_for_all_list = []
        for phase in self.phase_list:
            data_list = self.gen_prompt(phase, os.path.join(self.project_path,
                                                            rf'data\prompt\technical_analysis\ploutosgpt_technical_analysis_expert_{version}_input.txt'),
                                        os.path.join(self.project_path,
                                                     rf'data\prompt\technical_analysis\ploutosgpt_technical_analysis_expert_{version}_output.txt'),
                                        )
            instruct_for_all_list.extend(data_list)
            self.save_data_list(data_list, os.path.join(self.project_path,
                                                        rf'data\training\{self.dataset}-TA-{phase}-d{self.data_version}-p{version}.json'))
        with open(
                fr"./data/agent_label/input/{self.dataset}-TA-expert-d{self.data_version}-p{version}.tsv",
                "w",
                encoding="utf-8") as f:
            print(f"Saving {len(instruct_for_all_list)} TA")
            for instruct in instruct_for_all_list:
                f.write(json.dumps(instruct) + "\n")

    def get_sata_prompt(self, version="v3.0"):
        for phase in self.phase_list:
            input_prompt_path = os.path.join(self.project_path,
                                             f'data\prompt\scrape_reasoning\ploutosgpt_nounder_{version}.txt')
            save_path = os.path.join(self.project_path,
                                     rf'data\gpt_label\input\{version}\W3_hanstong_d{self.dataset.replace("_", "")}sataall{self.data_version}{phase}p{version}_ScrapeV2WP.tsv')

            if not os.path.exists(os.path.join(self.project_path, rf'data\gpt_label\input\{version}')):
                os.mkdir(os.path.join(self.project_path, rf'data\gpt_label\input\{version}'))

            input_prompt = self.load_prompt(input_prompt_path)
            record_list = self.phase2record_dict[phase]

            for i in range(len(record_list)):
                record_list[i]['COMPANY_NEWS'] = "\n".join(list(set(record_list[i]['COMPANY_NEWS'].split("\n")))[:30])

            prompt_list = [{"id": record["ID"], "Prompt": input_prompt.format(**record), "Max Length": 3000} for record
                           in record_list]
            self.save_prompt_list(prompt_list, save_path)

    def save_prompt_list(self, data_list, path):
        print(f"Save {len(data_list)} data to path {path}")

        with open(path, "w", encoding="utf-8") as f:
            for d in data_list:
                cur_data = {"Temperature": 0.0, "Top P": 1.0, "Frequency Penalty": 1.0,
                            "Presence Penalty": 1.0}
                cur_data.update(d)
                f.write(json.dumps([cur_data]) + "\t" + "800" + "\t" + "{}" + "\n")

        if len(data_list) > 10000:
            split_file(path, 3000)

    def save_data_list(self, line_list, path):
        print(f"Save {len(line_list)} data to path {path}")
        with open(path, "w", encoding="utf-8") as f:
            for new_line in line_list:
                f.write(json.dumps(new_line) + "\n")

    def load_prompt(self, path):
        cur_prompt = ""
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                cur_prompt += line
        return cur_prompt
