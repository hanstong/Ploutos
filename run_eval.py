import os, sys, pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
from eval_models import *
from load_model_util import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gen_eval_result(args, model, tokenizer, batch_size, eval_set):
    res = None
    line_list = []
    if eval_set == "tspairnumprice":
        res, line_list = TimeSeriesNumPricePredictionCompareEval(model, tokenizer, batch_size,
                                                                 dataset_key=args.input_path,
                                                                 max_length=args.max_length).eval()
    elif eval_set == "tspairprice":
        res, line_list = TimeSeriesPricePredictionCompareEval(model, tokenizer, batch_size,
                                                              dataset_key=args.input_path,
                                                              max_length=args.max_length).eval()
    elif eval_set == "noagentprice":
        res, line_list = NoAgentPricePredictionEval(model, tokenizer, batch_size, dataset_key=args.input_path,
                                                    max_length=args.max_length,temperature=args.temperature,model_args=args ).eval()
    elif eval_set == "fpb":
        res, line_list = FPBSentimentEval(model, tokenizer, batch_size, max_length=args.max_length, prompt_version = args.sentiment_prompt_version).eval()
    elif eval_set == "fiqa":
        res, line_list = FIQASentimentEval(model, tokenizer, batch_size, max_length=args.max_length, prompt_version = args.sentiment_prompt_version).eval()
    elif eval_set == "tfns":
        res, line_list = TFNSSentimentEval(model, tokenizer, batch_size, max_length=args.max_length, prompt_version = args.sentiment_prompt_version).eval()
    elif eval_set == "nwgi":
        res, line_list = NWGISentimentEval(model, tokenizer, batch_size, max_length=args.max_length, prompt_version = args.sentiment_prompt_version).eval()
    elif eval_set == "btsa":
        res, line_list = BTSASentimentEval(model, tokenizer, batch_size, max_length=args.max_length, prompt_version = args.sentiment_prompt_version).eval()
    elif eval_set == "sentprice":
        news_list, prediction_list = SentimentPricePredictionLabeling(model, tokenizer, batch_size,
                                                                      dataset_key=args.input_path,
                                                                      max_length=args.max_length, prompt_version = args.sentiment_prompt_version).label()
        se = SentimentPricePredictionEval(model, tokenizer, batch_size,
                                          max_length=args.max_length,  dataset_key=args.input_path)
        se.load_input2label_dict({k: v for k, v in zip(news_list, prediction_list)})
        res, line_list = se.eval()
    elif eval_set == "sentinstructprice":
        res, line_list = SentimentInstrucPricePredictionEval(model, tokenizer, batch_size,
                                                              dataset_key=args.input_path,
                                                              max_length=args.max_length).eval()
    elif eval_set == "file_sent_label":
        res = SentimentAnalysisFileLabeling(model, tokenizer, batch_size, args.max_length, args.input_path, prompt_version = args.sentiment_prompt_version).label()
    elif eval_set == "file_sentinstruct_label":
        res = LLMFileLabeling(model, tokenizer, batch_size, args.max_length, args.input_path).label()
    elif eval_set == "file_general_label":
        res = GeneralLLMFileLabeling(model, tokenizer, batch_size, args.max_length, args.input_path).label()
    elif eval_set == "file_ta_label":
        res = TechnicalAnalysisFileLabeling(model, tokenizer, batch_size, args.max_length, args.input_path).label()
    elif eval_set == "file_tanum_label":
        res = TechnicalAnalysisFileLabeling(model, tokenizer, batch_size, args.max_length, args.input_path).label()
    return res, line_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model", type=str, default="auto", help="model type")
    parser.add_argument("--pad_dir", type=str, default="left")
    parser.add_argument("--task_name", type=str, default="task")
    parser.add_argument("--sentiment_prompt_version", type=str, default="0")
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--input_path", type=str, default=r"data/training/stocknet-lrrv3.0-dev-dv004-pv5.0.json")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dataset_key", type=str, default="finance-overall-analysis-dev-v0.0.2", help="")
    parser.add_argument("--eval_set", type=str, default="noagentprice",
                        help="eval set name")
    args = parser.parse_args()
    args.eval_set = args.eval_set.split( ",")[0]
    if args.model == "auto":
        model, tokenizer = load_model_auto(args.path, pad_dir=args.pad_dir)
        model = model.cuda()
        print(count_parameters(model))
    elif args.model == "fake":
        model, tokenizer = load_fake_model()
        model = model.cuda()
    elif args.model == "vllm":
        from vllm import LLM
        if torch.cuda.device_count() > 1:
            model = LLM(model=args.path, dtype="float16", tensor_parallel_size=torch.cuda.device_count())
        else:
            model = LLM(model=args.path, dtype="float16")
        tokenizer = None
    else:
        raise RuntimeError
    batch_size = args.batch_size

    if "label" in args.eval_set:
        label_task = args.eval_set
        prediction_list, _ = gen_eval_result(args, model, tokenizer, batch_size, label_task)
        with open(os.path.join(args.output_dir, f"result_{args.eval_set}.tsv"), "w", encoding="utf-8") as f:
            for prediction in prediction_list:
                f.write(prediction + "\n")

    else:
        metric_line_list = []
        metric, debug_line_list = gen_eval_result(args, model, tokenizer, batch_size, args.eval_set)
        metric["name"] = args.eval_set
        print(metric)
        metric_line_list.append(json.dumps(metric))

        with open(os.path.join(args.output_dir, f"result_{args.eval_set}.tsv"), "w", encoding="utf-8") as f:
            for metric in metric_line_list:
                f.write(metric + "\n")

        with open(os.path.join(args.output_dir,f"debug_{args.eval_set}.tsv"), "w", encoding="utf-8") as f:
            for metric in debug_line_list:
                f.write(metric + "\n")
