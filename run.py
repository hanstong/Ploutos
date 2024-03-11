import argparse
import time
from config_loader import PathParser
from data_pipe import DataPipePrompt
import pickle
import os
from prompt_gen import PromptGen

def run_record_gen(config, args):
    """
    Generates and saves data records for different datasets.
    Args:
        config: Configuration settings for the project.
        args: Command-line arguments or other configurations passed to the function.

    The function iterates over the specified phases ('train', 'dev', 'test'), generates records using
    the DataPipePrompt class, and saves them as pickle files.
    """
    pipe = DataPipePrompt(config, args)
    version = args.data_version # v003
    phase_list = ["train", "dev", "test"]
    for phase in phase_list:
        if phase in ["dev", "test"] and args.align_devtest==1:
            args.alignment = 1
        st = time.time()
        print(f"=====Start Processing {phase} =====")
        g_list = pipe.sample_gen_multiprocess(phase=phase, num_processors=args.num_processors)
        with open(os.path.join(config.root, f'data\info\info-{args.dataset}-{phase}-{version}.pkl'), 'wb') as handle:
            pickle.dump(g_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"=====Finish Processing {phase} with Count {len(g_list)} and Time {time.time() - st} =====")

def run_prompt_gen(config, args):
    """
    Generates prompts for technical, sentiment analysis and other agents.
    Args:
        config: Configuration settings for the project.
        args: Command-line arguments or other configurations passed to the function.
    """
    pg = PromptGen(args, config, data_version=args.data_version)
    pg.gen_technical_prompt(version="v8.0")
    pg.gen_sentiment_prompt(version="v5.0")
    pg.get_sata_prompt(version="v6.0")

def run_lrr_prompt_gen(config, args):
    """
    Generates PloutosGen prompts.
    Args:
        config: Configuration settings for the project.
        args: Command-line arguments or other configurations passed to the function.
    """
    pg = PromptGen(args, config, data_version=args.data_version)
    pg.gen_lrr_prompt(sa_expert_path=f"result_{args.datasets}-SAM-train-dv004-pv5.0",
                      ta_expert_path=f"result_{args.datasets}-TA-train-dv004-pv8.0",
                      pre_prompt_version="v5.0",
                      ground_prompt_version = "v2.0",
                      mv_prompt_version = "v2.0",
                      lrr_version = "v3.0")

    pg.gen_ground_prompt(args.path,
                         ground_prompt_version = "v2.0",ground_name = "ploutos")

def main(config, args):
    if args.task == "gen_record":
        run_record_gen(config, args)
    elif args.task == "gen_prompt":
        run_prompt_gen(config, args)
    elif args.task == "gen_lrr_prompt":
        run_lrr_prompt_gen(config, args)
    else:
        raise RuntimeError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')
    parser.add_argument('--data_version', type=str, help='use alignment', default="data_version")
    parser.add_argument('--alignment', type=int, help='use alignment', default=1)
    parser.add_argument('--align_devtest', type=int, help='use alignment', default=1)
    parser.add_argument('--dedup_news', type=int, help='use dedup', default=0)
    parser.add_argument('--dataset', type=str, help='stocknet,cmin_us,cmin_cn', default='stocknet')
    parser.add_argument('--task', type=str, help='gen_record,gen_prompt,gen_lrr_prompt', default='gen_record')
    parser.add_argument('--path', type=str, help='gen_record,gen_prompt,gen_lrr_prompt', default='path for lrr prompt gen')
    parser.add_argument('--num_processors', type=int, help='gen_record,gen_prompt', default=16)
    parser.add_argument('--debug', default=False, action="store_true", help='True for word, False for char')
    args = parser.parse_args()
    if args.debug:
        args.num_processors = None
    config = PathParser(config_name=f"config_{args.dataset}.yml")
    main(config, args)
