
from datetime import date
import csv
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import timedelta
import pandas as pd
from datetime import datetime
import math
import os
import random

def plot_price_df(final_df, stock_name):
    fig, ax = plt.subplots(figsize=(15,8))
    ax.plot(final_df['Date'], final_df['Close'], color='#008B8B')
    ax.set(xlabel="Date", ylabel="USD", title=f"{stock_name} Stock Price")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.show()

def bin_mapping(ret):
    ret = float(ret)
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    return up_down + (str(integer) if integer <= 5 else '5+')

def n_weeks_before(date_string, n):
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    return date.strftime("%Y-%m-%d")

def get_curday():
    return date.today().strftime("%Y-%m-%d")

def initialize_csv(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer"])

def sample_news(news, k=5):
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]

def map_bin_label(bin_lb):
    lb = bin_lb.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    if lb.endswith('+'):
        lb = lb.replace('5', 'more than 5%')
    #         lb = lb.replace('5+', '5+%')
    else:
        lb = lb.replace('5', '4-5%')

    return lb

def append_to_csv(filename, input_data, output_data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input_data, output_data])

def split_file(file_path, lines_per_file):
    # Make sure lines_per_file is a positive integer
    if lines_per_file <= 0:
        raise ValueError("lines_per_file must be a positive integer")

    # Open the large file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        print("length of file", len(lines))

    # Open the large file
    with open(file_path, 'r') as file:
        # Read the file line by line
        for i, line in enumerate(file):
            # Every lines_per_file lines, create a new file
            if i % lines_per_file == 0:
                if i > 0:  # Close the previous file if it exists
                    small_file.close()
                small_file_path = file_path.replace("train", f"trainpart{i // lines_per_file + 1}")
                small_file = open(small_file_path, 'w')
            small_file.write(line)
        # Close the last small file
        small_file.close()

def merge_files_from_dir(merged_file_path, directory):
    # List all files in the directory that contain 'part' in their filenames
    part_files = [f for f in os.listdir(directory) if 'part' in f]
    # Sort the files by part number to ensure correct order
    part_files.sort(key=lambda f: int(f.split('part')[-1].split("_")[0].split("pv")[0]))

    all_list = []
    for part_file in part_files:
        part_file_path = os.path.join(directory, part_file)
        scrape_df = pd.read_csv(part_file_path, sep='\t',header=0)
        all_list.append(scrape_df)

    merge_df = pd.concat(all_list, axis=0, ignore_index=True)
    print("length of file", str(merge_df.shape))
    merge_df.to_csv(merged_file_path, sep = '\t',index=False,header=True,encoding='utf-8')