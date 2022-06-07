import json
import argparse
import jsonlines

import prettytable as pt

from tqdm import tqdm
from statistics import mean
from nltk.tokenize import word_tokenize

def read_jsonlines(eval_file_name):
    lines = []
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def evaluate_top_k_hit(results, gt_answers, max_token_num=5000, difficulty_dict={}):
    per_lang = {}
    for item in tqdm(results):
        q_id = item["q_id"]
        lang = item["lang"]
        per_lang.setdefault(lang, {"count": 0, "hit": 0})
        ctxs = item["ctxs"]

        if q_id not in gt_answers:
            continue

        answers = gt_answers[q_id]

        span_answers = []
        # Skip yes/no examples during XOR-Retrieve evaluations
        for answer in answers:
            if answer not in ["yes", "no"]:
                span_answers.append(answer)
        if len(span_answers) == 0:
            continue

        per_lang[lang]["count"] += 1

        concat_string_tokens = []
        for ctx_text in ctxs:
            tokenized_text = word_tokenize(ctx_text)
            concat_string_tokens += tokenized_text
            if len(concat_string_tokens) >= max_token_num:
                break
        concat_string_tokens = concat_string_tokens[:max_token_num]
        concat_string = " ".join(concat_string_tokens)
        hit = False
        for answer in span_answers:
            if answer in concat_string:
                hit = True
        if hit is True:
            per_lang[lang]["hit"] += 1

    final_results = {lang: result for lang,
                     result in per_lang.items() if result["count"] > 0}

    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/xorqa/dev.jsonl", type=str)
    parser.add_argument("--pred_file", default=None, type=str)
    parser.add_argument("--max_token_num", default=5000, type=int)

    args = parser.parse_args()
    predictions = json.load(open(args.pred_file))
    input_data = read_jsonlines(args.data_file)
    # convert input open-domain data into the qid2answer dictionary
    qid2answers = {item["id"]: item["answers"] for item in input_data}

    table = pt.PrettyTable()
    table.field_names = ['metrics', 'ar', 'bn', 'fi', 'ja', 'ko', 'ru', 'te', 'avg']

    for topk in [2, 5]:
        pred_per_lang_results = evaluate_top_k_hit(
            predictions, qid2answers, topk * 1000)
        scores = []
        for lang in ['ar', 'bn', 'fi', 'ja', 'ko', 'ru', 'te']:
            per_lang_score = (pred_per_lang_results[lang]["hit"] / pred_per_lang_results[lang]["count"]) * 100
            scores.append(per_lang_score)
        scores.append(mean(scores))
        table.add_row(['R@%dk' % topk] + ['%.1f' % s for s in scores])
    
    print(table)


if __name__ == "__main__":
    main()