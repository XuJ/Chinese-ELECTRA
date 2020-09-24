# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 10:14
# @Author  : jiaoxu
# @email   : jiaoxu@myhexin.com
import argparse
import collections
import os

import numpy as np
from functional import seq

from finetune.qa.squad_official_eval import main2
from util import utils


def eval_bagging_best_score(data_dir):
  all_models = []
  for task_idx in ['', 2]:
    for batch_size in [24, 32]:
      for max_seq_length in [384, 480, 512]:
        for epoch in [2, 3]:
          model_name = "electra_ensemble{}_{}_{}_{}".format(task_idx, batch_size, max_seq_length, epoch)
          all_models.append(model_name)
  models = [all_models[x] for x in [0, 1, 2, 3, 4, 5]]

  all_nbest = []
  all_odds = []
  all_preds = []
  for dire in [os.path.join(data_dir, d) for d in models]:
    all_nbest.append(utils.load_pickle(
      (os.path.join(dire, 'models', 'electra_large', 'results', 'ccks42ee_qa', 'ccks42ee_dev_all_nbest.pkl'))))
    all_odds.append(utils.load_json(
      (os.path.join(dire, 'models', 'electra_large', 'results', 'ccks42ee_qa', 'ccks42ee_dev_null_odds.json'))))
    all_preds.append(utils.load_json(
      (os.path.join(dire, 'models', 'electra_large', 'results', 'ccks42ee_qa', 'ccks42ee_dev_preds.json'))))
  qids = seq(all_preds[0].keys()).list()

  qid_answers = collections.OrderedDict()
  qid_questions = collections.OrderedDict()
  dataset = utils.load_json((os.path.join(data_dir, 'electra_ensemble', 'finetuning_data', 'ccks42ee', 'dev.json')))[
    'data']
  for article in dataset:
    for paragraph in article["paragraphs"]:
      for qa in paragraph['qas']:
        _qid = qa['id']
        qid_answers[_qid] = qa['answers']
        qid_questions[_qid] = qa['question']

  all_nbest = filter_short_ans(all_nbest)
  vote1(dataset, all_nbest, all_odds, qid_answers)
  vote2(dataset, all_nbest, all_odds, qid_answers)
  vote_with_post_processing(dataset, all_nbest, all_odds, qid_answers, qid_questions, models)


def filter_short_ans(all_nbest):
  for nbest in all_nbest:
    for qid, answers in nbest.items():
      new_answers = []
      for ans in answers:
        if len(ans['text']) > 1:
          new_answers.append(ans)
      nbest[qid] = new_answers
  return all_nbest


def vote1(dataset, all_nbest, all_odds, qid_answers):
  bagging_preds = collections.OrderedDict()
  bagging_odds = collections.OrderedDict()

  for qid in qid_answers:
    bagging_preds[qid] = \
      (seq([nbest[qid][0] for nbest in all_nbest]).sorted(key=lambda x: x['probability'])).list()[-1]['text']
    bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

  # json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
  # json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

  out_eval = main2(dataset, bagging_preds, bagging_odds)
  utils.log('vote1')
  utils.log(out_eval)


def vote2(dataset, all_nbest, all_odds, qid_answers):
  bagging_preds = collections.OrderedDict()
  bagging_odds = collections.OrderedDict()

  for qid in qid_answers:
    preds_scores = (seq(all_nbest).map(lambda x: x[qid][0]).map(lambda x: (x['text'], x['probability']))).dict()
    compare = collections.defaultdict(lambda: 0.)
    for pred, score in preds_scores.items():
      compare[pred] += score
    compare = seq(compare.items()).sorted(lambda x: x[1]).reverse().list()
    bagging_preds[qid] = compare[0][0]

    bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

  # json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
  # json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

  out_eval = main2(dataset, bagging_preds, bagging_odds)
  utils.log('vote2')
  utils.log(out_eval)


def vote_with_post_processing(dataset, all_nbest, all_odds, qid_answers, qid_questions, models):
  bagging_preds = collections.OrderedDict()
  bagging_odds = collections.OrderedDict()

  def post_process(question, candi, weight=1):
    question = question.lower()
    first_token = candi['text'].split()[0]
    th = 0.
    if "when" in question:
      if first_token in ['before', 'after', 'about', 'around', 'from', 'during']:
        candi['probability'] += th
    elif "where" in question:
      if first_token in ['in', 'at', 'on', 'behind', 'from', 'through', 'between', 'throughout']:
        candi['probability'] += th
    elif "whose" in question:
      if "'s" in candi['text']:
        candi['probability'] += th
    elif "which" in question:
      if first_token == "the":
        candi['probability'] += th
    candi['probability'] *= weight
    return candi

  cof = 0.2

  for qid in qid_answers:
    question = qid_questions[qid]
    post_process_candidates = (
      seq(zip(all_nbest, models)).map(lambda x: (x[0][qid], cof if 'lr_epoch_results' in x[1] else 1.)).map(
        lambda x: seq(x[0]).map(lambda y: post_process(question, y, x[1])).list()).flatten()).list()
    preds_probs = collections.defaultdict(lambda: [])
    for pred in post_process_candidates:
      preds_probs[pred['text']].append(pred['probability'])
    for pred in post_process_candidates:
      preds_probs[pred['text']] = np.mean(preds_probs[pred['text']]).__float__()
    bagging_preds[qid] = (seq(preds_probs.items()).sorted(lambda x: x[1]).reverse().map(lambda x: x[0])).list()[0]
    bagging_odds[qid] = np.mean(
      [odds[qid] * cof if 'lr_epoch_results' in model else odds[qid] for odds, model in zip(all_odds, models)])

  out_eval = main2(dataset, bagging_preds, bagging_odds)
  utils.log('vote_with_post_processing')
  utils.log(out_eval)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", required=True, help="location of all models")
  args = parser.parse_args()
  eval_bagging_best_score(args.dir)


if __name__ == '__main__':
  main()
