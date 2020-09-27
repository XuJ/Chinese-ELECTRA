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


def eval_bagging_best_score(data_dir, split, task_name, selected_idx):
  if task_name == 'ccks42ee':
    model_name_part = 'electra_ensemble'
  else:
    model_name_part = 'electra_ensemble2'
  all_models = []
  for batch_size in [24, 32]:
    for max_seq_length in [384, 480, 512]:
      for epoch in [2, 3]:
        model_name = "{}_{}_{}_{}".format(model_name_part, batch_size, max_seq_length, epoch)
        all_models.append(model_name)
  models = [all_models[int(x)] for x in selected_idx.split('-')]

  all_nbest = []
  all_odds = []
  for dire in [os.path.join(data_dir, d) for d in models]:
    all_nbest.append(utils.load_pickle((os.path.join(dire, 'models', 'electra_large', 'results', '{}_qa'.format(task_name),
      '{}_{}_all_nbest.pkl'.format(task_name, split)))))
    all_odds.append(utils.load_json((os.path.join(dire, 'models', 'electra_large', 'results', '{}_qa'.format(task_name),
      '{}_{}_null_odds.json'.format(task_name, split)))))

  qid_answers = collections.OrderedDict()
  qid_questions = collections.OrderedDict()
  dataset = utils.load_json(
    (os.path.join(data_dir, model_name_part, 'finetuning_data', task_name, '{}.json'.format(split))))['data']
  for article in dataset:
    for paragraph in article["paragraphs"]:
      for qa in paragraph['qas']:
        _qid = qa['id']
        qid_answers[_qid] = qa['answers'] if 'answers' in qa else ''
        qid_questions[_qid] = qa['question']

  all_nbest = filter_short_ans(all_nbest)
  output_dir = os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging', task_name)

  vote1(dataset, all_nbest, all_odds, qid_answers, split, output_dir)
  vote2(dataset, all_nbest, all_odds, qid_answers, split, output_dir)
  vote3(dataset, all_nbest, all_odds, qid_answers, qid_questions, models, split, output_dir)


def filter_short_ans(all_nbest):
  for nbest in all_nbest:
    for qid, answers in nbest.items():
      new_answers = []
      for ans in answers:
        if len(ans['text']) > 1:
          new_answers.append(ans)
      nbest[qid] = new_answers
  return all_nbest


def vote1(dataset, all_nbest, all_odds, qid_answers, split, output_dir):
  bagging_preds = collections.OrderedDict()
  bagging_odds = collections.OrderedDict()
  bagging_all_nbest = collections.OrderedDict()

  for qid in qid_answers:
    bagging_preds[qid] = \
      (seq([nbest[qid][0] for nbest in all_nbest]).sorted(key=lambda x: x['probability'])).list()[-1]['text']
    bagging_all_nbest[qid] = \
      [(seq([nbest[qid][0] for nbest in all_nbest]).sorted(key=lambda x: x['probability'])).list()[-1]]
    bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

  utils.write_json(bagging_preds, os.path.join(output_dir, 'vote1', 'ccks42bagging_{}_preds.json'.format(split)))
  utils.write_pickle(bagging_all_nbest, os.path.join(output_dir, 'vote1', 'ccks42bagging_{}_all_nbest.pkl'.format(split)))
  utils.write_json(bagging_odds, os.path.join(output_dir, 'vote1', 'ccks42bagging_{}_null_odds.json'.format(split)))

  if split in ['train', 'dev']:
    out_eval = main2(dataset, bagging_preds, bagging_odds)
    utils.log('vote1')
    utils.log(out_eval)
  elif split == 'eval':
    for qid in bagging_preds.keys():
      if bagging_odds[qid] > -2.75:
        bagging_preds[qid] = ""
    utils.write_json(bagging_preds, os.path.join(output_dir, 'vote1', 'ccks42bagging_{}_1_preds.json'.format(split)))
  else:
    utils.log('{} split is not supported'.format(split))


def vote2(dataset, all_nbest, all_odds, qid_answers, split, output_dir):
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

  utils.write_json(bagging_preds, os.path.join(output_dir, 'vote2', 'ccks42bagging_{}_preds.json'.format(split)))
  utils.write_json(bagging_odds, os.path.join(output_dir, 'vote2', 'ccks42bagging_{}_null_odds.json'.format(split)))

  if split in ['train', 'dev']:
    out_eval = main2(dataset, bagging_preds, bagging_odds)
    utils.log('vote2')
    utils.log(out_eval)
  elif split == 'eval':
    for qid in bagging_preds.keys():
      if bagging_odds[qid] > -2.75:
        bagging_preds[qid] = ""
    utils.write_json(bagging_preds, os.path.join(output_dir, 'vote2', 'ccks42bagging_{}_1_preds.json'.format(split)))
  else:
    utils.log('{} split is not supported'.format(split))


def vote3(dataset, all_nbest, all_odds, qid_answers, qid_questions, models, split, output_dir):
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

  utils.write_json(bagging_preds, os.path.join(output_dir, 'vote3', 'ccks42bagging_{}_preds.json'.format(split)))
  utils.write_json(bagging_odds, os.path.join(output_dir, 'vote3', 'ccks42bagging_{}_null_odds.json'.format(split)))

  if split in ['train', 'dev']:
    out_eval = main2(dataset, bagging_preds, bagging_odds)
    utils.log('vote3')
    utils.log(out_eval)
  elif split == 'eval':
    for qid in bagging_preds.keys():
      if bagging_odds[qid] > -2.75:
        bagging_preds[qid] = ""
    utils.write_json(bagging_preds, os.path.join(output_dir, 'vote3', 'ccks42bagging_{}_1_preds.json'.format(split)))
  else:
    utils.log('{} split is not supported'.format(split))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", required=True, help="location of all models")
  parser.add_argument("--split", required=True, help="dataset: train/dev/eval dataset")
  parser.add_argument("--task", required=True, help="task_name: ccks42ee/ccks42single/ccks42multi")
  parser.add_argument("--selected", default='0-1-2-3-4-5-6-7-8-9-10-11', help="selected_idx: selected model idx join by -")
  args = parser.parse_args()
  eval_bagging_best_score(args.dir, args.split, args.task, args.selected)


if __name__ == '__main__':
  main()
