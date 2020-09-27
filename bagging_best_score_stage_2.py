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


def eval_bagging_best_score(data_dir, split):
  all_nbest = []
  all_odds = []

  tmp_nbest_dict = utils.load_pickle((
    os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging', 'ccks42single',
      'vote1', 'ccks42bagging_{}_all_nbest.pkl'.format(split))))
  tmp_odds_dict = utils.load_json((
    os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging', 'ccks42single',
      'vote1', 'ccks42bagging_{}_null_odds.json'.format(split))))
  tmp_nbest_dict.update(utils.load_pickle((
    os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging', 'ccks42multi',
      'vote1', 'ccks42bagging_{}_all_nbest.pkl'.format(split)))))
  tmp_odds_dict.update(utils.load_json((
    os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging', 'ccks42multi',
      'vote1', 'ccks42bagging_{}_null_odds.json'.format(split)))))
  all_nbest.append(tmp_nbest_dict)
  all_odds.append(tmp_odds_dict)
  all_nbest.append(utils.load_pickle((
    os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging', 'ccks42ee', 'vote1',
      'ccks42bagging_{}_all_nbest.pkl'.format(split)))))
  all_odds.append(utils.load_json((
    os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging', 'ccks42ee', 'vote1',
      'ccks42bagging_{}_null_odds.json'.format(split)))))

  qid_answers = collections.OrderedDict()
  qid_questions = collections.OrderedDict()
  dataset = utils.load_json(
    (os.path.join(data_dir, 'electra_ensemble', 'finetuning_data', 'ccks42ee', '{}.json'.format(split))))['data']
  for article in dataset:
    for paragraph in article["paragraphs"]:
      for qa in paragraph['qas']:
        _qid = qa['id']
        qid_answers[_qid] = qa['answers'] if 'answers' in qa else ''
        qid_questions[_qid] = qa['question']

  output_dir = os.path.join(data_dir, 'electra_best', 'models', 'electra_large', 'results', 'ccks42bagging',
    'final_result')

  vote1(dataset, all_nbest, all_odds, qid_answers, split, output_dir)


def vote1(dataset, all_nbest, all_odds, qid_answers, split, output_dir):
  bagging_preds = collections.OrderedDict()
  bagging_odds = collections.OrderedDict()

  for qid in qid_answers:
    bagging_preds[qid] = \
      (seq([nbest[qid][0] for nbest in all_nbest]).sorted(key=lambda x: x['probability'])).list()[-1]['text']
    bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

  utils.write_json(bagging_preds, os.path.join(output_dir, 'vote1', 'ccks42bagging_{}_preds.json'.format(split)))
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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", required=True, help="location of all models")
  parser.add_argument("--split", required=True, help="dataset: train/dev/eval dataset")
  args = parser.parse_args()
  eval_bagging_best_score(args.dir, args.split)


if __name__ == '__main__':
  main()
