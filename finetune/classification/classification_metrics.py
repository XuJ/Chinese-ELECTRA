# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation metrics for classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import abc
import numpy as np
import scipy
import sklearn
import configure_finetuning

from finetune import scorer
from util import utils


class SentenceLevelScorer(scorer.Scorer):
  """Abstract scorer for classification/regression tasks."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(SentenceLevelScorer, self).__init__()
    self._total_loss = 0
    self._true_labels = []
    self._preds = []

  def update(self, results):
    super(SentenceLevelScorer, self).update(results)
    self._total_loss += results['loss']
    self._true_labels.append(results['label_ids'] if 'label_ids' in results
                             else results['targets'])
    self._preds.append(results['predictions'])

  def get_loss(self):
    return self._total_loss / len(self._true_labels)


class AccuracyScorer(SentenceLevelScorer):

  def _get_results(self):
    correct, count = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      count += 1
      correct += (1 if y_true == pred else 0)
    return [
        ('accuracy', 100.0 * correct / count),
        ('loss', self.get_loss()),
    ]


class F1Scorer(SentenceLevelScorer):
  """Computes F1 for classification tasks."""

  def __init__(self):
    super(F1Scorer, self).__init__()
    self._positive_label = 1

  def _get_results(self):
    n_correct, n_predicted, n_gold = 0, 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      if pred == self._positive_label:
        n_gold += 1
        if pred == self._positive_label:
          n_predicted += 1
          if pred == y_true:
            n_correct += 1
    if n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * n_correct / n_predicted
      r = 100.0 * n_correct / n_gold
      f1 = 2 * p * r / (p + r)
    return [
        ('precision', p),
        ('recall', r),
        ('f1', f1),
        ('loss', self.get_loss()),
    ]


class ModifiedAccuracyScorer(SentenceLevelScorer):
  """Computes Modified Accuracy Score for classification tasks."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, task, split):
    super(ModifiedAccuracyScorer, self).__init__()
    self._positive_label = 1
    self._config = config
    self._task = task
    self._name = task.name
    self._split = split
    self._eval_examples = task.get_examples(split)
    self._label_list = task.get_label_list()

  def _get_results(self):
    self.write_predictions()
    correct, count = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      count += 1
      correct += (1 if y_true == pred else 0)
    return [
      ('accuracy', 100.0 * correct / count),
      ('loss', self.get_loss()),
    ]

  def write_predictions(self):
    """Write final predictions to the json file."""
    all_predictions = collections.OrderedDict()
    unique_id_to_text_a = {}
    label_id_to_label_text = {}
    for example in self._eval_examples:
      unique_id_to_text_a[example.eid] = example.text_a
    for label_id, label_text in enumerate(self._label_list):
      label_id_to_label_text[label_id] = label_text
    for eid, pred in zip(self._eid, self._preds):
      text_a = unique_id_to_text_a[eid]
      label_text = label_id_to_label_text[int(pred)]
      all_predictions[text_a] = label_text
    utils.write_json(dict(all_predictions),
                     self._config.cl_preds_file(self._name+"_"+self._split))


class MCCScorer(SentenceLevelScorer):

  def _get_results(self):
    return [
        ('mcc', 100 * sklearn.metrics.matthews_corrcoef(
            self._true_labels, self._preds)),
        ('loss', self.get_loss()),
    ]


class RegressionScorer(SentenceLevelScorer):

  def _get_results(self):
    preds = np.array(self._preds).flatten()
    return [
        ('pearson', 100.0 * scipy.stats.pearsonr(
            self._true_labels, preds)[0]),
        ('spearman', 100.0 * scipy.stats.spearmanr(
            self._true_labels, preds)[0]),
        ('mse', np.mean(np.square(np.array(self._true_labels) - self._preds))),
        ('loss', self.get_loss()),
    ]
