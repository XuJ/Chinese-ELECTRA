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

"""Metrics for sequence tagging tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import collections
import configure_finetuning

import numpy as np

from finetune import scorer
from finetune.tagging import tagging_utils
from util import utils


class WordLevelScorer(scorer.Scorer):
  """Base class for tagging scorers."""
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(WordLevelScorer, self).__init__()
    self._total_loss = 0
    self._total_words = 0
    self._eids = []
    self._labels = []
    self._preds = []
    self._labeled_positions = []

  def update(self, results):
    super(WordLevelScorer, self).update(results)
    self._total_loss += results['loss']
    n_words = int(round(np.sum(results['labels_mask'])))
    self._eids.append(results['eid'])
    self._labels.append(results['labels'][:n_words])
    self._preds.append(results['predictions'][:n_words])
    self._labeled_positions.append(results['labeled_positions'][:n_words])
    # self._total_loss += np.sum(results['loss'])
    self._total_words += n_words

  def get_loss(self):
    return self._total_loss / max(1, self._total_words)


class AccuracyScorer(WordLevelScorer):
  """Computes accuracy scores."""

  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()
    self._auto_fail_label = auto_fail_label

  def _get_results(self):
    correct, count = 0, 0
    for labels, preds in zip(self._labels, self._preds):
      for y_true, y_pred in zip(labels, preds):
        count += 1
        correct += (1 if y_pred == y_true and y_true != self._auto_fail_label
                    else 0)
    return [
        ('accuracy', 100.0 * correct / count),
        ('loss', self.get_loss())
    ]


class F1Scorer(WordLevelScorer):
  """Computes F1 scores."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(F1Scorer, self).__init__()
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0

  def _get_results(self):
    if self._n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * self._n_correct / self._n_predicted
      r = 100.0 * self._n_correct / self._n_gold
      f1 = 2 * p * r / (p + r)
    return [
        ('precision', p),
        ('recall', r),
        ('f1', f1),
        ('loss', self.get_loss()),
    ]


class EntityLevelF1Scorer(F1Scorer):
  """Computes F1 score for entity-level tasks such as NER."""

  def __init__(self, label_mapping):
    super(EntityLevelF1Scorer, self).__init__()
    self._inv_label_mapping = {v: k for k, v in six.iteritems(label_mapping)}

  def _get_results(self):
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    for labels, preds in zip(self._labels, self._preds):
      sent_spans = set(tagging_utils.get_span_labels(
          labels, self._inv_label_mapping))
      span_preds = set(tagging_utils.get_span_labels(
          preds, self._inv_label_mapping))
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)
    return super(EntityLevelF1Scorer, self)._get_results()


class ModifiedEntityLevelF1Scorer(F1Scorer):
  """Computes Modified F1 score for NER."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, task, split):
    super(ModifiedEntityLevelF1Scorer, self).__init__()
    self._config = config
    self._task = task
    self._name = task.name
    self._split = split
    self._inv_label_mapping = {v: k for k, v in six.iteritems(task._get_label_mapping())}
    self._eval_examples = task.get_examples(split)
    self._word_to_char_mapping = task.get_word_to_char_mapping(split)

  def _get_results(self):
    self.write_predictions()
    self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
    all_sent_results = self._get_improved_span_labels(True, False)
    all_pred_results = self._get_improved_span_labels(False, True)
    assert len(all_sent_results) == len(all_pred_results)

    for sent_spans, span_preds in zip(all_sent_results, all_pred_results):
      self._n_correct += len(sent_spans & span_preds)
      self._n_gold += len(sent_spans)
      self._n_predicted += len(span_preds)
    utils.log(self._n_correct, self._n_predicted, self._n_gold)
    return super(ModifiedEntityLevelF1Scorer, self)._get_results()

  def _get_improved_span_labels(self, generate_labels, generate_preds):
    eid_to_idx_dict = {eid: idx for idx, eid in enumerate(self._eids)}
    all_results = []
    for example_index, example in enumerate(self._eval_examples):
      result_spans = []
      features = self._task.featurize(example, False, for_eval=True)
      for (feature_index, feature) in enumerate(features):
        idx = eid_to_idx_dict[feature[self._name + "_eid"]]
        if generate_labels:
          sentence_tags = self._labels[idx]
        if generate_preds:
          sentence_tags = self._preds[idx]
        labeled_positions = self._labeled_positions[idx]
        for (s, e, l) in tagging_utils.get_span_labels(
            sentence_tags, self._inv_label_mapping):
          start_index = labeled_positions[s]
          end_index = labeled_positions[e]
          s = s + feature[self._name + "_doc_span_orig_start"]
          e = e + feature[self._name + "_doc_span_orig_start"]
          start_index = start_index - feature[self._name + "_doc_span_start"] + 1
          end_index = end_index - feature[self._name + "_doc_span_start"] + 1
          if start_index not in feature[self._name + "_token_to_orig_map"]:
            utils.log(example.orig_id, generate_labels, "".join(example.words[s:e+1]), l, "error", "4")
            continue
          if end_index not in feature[self._name + "_token_to_orig_map"]:
            utils.log(example.orig_id, generate_labels, "".join(example.words[s:e+1]), l, "error", "5")
            continue
          if not feature[self._name + "_token_is_max_context"].get(
              start_index, False):
            utils.log(example.orig_id, generate_labels, "".join(example.words[s:e+1]), l, "error", "6")
            continue
          if end_index < start_index:
            utils.log(example.orig_id, generate_labels, "".join(example.words[s:e+1]), l, "error", "7")
            continue
          length = end_index - start_index + 1
          if length > self._config.max_answer_length:
            utils.log(example.orig_id, generate_labels, "".join(example.words[s:e+1]), l, "error", "8")
            continue
          result_spans.append((s, e, l))
      all_results.append(set(result_spans))
    return all_results

  def write_predictions(self):
    """Write final predictions to the json file."""
    all_predictions = collections.OrderedDict()
    all_pred_results = self._get_improved_span_labels(False, True)
    assert len(self._eval_examples) == len(all_pred_results)

    for example, span_preds in zip(self._eval_examples, all_pred_results):
      words = example.words
      orig_id = example.orig_id
      word_to_char_mapping = self._word_to_char_mapping[orig_id]
      answers = collections.OrderedDict()
      for s, e, l in span_preds:
        if l not in answers.keys():
          answers[l] = []
        answers[l].append(("".join(words[s:e+1]), word_to_char_mapping[s], word_to_char_mapping[e]))
      all_predictions[orig_id] = answers
    utils.write_json(dict(all_predictions),
                     self._config.ner_preds_file(self._name + "_" + self._split))
