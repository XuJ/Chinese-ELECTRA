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

"""Sequence tagging tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os
import json
import six
import tensorflow.compat.v1 as tf
from tensorflow.contrib import crf

import configure_finetuning
from finetune import feature_spec
from finetune import task
from finetune.tagging import tagging_metrics
from finetune.tagging import tagging_utils
from model import tokenization
from pretrain import pretrain_helpers
from util import utils

LABEL_ENCODING = "BIOES"


class TaggingExample(task.Example):
  """A single tagged input sequence."""

  def __init__(self, eid, task_name, words, tags, is_token_level,
               label_mapping):
    super(TaggingExample, self).__init__(task_name)
    self.eid = eid
    self.words = words
    if is_token_level:
      labels = tags
    else:
      span_labels = tagging_utils.get_span_labels(tags)
      labels = tagging_utils.get_tags(
        span_labels, len(words), LABEL_ENCODING)
    self.labels = [label_mapping[l] for l in labels]


class NerExample(task.Example):
  """A single ner input sequence"""

  def __init__(self, eid, task_name, orig_id, words, tags, text_b,
               label_mapping):
    super(NerExample, self).__init__(task_name)
    self.eid = eid
    self.orig_id = orig_id
    self.words = words
    self.labels = [label_mapping[l] for l in tags]
    self.text_b = text_b


class TaggingTask(task.Task):
  """Defines a sequence tagging task (e.g., part-of-speech tagging)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, is_token_level):
    super(TaggingTask, self).__init__(config, name)
    self._tokenizer = tokenizer
    self._label_mapping_path = os.path.join(
      self.config.preprocessed_data_dir,
      ("debug_" if self.config.debug else "") + self.name +
      "_label_mapping.pkl")
    self._is_token_level = is_token_level
    self._label_mapping = None

  def get_examples(self, split):
    sentences = self._get_labeled_sentences(split)
    examples = []
    label_mapping = self._get_label_mapping(split, sentences)
    for i, (words, tags) in enumerate(sentences):
      examples.append(TaggingExample(
        i, self.name, words, tags, self._is_token_level, label_mapping
      ))
    return examples

  def _get_label_mapping(self, provided_split=None, provided_sentences=None):
    if self._label_mapping is not None:
      return self._label_mapping
    if tf.io.gfile.exists(self._label_mapping_path):
      self._label_mapping = utils.load_pickle(self._label_mapping_path)
      return self._label_mapping
    utils.log("Writing label mapping for task", self.name)
    tag_counts = collections.Counter()
    train_tags = set()
    for split in ["train", "dev", "test"]:
      if not tf.io.gfile.exists(os.path.join(
          self.config.raw_data_dir(self.name), split + ".txt")):
        continue
      if split == provided_split:
        split_sentences = provided_sentences
      else:
        split_sentences = self._get_labeled_sentences(split)
      for _, tags in split_sentences:
        if not self._is_token_level:
          span_labels = tagging_utils.get_span_labels(tags)
          tags = tagging_utils.get_tags(span_labels, len(tags), LABEL_ENCODING)
        for tag in tags:
          tag_counts[tag] += 1
          if provided_split == "train":
            train_tags.add(tag)
    if self.name == "ccg":
      infrequent_tags = []
      for tag in tag_counts:
        if tag not in train_tags:
          infrequent_tags.append(tag)
      label_mapping = {
        label: i for i, label in enumerate(sorted(filter(
        lambda t: t not in infrequent_tags, tag_counts.keys())))
        }
      n = len(label_mapping)
      for tag in infrequent_tags:
        label_mapping[tag] = n
    else:
      labels = sorted(tag_counts.keys())
      label_mapping = {label: i for i, label in enumerate(labels)}
    utils.write_pickle(label_mapping, self._label_mapping_path)
    self._label_mapping = label_mapping
    return label_mapping

  def featurize(self, example: TaggingExample, is_training, log=False):
    words_to_tokens = tokenize_and_align(self._tokenizer, example.words)
    input_ids = []
    tagged_positions = []
    for word_tokens in words_to_tokens:
      if len(word_tokens) + len(input_ids) + 1 > self.config.max_seq_length:
        input_ids.append(self._tokenizer.vocab["[SEP]"])
        break
      if "[CLS]" not in word_tokens and "[SEP]" not in word_tokens:
        tagged_positions.append(len(input_ids))
      for token in word_tokens:
        input_ids.append(self._tokenizer.vocab[token])

    pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
    labels = pad(example.labels[:self.config.max_seq_length])
    labeled_positions = pad(tagged_positions)
    labels_mask = pad([1.0] * len(tagged_positions))
    segment_ids = pad([1] * len(input_ids))
    input_mask = pad([1] * len(input_ids))
    input_ids = pad(input_ids)
    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length
    assert len(labels) == self.config.max_seq_length
    assert len(labels_mask) == self.config.max_seq_length

    return {
      "input_ids": input_ids,
      "input_mask": input_mask,
      "segment_ids": segment_ids,
      "task_id": self.config.task_names.index(self.name),
      self.name + "_eid": example.eid,
      self.name + "_labels": labels,
      self.name + "_labels_mask": labels_mask,
      self.name + "_labeled_positions": labeled_positions
    }

  def _get_labeled_sentences(self, split):
    sentences = []
    with tf.io.gfile.GFile(os.path.join(self.config.raw_data_dir(self.name),
                                        split + ".txt"), "r") as f:
      sentence = []
      for line in f:
        line = line.strip().split()
        if not line:
          if sentence:
            words, tags = zip(*sentence)
            sentences.append((words, tags))
            sentence = []
            if self.config.debug and len(sentences) > 100:
              return sentences
          continue
        if line[0] == "-DOCSTART-":
          continue
        word, tag = line[0], line[-1]
        sentence.append((word, tag))
    return sentences

  def get_scorer(self):
    return tagging_metrics.AccuracyScorer() if self._is_token_level else \
      tagging_metrics.EntityLevelF1Scorer(self._get_label_mapping())

  def get_feature_specs(self):
    return [
      feature_spec.FeatureSpec(self.name + "_eid", []),
      feature_spec.FeatureSpec(self.name + "_labels",
                               [self.config.max_seq_length]),
      feature_spec.FeatureSpec(self.name + "_labels_mask",
                               [self.config.max_seq_length],
                               is_int_feature=False),
      feature_spec.FeatureSpec(self.name + "_labeled_positions",
                               [self.config.max_seq_length]),
    ]

  def get_prediction_module(
      self, bert_model, features, is_training, percent_done):
    n_classes = len(self._get_label_mapping())
    reprs = bert_model.get_sequence_output()
    reprs = pretrain_helpers.gather_positions(
      reprs, features[self.name + "_labeled_positions"])
    logits = tf.layers.dense(reprs, n_classes)
    losses = tf.nn.softmax_cross_entropy_with_logits(
      labels=tf.one_hot(features[self.name + "_labels"], n_classes),
      logits=logits)
    losses *= features[self.name + "_labels_mask"]
    losses = tf.reduce_sum(losses, axis=-1)
    return losses, dict(
      loss=losses,
      logits=logits,
      predictions=tf.argmax(logits, axis=-1),
      labels=features[self.name + "_labels"],
      labels_mask=features[self.name + "_labels_mask"],
      eid=features[self.name + "_eid"],
    )

  def _create_examples(self, lines, split):
    pass


def tokenize_and_align(tokenizer, words, cased=False):
  """Splits up words into subword-level tokens."""
  words = ["[CLS]"] + list(words) + ["[SEP]"]
  basic_tokenizer = tokenizer.basic_tokenizer
  tokenized_words = []
  for word in words:
    word = tokenization.convert_to_unicode(word)
    word = basic_tokenizer._clean_text(word)
    if word == "[CLS]" or word == "[SEP]":
      word_toks = [word]
    else:
      if not cased:
        word = word.lower()
        word = basic_tokenizer._run_strip_accents(word)
      word_toks = basic_tokenizer._run_split_on_punc(word)
    tokenized_word = []
    for word_tok in word_toks:
      tokenized_word += tokenizer.wordpiece_tokenizer.tokenize(word_tok)
    tokenized_words.append(tokenized_word)
  assert len(tokenized_words) == len(words)
  return tokenized_words


class Chunking(TaggingTask):
  """Text chunking."""

  def __init__(self, config, tokenizer):
    super(Chunking, self).__init__(config, "chunk", tokenizer, False)


class NerBaseTask(task.Task):
  """Ner Base Task"""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer):
    super(NerBaseTask, self).__init__(config, name)
    self._tokenizer = tokenizer
    self._label_mapping_path = os.path.join(
      self.config.preprocessed_data_dir,
      ("debug_" if self.config.debug else "") + self.name +
      "_label_mapping.pkl")
    self._label_mapping = None
    self._examples = {}
    self._word_to_char_mapping = collections.OrderedDict()

  def get_word_to_char_mapping(self, split):
    return self._word_to_char_mapping[split]

  def get_examples(self, split):
    if split in self._examples:
      return self._examples[split]

    sentences, entry_ids = self._get_labeled_sentences(split)
    examples = []
    label_mapping = self._get_label_mapping(split, sentences)
    for i, (words, tags, text_b) in enumerate(sentences):
      examples.append(NerExample(
        i, self.name, entry_ids[i], words, tags, text_b, label_mapping
      ))

    self._examples[split] = examples
    utils.log("{:} examples created".format(len(examples)))
    return examples

  def _get_labeled_sentences(self, split):
    sentences = []
    entry_ids = []
    if split not in self._word_to_char_mapping:
      self._word_to_char_mapping[split] = collections.OrderedDict()
    with tf.io.gfile.GFile(os.path.join(
        self.config.raw_data_dir(self.name),
                split + ("-debug" if self.config.debug else "") + ".json"), "r") as f:
      input_data = json.load(f)["data"]

    for entry in input_data:
      entry_ids.append(entry["id"])
      for paragraph in entry["paragraphs"]:
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        span_labels = []
        text_b_texts = []

        prev_is_whitespace = True
        prev_is_chinese = True
        for c in paragraph_text:
          if tagging_utils.is_whitespace(c):
            prev_is_whitespace = True
          else:
            if prev_is_whitespace or prev_is_chinese or tagging_utils.is_chinese_char(c):
              doc_tokens.append(c)
              prev_is_chinese = True if tagging_utils.is_chinese_char(c) else False
            else:
              doc_tokens[-1] += c
              prev_is_chinese = False
            prev_is_whitespace = False
          char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in paragraph["qas"]:
          question_text = qa["question"]
          text_b_text = tagging_utils.get_event_type(question_text)
          label_text = tagging_utils.get_question_text(question_text)
          text_b_texts.append(text_b_text)
          if split == "train" or split == "dev":
            is_impossible = qa["is_impossible"]
            if not is_impossible:
              answer = qa["answers"][0]
              answer_offset = answer["answer_start"]
              answer_length = len(answer["text"])
              start_position = char_to_word_offset[answer_offset]
              if answer_offset + answer_length - 1 >= len(char_to_word_offset):
                utils.log("End position is out of document!")
                continue
              end_position = char_to_word_offset[answer_offset + answer_length - 1]
              span_labels.append((start_position, end_position, label_text))
        assert len(set(text_b_texts)) == 1
        tags = tagging_utils.get_tags(span_labels, len(doc_tokens), LABEL_ENCODING)

        sentence = []
        for word, tag in zip(doc_tokens, tags):
          sentence.append((word, tag))

        words, tags = zip(*sentence)
        sentences.append((words, tags, text_b_texts[0]))

        self._word_to_char_mapping[split][entry["id"]] = {w: c for c, w in enumerate(char_to_word_offset)}
    assert len(sentences) == len(entry_ids)
    return sentences, entry_ids

  def _get_label_mapping(self, provided_split=None, provided_sentences=None):
    if self._label_mapping is not None:
      return self._label_mapping
    if tf.io.gfile.exists(self._label_mapping_path):
      self._label_mapping = utils.load_pickle(self._label_mapping_path)
      return self._label_mapping
    utils.log("Writing label mapping for task", self.name)
    tag_counts = collections.Counter()
    train_tags = set()
    for split in ["train", "dev", "eval"]:
      if not tf.io.gfile.exists(os.path.join(
          self.config.raw_data_dir(self.name), split + ".json")):
        continue
      if split == provided_split:
        split_sentences = provided_sentences
      else:
        split_sentences, _id = self._get_labeled_sentences(split)
      for _w, tags, _t in split_sentences:
        for tag in tags:
          tag_counts[tag] += 1
          if provided_split == "train":
            train_tags.add(tag)
    labels = sorted(tag_counts.keys())
    label_mapping = {label: i for i, label in enumerate(labels)}
    utils.write_pickle(label_mapping, self._label_mapping_path)
    self._label_mapping = label_mapping
    return label_mapping

  def featurize(self, example: NerExample, is_training, log=False, for_eval=False):
    all_features = []
    query_tokens = self._tokenizer.tokenize(example.text_b)

    if len(query_tokens) > self.config.max_query_length:
      query_tokens = query_tokens[0:self.config.max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.words):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = self._tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = self.config.max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length", "orig_start", "orig_end"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      orig_start = tok_to_orig_index[start_offset]
      orig_end = tok_to_orig_index[start_offset + length - 1]
      doc_spans.append(_DocSpan(start=start_offset, length=length, orig_start=orig_start, orig_end=orig_end))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, self.config.doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = tagging_utils._check_is_max_context(doc_spans, doc_span_index,
                                                             split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      labels = example.labels[doc_span.orig_start:doc_span.orig_end+1]
      labeled_positions = orig_to_tok_index[doc_span.orig_start:doc_span.orig_end+1]
      labels_mask = [1] * len(labeled_positions)

      # Zero-pad up to the sequence length.
      pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
      input_ids = pad(input_ids)
      input_mask = pad(input_mask)
      segment_ids = pad(segment_ids)
      labels = pad(labels)
      labels_mask = pad(labels_mask)
      labeled_positions = pad(labeled_positions)

      assert len(input_ids) == self.config.max_seq_length
      assert len(input_mask) == self.config.max_seq_length
      assert len(segment_ids) == self.config.max_seq_length
      assert len(labels) == self.config.max_seq_length
      assert len(labels_mask) == self.config.max_seq_length
      assert len(labeled_positions) == self.config.max_seq_length

      if log:
        utils.log("*** Example ***")
        utils.log("doc_span_index: %s" % doc_span_index)
        utils.log("doc_span_orig_start: %s" % doc_span.orig_start)
        utils.log("doc_span_start: %s" % doc_span.start)
        utils.log("token_to_orig_map: %s" % " ".join(
          ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        utils.log("token_is_max_context: %s" % " ".join([
                                                          "%d:%s" % (x, y) for (x, y) in
                                                          six.iteritems(token_is_max_context)
                                                          ]))
        utils.log("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        utils.log("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        utils.log("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        utils.log("labels: %s" % " ".join([str(x) for x in labels]))
        utils.log("labels_mask: %s" % " ".join([str(x) for x in labels_mask]))
        utils.log("labeled_positions: %s" % " ".join([str(x) for x in labeled_positions]))

      features = {
        "task_id": self.config.task_names.index(self.name),
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        self.name + "_eid": (1000 * example.eid) + doc_span_index,
        self.name + "_labels": labels,
        self.name + "_labels_mask": labels_mask,
        self.name + "_labeled_positions": labeled_positions
      }
      if for_eval:
        features.update({
          self.name + "_doc_span_index": doc_span_index,
          self.name + "_doc_span_orig_start": doc_span.orig_start,
          self.name + "_doc_span_start": doc_span.start,
          self.name + "_token_to_orig_map": token_to_orig_map,
          self.name + "_token_is_max_context": token_is_max_context,
        })
      all_features.append(features)
    return all_features

  def get_scorer(self, split="dev"):
    return tagging_metrics.ModifiedEntityLevelF1Scorer(self.config, self, split)

  def get_feature_specs(self):
    return [
      feature_spec.FeatureSpec(self.name + "_eid", []),
      feature_spec.FeatureSpec(self.name + "_labels",
                               [self.config.max_seq_length]),
      feature_spec.FeatureSpec(self.name + "_labels_mask",
                               [self.config.max_seq_length],
                               is_int_feature=False),
      feature_spec.FeatureSpec(self.name + "_labeled_positions",
                               [self.config.max_seq_length]),
    ]

  def get_prediction_module(
      self, bert_model, features, is_training, percent_done):
    n_classes = len(self._get_label_mapping())
    reprs = bert_model.get_sequence_output()
    reprs = pretrain_helpers.gather_positions(
      reprs, features[self.name + "_labeled_positions"])
    seq_lengths = tf.cast(tf.reduce_sum(features[self.name + "_labels_mask"], axis=1), tf.int32)
    logits = tf.layers.dense(reprs, n_classes)

    with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
      trans_val = tf.get_variable(
        "transition",
        shape=[n_classes, n_classes],
        dtype=tf.float32)
    predict_ids, _ = crf.crf_decode(logits, trans_val, seq_lengths)
    actual_ids = features[self.name + "_labels"]
    log_likelihood, _ = crf.crf_log_likelihood(
      inputs=logits,
      tag_indices=actual_ids,
      sequence_lengths=seq_lengths,
      transition_params=trans_val)
    losses = -log_likelihood

    return losses, dict(
      loss=losses,
      logits=logits,
      predictions=predict_ids,
      labels=features[self.name + "_labels"],
      labels_mask=features[self.name + "_labels_mask"],
      labeled_positions=features[self.name + "_labeled_positions"],
      eid=features[self.name + "_eid"],
    )


class Ner(NerBaseTask):
  """Chinese Ner"""

  def __init__(self, config, tokenizer):
    super(Ner, self).__init__(config, "ner", tokenizer)
