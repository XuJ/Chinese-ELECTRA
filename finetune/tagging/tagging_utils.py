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

"""Utilities for sequence tagging tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_span_labels(sentence_tags, inv_label_mapping=None):
  """Go from token-level labels to list of entities (start, end, class)."""
  if inv_label_mapping:
    sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
  span_labels = []
  last = 'O'
  start = -1
  for i, tag in enumerate(sentence_tags):
    pos, _ = (None, 'O') if tag == 'O' else tag.split('-')
    if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
      span_labels.append((start, i - 1, last.split('-')[-1]))
    if pos == 'B' or pos == 'S' or last == 'O':
      start = i
    last = tag
  if sentence_tags[-1] != 'O':
    span_labels.append((start, len(sentence_tags) - 1,
                        sentence_tags[-1].split('-')[-1]))
  return span_labels


def get_tags(span_labels, length, encoding):
  """Converts a list of entities to token-label labels based on the provided
  encoding (e.g., BIOES).
  """

  tags = ['O' for _ in range(length)]
  for s, e, t in span_labels:
    for i in range(s, e + 1):
      tags[i] = 'I-' + t
    if 'E' in encoding:
      tags[e] = 'E-' + t
    if 'B' in encoding:
      tags[s] = 'B-' + t
    if 'S' in encoding and s - e == 0:
      tags[s] = 'S-' + t
  return tags


def is_whitespace(c):
  return c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F


def is_chinese_char(cp):
  """Checks whether CP is the codepoint of a CJK character."""
  # This defines a "chinese character" as anything in the CJK Unicode block:
  #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
  #
  # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
  # despite its name. The modern Korean Hangul alphabet is a different block,
  # as is Japanese Hiragana and Katakana. Those alphabets are used to write
  # space-separated words, so they are not treated specially and handled
  # like the all of the other languages.
  cp = ord(cp)
  if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
    return True

  # 中文标点符号也视为中文字符，单独切开处理
  elif cp in [0x3002, 0xFF1F, 0xFF01, 0x3010, 0x3011, 0xFF0C, 0x3001, 0xFF1B, 0xFF1A, 0x300C,
              0x300D, 0x300E, 0x300F, 0x2019, 0x201C, 0x201D, 0x2018, 0xFF08, 0xFF09, 0x3014,
              0x3015, 0x2026, 0x2013, 0xFF0E, 0x2014, 0x300A, 0x300B, 0x3008, 0x3009]:
    return True

  return False


def get_event_type(c):
  return c.strip().split('|')[1]


def get_question_text(c):
  return c.strip().split('：')[-1]


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index
