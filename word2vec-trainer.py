#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import time
import argparse
import jieba
import os
import warnings
import multiprocessing
from gensim.models import word2vec
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def as_time(seconds_gap):
    total_minutes, seconds = divmod(int(seconds_gap), 60)
    hours, minutes = divmod(total_minutes, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


def tokenize_by_word(path_src, path_des, workers=1):
    start = time.time()
    jieba.enable_parallel(workers)
    with open(path_src, 'rb') as fr:
        content = fr.read()
    words = ' '.join(jieba.cut(content))
    path_des = os.path.join(path_des, 'text_tokenized_by_word.txt')
    with open(path_des, 'wb') as fw:
        fw.write(words.encode('utf-8'))
    end = time.time()
    print('word-tokenization costs {}'.format(as_time(end-start)))
    return path_des


def tokenize_by_char(path_src, path_des):
    start = time.time()
    path_des = os.path.join(path_des, 'text_tokenized_by_char.txt')
    fw = open(path_des, 'w', encoding='utf-8')
    with open(path_src, 'r', encoding='utf-8') as fr:
        for line in fr:
            tokens = list(line.strip())
            fw.write('{}\n'.format(' '.join(tokens)))
    fw.close()
    end = time.time()
    print('char-tokenization costs {}'.format(as_time(end-start)))
    return path_des


def train_embedding(input_file, args, token_level=None):
    start = time.time()
    sentences = word2vec.Text8Corpus(input_file)
    model = word2vec.Word2Vec(
        sentences,
        size=args.size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        hs=args.hs,
        negative=args.negative,
        seed=args.seed,
        sample=args.sample,
    )
    if token_level is not None:
        filepath = os.path.join(args.output_dir, '{}_{}_{}d'.format(args.filename, token_level, args.size))
    else:
        filepath = os.path.join(args.output_dir, '{}_{}d'.format(args.filename, args.size))
    model.save(filepath + '.model')
    model.wv.save_word2vec_format(filepath + '.txt', binary=False)
    end = time.time()
    if token_level is None:
        info_str = 'train embedding costs {}'.format(as_time(end-start))
    else:
        info_str = 'train {}-level embedding costs {}'.format(token_level, as_time(end-start))
    print(info_str)


def run(args, token_level):
    input_file = args.input_file
    if token_level == 'word':
        input_file = tokenize_by_word(args.input_file, args.output_dir, workers=args.workers)
    elif token_level == 'char':
        input_file = tokenize_by_char(args.input_file, args.output_dir)

    train_embedding(input_file, args, token_level=token_level)


def train(args):
    if not os.path.isfile(args.input_file):
        raise FileNotFoundError
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.raw_file:
        if args.tokenize_level == 'word':
            run(args, 'word')
        elif args.tokenize_level == 'char':
            run(args, 'char')
        else:
            p1 = multiprocessing.Process(target=run, args=(args, 'word'))
            p2 = multiprocessing.Process(target=run, args=(args, 'char'))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
    else:
        run(args, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'a simple tool to train word vectors'

    parser.add_argument('input_file', type=str,
                        help='path of input file')

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='filename of output dir')
    parser.add_argument('--filename', type=str, default='word2vec',
                        help='name of target files')
    parser.add_argument('--raw_file', action='store_true', default=False,
                        help='if False, text need to word/char tokenization')
    parser.add_argument('--tokenize_level', type=str, default='word',
                        choices=['char', 'word', 'all'],
                        help='tokenize level if need to tokenize')

    parser.add_argument('--size', type=int, default=50,
                        help='dimensionality of the word vectors')
    parser.add_argument('--window', type=int, default=5,
                        help='maximum distance between the current and predicted word within a sentence')
    parser.add_argument('--min_count', type=int, default=5,
                        help='ignores all words with total frequency lower than this')
    parser.add_argument('--workers', type=int, default='1',
                        help='number of threads, valid in cpython')
    parser.add_argument('--sg', type=int, default=0,
                        choices=[0, 1],
                        help='training algorithm: 1 for skip-gram; otherwise CBOW')
    parser.add_argument('--hs', type=int, default=0,
                        choices=[0, 1],
                        help='if 1, hierarchical softmax will be used for model training. \
                              if 0, and negative is non-zero, negative sampling will be used.')
    parser.add_argument('--negative', type=int, default=5,
                        help='if > 0, negative sampling will be used. \
                              if set to 0, no negative sampling is used.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for the random number generator.')
    parser.add_argument('--sample', type=float, default=1e-3,
                        help='threshold for configuring which higher-frequency words are \
                              randomly downsampled, useful range is (0, 1e-5)')
    args = parser.parse_args()

    print(args)
    train(args)
