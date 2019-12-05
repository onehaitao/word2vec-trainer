#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


from gensim.models import word2vec
import argparse
import os
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def train(args):
    input = args.input
    output = args.output
    size = args.size
    workers = args.workers
    min_count = args.min_count
    sg = args.sg
    if not os.path.isfile(input):
        raise FileNotFoundError
    sentences = word2vec.Text8Corpus(input)
    model = word2vec.Word2Vec(
        sentences,
        sg=sg,
        size=size,
        window=5,
        min_count=min_count,
        negative=3,
        sample=1e-3,
        workers=workers
    )
    model.save(output + str(size) + 'd.model')
    model.wv.save_word2vec_format(output + str(size) + 'd.txt', binary=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'a simple tool to train word vectors'
    parser.add_argument('input', type=str,
                        help='path of input file, processed by word segmentation')
    parser.add_argument('-o', '--output', type=str, default='./word2vec_',
                        help='path of output file, models and txt file')
    parser.add_argument('-s', '--size', type=int, default='50',
                        help='dimensionality of the word vectors')
    parser.add_argument('-w', '--workers', type=int, default='1',
                        help='number of threads, valid in cpython')
    parser.add_argument('-m', '--min_count', type=int, default='5',
                        help='ignores all words with total frequency lower than this')
    parser.add_argument('-sg', '--sg', type=int, default='1', choices=[0, 1],
                        help='training algorithm: 1 for skip-gram; otherwise CBOW')
    args = parser.parse_args()

    train(args)
