# word2vec-trainer

## Introduction
A simple tool to train word vectors.
### Files
* word2vec-trainer.py
### Environment Requirements
* python 3.6
* gensim
* jieba
### How to use
1. Use raw text to train word-level embedding.
```
python word2vec-trainer.py input_file --raw_file --tokenize_level=word
```
2. Use raw text to train char-level embedding.
```
python word2vec-trainer.py input_file --raw_file --tokenize_level=char
```
3. Use raw text to train both word-level and char-level embedding.
```
python word2vec-trainer.py input_file --raw_file --tokenize_level=all
```
4. Use tokenized text to train corresponding embedding.
```
python word2vec-trainer.py input_file
```

You can use the following command to look for more detailed help:
```
python word2vec-trainer.py -h
```
The path of a `input_file` is necessary.

## Related links
* [Gensim](https://radimrehurek.com/gensim/models/word2vec.html)
* [Jieba](https://github.com/fxsjy/jieba)
* [Argparse](https://docs.python.org/3/howto/argparse.html#introducing-positional-arguments)
