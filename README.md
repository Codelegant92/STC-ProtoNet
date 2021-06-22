# STC-ProtoNet
## A PyTorch Implementation of Prototypical Networks for Few-shot Spoken Term Classification with Varying Classes and Examples

This repository presents an extended-ProtoNet approach to address the user-defined spoken term classification task.

Our implementation is based on a PyTorch implementation of an integrated testbed of few-shot classification https://github.com/wyharveychen/CloserLookFewShot.

## Prerequisites
+ python: 3.x
+ PyTorch: 1.0+
+ librosa: 0.8

## Dataset - Google Speech Commands dataset v2
1. We use the raw data from the dataset which contains 35 keywords: 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'cat', 'tree', 'house', 'bird', 'visual', 'backward', 'follow', 'forward', 'learn', 'sheila', 'bed', 'dog', 'happy', 'marvin', 'wow'.
2. We choose 20 keywords to form normal source data: 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'cat', 'tree', 'house', 'bird', 'visual', 'backward', 'follow', 'forward', 'learn', 'sheila'; 10 keywords to form target data: 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'; 5 keywords to form the unknown class: 'bed', 'dog', 'happy', 'marvin', 'wow'. 
3. We split the dataset into different parts to satisfy our experimental setting:
```shell
data/
├──speech_commands/
    ├──yes
        ├──00f0204f_nohash_0.wav
        ├──d962e5ac_nohash_1.wav
        ...
    ...
    ├──unknown
        ├──happy
           ├──299c14b1_nohash_2.wav
           ...
filelists/
├──base.json
├──base_unk.json
├──base_sil.json
├──val.json
├──val_unk.json
├──val_sil.json
├──novel.json
├──novel_unk.json
├──novel_sil.json

```
## Train and test
1. Run `python craft_MMCenters.py` to generate the hard points.
2. Run `python train.py` followed by a series of arguments:
```shell
--dataset
--model
--train_n_way
--test_n_way
--train_n_shot
--test_n_shot
--fixed_way
--train_max_way
--train_min_way
--test_max_way
--test_min_way
--max_shot
--min_shot
...
```
3. Run `python save_features.py` to generate embeddings of the testing examples of a training method by choosing the arguments.
4. Run `python test.py` to to evaluation.
