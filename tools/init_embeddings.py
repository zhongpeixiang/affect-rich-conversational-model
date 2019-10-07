#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
To contruct embeddings with specified dimensions without pre-trained models and with affect embedding
"""
from __future__ import print_function
from __future__ import division
import six
import sys
import pickle
import numpy as np
import argparse
import torch

from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
parser.add_argument('-emb_size', required=True, type=int,
                    help="Dimension of word embeddings")
parser.add_argument('-affect_file', required=False,
                    help="Affect embeddings from this file")
parser.add_argument('-output_file', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-dict_file', required=True,
                    help="Dictionary file")
parser.add_argument('-adj_only', action="store_true",
                    help="If true, generate affect embedding for adjectives, adverbs and verbs")
parser.add_argument('-verbose', action="store_true", default=False)
opt = parser.parse_args()


def get_vocabs(dict_file):
    vocabs = torch.load(dict_file)
    enc_vocab, dec_vocab = vocabs[0][1], vocabs[-1][1]

    print("From: %s" % dict_file)
    print("\t* source vocab: %d words" % len(enc_vocab))
    print("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def match_embeddings(vocab, emb):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.stoi.items():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                print(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count

"""
My affect embedding implementation
"""
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def get_affect_embeddings(file):
    # Load dictionary of word to vad values 
    with open(file, 'rb') as f:
        return pickle.load(f)


def match_affect_embeddings(vocab, emb_size, affect_emb):
    lmtzr = WordNetLemmatizer()
    base_affect_embedding = np.array([5, 3, 5])
    affect_size = len(six.next(six.itervalues(affect_emb)))
    filtered_embeddings = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size + affect_size))
    count_affect = {"match": 0, "miss": 0}
    
    if opt.adj_only:
        with open("./data/opensub/vocab-adj.pkl", "rb") as f:
            vocab_adj = pickle.load(f)
    
    for w, w_id in vocab.stoi.items():
        w_pos = get_wordnet_pos(pos_tag([w])[0][1])
        if opt.adj_only and w not in vocab_adj:
            filtered_embeddings[w_id][emb_size:] = base_affect_embedding
            if opt.verbose:
                print(u"noun:\t{}".format(w), file=sys.stderr)
            count_affect['miss'] += 1
        else:
            w = lmtzr.lemmatize(w, pos=w_pos)
            if w in affect_emb:
                filtered_embeddings[w_id][emb_size:] = np.array(affect_emb[w])
                count_affect['match'] += 1
            else:
                filtered_embeddings[w_id][emb_size:] = base_affect_embedding
                if opt.verbose:
                    print(u"not found:\t{}".format(w), file=sys.stderr)
                count_affect['miss'] += 1            
        
    return torch.Tensor(filtered_embeddings), count_affect


def main():
    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    affect_embeddings = get_affect_embeddings(opt.affect_file)

    filtered_enc_embeddings, enc_count_affect = match_affect_embeddings(enc_vocab, opt.emb_size, affect_embeddings)
    filtered_dec_embeddings, dec_count_affect = match_affect_embeddings(dec_vocab, opt.emb_size, affect_embeddings)

    print("\nAffect Matching: ")
    match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                    for _ in [enc_count_affect, dec_count_affect]]
    print("\t* enc: %d match, %d missing, (%.2f%%)" % (enc_count_affect['match'],
                                                    enc_count_affect['miss'],
                                                    match_percent[0]))
    print("\t* dec: %d match, %d missing, (%.2f%%)" % (dec_count_affect['match'],
                                                    dec_count_affect['miss'],
                                                    match_percent[1]))

    print("\nFiltered embeddings:")
    print("\t* enc: ", filtered_enc_embeddings.size())
    print("\t* dec: ", filtered_dec_embeddings.size())

    enc_output_file = opt.output_file + ".enc.pt"
    dec_output_file = opt.output_file + ".dec.pt"
    print("\nSaving embedding as:\n\t* enc: %s\n\t* dec: %s"
          % (enc_output_file, dec_output_file))
    torch.save(filtered_enc_embeddings, enc_output_file)
    torch.save(filtered_dec_embeddings, dec_output_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
