#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch
import pickle

from itertools import count
import numpy as np
from nltk import bigrams

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def _report_score(name, score_total, words_total):
    score = "%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total))
    print(score)
    return score


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def rescore(pred, affect_embedding):
    output_affect_embedding = affect_embedding[pred] # output_affect_embedding: (sent_len, 3)
    matrix = np.asarray(output_affect_embedding)
    base_affect_embedding = np.array([5, 3, 5])
    affect_strength = np.linalg.norm(matrix - base_affect_embedding, axis=1).mean()
    # for w_id in pred:
    #     if w_id in emotional_word_ids:
    #         affect_strength = affect_strength*1.1
    #         break
    return affect_strength

def rerank(model, batch_data):
    affect_embedding = model.decoder.embeddings.embedding_copy.weight.data.cpu().numpy()[:,-3:] # (vocab, 3)
    for batch_idx, n_best in enumerate(batch_data["predictions"]):
        preds = []
        for sent_idx, pred in enumerate(n_best):
            score = rescore(pred, affect_embedding)
            preds.append((pred, batch_data["scores"][batch_idx][sent_idx], batch_data["attention"][batch_idx][sent_idx], score))
        # Sort predicitons in n_best list by affect strength
        sorted_preds = sorted(preds, key=lambda x:x[3], reverse=True)
        batch_data["predictions"][batch_idx] = [pred[0] for pred in sorted_preds]
        batch_data["scores"][batch_idx] = [pred[1] for pred in sorted_preds]
        batch_data["attention"][batch_idx] = [pred[2] for pred in sorted_preds]
    return batch_data

def evaluate_predictions(model, pretrained_adj, predictions, word_freq):
    total_unigrams = []
    total_bigrams = []
    total_strength = 0
    total_strength_adj = 0
    affect_embedding = model.decoder.embeddings.embedding_copy.weight.data.cpu().numpy()[:,-3:]
    affect_embedding_adj = None
    if pretrained_adj is not None:
        affect_embedding_adj = pretrained_adj.numpy()[:,-3:] # (vocab, 3)
    base_affect_embedding = np.array([5, 3, 5])
    for n_best in predictions:
        pred = n_best[0]
        
        # Calculate distinct-1 and distinct-2
        total_unigrams += pred
        total_bigrams += bigrams(pred)

        # Get word frequency as weights
        if word_freq is not None:
            a = 0.0001
            word_weights = a/(a + word_freq[pred])

        # Calculate affect strength
        output_affect_embedding = affect_embedding[pred] # output_affect_embedding: (sent_len, 3)
        matrix = np.asarray(output_affect_embedding)
        total_strength += np.linalg.norm(matrix - base_affect_embedding, axis=1).mean()

        if word_freq is not None:
            total_strength += (word_weights * np.linalg.norm(matrix - base_affect_embedding, axis=1)).mean()

        # Calculate affect strength adj
        if affect_embedding_adj is not None:
            output_affect_embedding_adj = affect_embedding_adj[pred] # output_affect_embedding: (sent_len, 3)
            matrix_adj = np.asarray(output_affect_embedding_adj)
            total_strength_adj += np.linalg.norm(matrix_adj - base_affect_embedding, axis=1).mean()
        
    distinct_1 = len(set(total_unigrams))/len(total_unigrams)
    distinct_2 = len(set(total_bigrams))/len(total_bigrams)
    avg_strength = total_strength/len(predictions)
    avg_strength_adj = total_strength_adj/len(predictions)
    print("Distinct-1: {0}, distinct-2: {1}, affect strength: {2}, affect strength adj: {3}".format(distinct_1, distinct_2, avg_strength, avg_strength_adj))
    return "Distinct-1: {0}, distinct-2: {1}, affect strength: {2}, affect strength adj: {3}".format(distinct_1, distinct_2, avg_strength, avg_strength_adj)

def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.Translator(model, fields,
                                           beam_size=opt.beam_size,
                                           n_best=opt.n_best,
                                           global_scorer=scorer,
                                           max_length=opt.max_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=opt.cuda,
                                           beam_trace=opt.dump_beam != "",
                                           dump_beam=opt.dump_beam,
                                           min_length=opt.min_length,
                                           antilm_lambda=opt.antilm_lambda,
                                           antilm_eta=opt.antilm_eta,
                                           antilm_equal_src=opt.antilm_equal_src,
                                           lambda_ADBS=opt.lambda_ADBS,
                                           affective_decoding=opt.affective_decoding,
                                           k=opt.k,
                                           sort_AS=opt.sort_AS,
                                           sort_similarity=opt.sort_similarity,
                                           penalize_repeats=opt.penalize_repeats)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    predictions = []

    # Test word embedding
    # print(model.decoder.embeddings.word_lut.weight.data[200:220, -10:])
    # print(model.decoder.embeddings.embedding_copy.weight.data[200:220, -10:])
    # Load adj vocab
    pretrained_adj = None
    if opt.adj_vocab:
        print("Loading adj vocab...")
        pretrained_adj = torch.load(opt.adj_vocab)

    word_freq = None
    if opt.weighted_AS:
        print("Loading unigram frequency...")
        with open(opt.weighted_AS, "rb") as f:
            word_freq = np.array(pickle.load(f))

    # Load word embedding matrix and VAD embedding matrix, pass them to translate_batch()
    if opt.save_attn:
        pred_ids = []
        attns = []
        indices = []

    for batch in data_iter:
        batch_data = translator.translate_batch(batch, data)

        if opt.save_attn:
            pred_ids.append(batch_data["predictions"])
            attns.append(batch_data["attention"])
            indices.append(batch_data["batch"].indices)

        # Rerank beams
        if opt.rerank:
            batch_data = rerank(model, batch_data)
        predictions += batch_data["predictions"]
        translations = builder.from_batch(batch_data)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)

            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]
            if opt.display_1:
                n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:1]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()
            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))
    
    if opt.save_attn:
        with open(opt.save_attn + ".pkl", "wb") as f:
            pickle.dump(attns, f)
        with open(opt.save_attn + "_predictions" + ".pkl", "wb") as f:
            pickle.dump(pred_ids, f)
        with open(opt.save_attn + "_indices" + ".pkl", "wb") as f:
            pickle.dump(indices, f)

    pred_score = _report_score('PRED', pred_score_total, pred_words_total)
    out_file.write(pred_score + "\n")
    out_file.flush()
    
    # Evaluate predictions here
    metrics = evaluate_predictions(model, pretrained_adj, predictions, word_freq)
    out_file.write(metrics + "\n")
    out_file.flush()

    if opt.tgt:
        gold_score = _report_score('GOLD', gold_score_total, gold_words_total)
        if opt.report_bleu:
            _report_bleu()
        if opt.report_rouge:
            _report_rouge()
        out_file.write(gold_score + "\n")
    out_file.flush()

    if opt.dump_beam:
        import json
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
