from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from nltk import bigrams
from scipy.spatial.distance import cosine

import onmt
import onmt.io
import onmt.modules


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0, unigrams=list(), bigrams=list(), embed_greedy=0, embed_avg=0, \
        embed_extrema=0, affect_dist=0, affect_strength=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

        # Affect metrics
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.embed_greedy = embed_greedy
        self.embed_avg = embed_avg
        self.embed_extrema = embed_extrema
        self.affect_dist = affect_dist
        self.affect_strength = affect_strength

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        
    def update_affect(self, stat):
        # Affect metrics #px
        self.unigrams += stat.unigrams
        self.bigrams += stat.bigrams
        self.embed_greedy += stat.embed_greedy
        self.embed_avg += stat.embed_avg
        self.embed_extrema += stat.embed_extrema
        self.affect_dist += stat.affect_dist
        self.affect_strength += stat.affect_strength

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def distinct_1(self):
        if len(self.unigrams) == 0:
            return 0
        return len(set(self.unigrams))/len(self.unigrams)

    def distinct_2(self):
        if len(self.bigrams) == 0:
            return 0
        return len(set(self.bigrams))/len(self.bigrams)

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            affect_bias(float): added bias coefficient to output softmax during validation
            tgt_vocab: vocabulary for target datset
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, affect_bias=0, tgt_vocab=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.affect_bias = affect_bias
        self.tgt_vocab = tgt_vocab

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func, valid_iter, evaluate_every, report_evaluation):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging
            valid_iter: validation data iterator
            evaluate_every(int): interval for evaluating validation dataset during training
            report_evaluation(fn): function to print evaluation statistics

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                normalization += batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            total_stats.start_time, self.optim.lr,
                            report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

            # Evaluate on validation dataset px
            if (i+1) % evaluate_every == 0:
                valid_stats, num_val_batches = self.validate(valid_iter)
                report_evaluation(valid_stats, num_val_batches)



        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    # Calculate greedy similarity between two sentences
    def greedy_similarity(self, sent1, sent2):
        # sent1: a non-empty list of word embedding
        greedy_score = 0
        for embed_1 in sent1:
            max_score = 0
            for embed_2 in sent2:
                score = 1 - cosine(embed_1, embed_2)
                if score > max_score:
                    max_score = score
            greedy_score += max_score
        greedy_score = greedy_score/len(sent1)
        return greedy_score

    # Calculate average similarity between two sentences
    def avg_similarity(self, sent1, sent2):
        # sent1: a non-empty list of word embedding
        avg_score = 0
        matrix1 = np.asarray(sent1)
        matrix2 = np.asarray(sent2)
        # sum_1 = 0
        # sum_2 = 0
        # for embed in sent1:
        #     sum_1 += embed
        # for embed in sent2:
        #     sum_2 += embed
        # sum_1 = sum_1/len(sent1)
        # sum_2 = sum_2/len(sent2)
        avg_score = 1 - cosine(matrix1.mean(axis=0), matrix2.mean(axis=0))
        return avg_score

    # Calculate average similarity between two sentences
    def extrema_similarity(self, sent1, sent2):
        # sent1: a non-empty list of word embedding
        extrema_score = 0
        matrix1 = np.asarray(sent1)
        matrix2 = np.asarray(sent2)
        extrema_score = 1 - cosine(np.max(matrix1, axis=0), np.max(matrix2, axis=0))
        return extrema_score

    # Calculate embedding similaries for three modes: greedy matching, average, and extrema
    def embed_similarity(self, batch_tokens, target_tokens, output_embedding, target_embedding, EOS_TOKEN):
        # batch_tokens, target_tokens: (seq_len, batch_size)
        # output_embedding, target_embedding: (seq_len, batch_size, embedding_size)
        
        # Calculate greedy similarity
        greedy_score = 0
        avg_score = 0
        extrema_score = 0
        for idx in range(output_embedding.size(1)):
            # Iterate through a generated sentence
            sent1 = []
            sent2 = []
            for token_idx, token in enumerate(batch_tokens[:, idx]):
                if token == EOS_TOKEN:
                    break
                sent1.append(output_embedding[token_idx][idx].numpy())
            
            # Iterate through a target sentence
            for token_idx, token in enumerate(target_tokens[:, idx]):
                if token == EOS_TOKEN:
                    break
                sent2.append(target_embedding[token_idx][idx].numpy())
            
            if len(sent1) != 0 and len(sent2) != 0: 
                greedy_score += 0.5 * (self.greedy_similarity(sent1, sent2) + self.greedy_similarity(sent2, sent1))
                avg_score += self.avg_similarity(sent1, sent2)
                extrema_score += self.extrema_similarity(sent1, sent2)
        greedy_score = greedy_score/output_embedding.size(1)
        avg_score = avg_score/output_embedding.size(1)
        extrema_score = extrema_score/output_embedding.size(1)

        return greedy_score, avg_score, extrema_score


    def affect_distance(self, sent1, sent2):
        avg_distance = 0
        matrix1 = np.asarray(sent1)
        matrix2 = np.asarray(sent2)
        avg_distance = np.linalg.norm(matrix1.mean(axis=0) - matrix2.mean(axis=0))

        # sum_1 = 0
        # sum_2 = 0
        # for embed in sent1:
        #     sum_1 += embed
        # for embed in sent2:
        #     sum_2 += embed
        # sum_1 = sum_1/len(sent1)
        # sum_2 = sum_2/len(sent2)
        # avg_distance = np.linalg.norm(sum_1 - sum_2)
        return avg_distance


    def affect_sent_strength(self, sent):
        avg_strength = 0
        matrix = np.asarray(sent)
        base_affect_embedding = np.array([5, 3, 5])
        avg_strength = np.linalg.norm(matrix - base_affect_embedding, axis=1).mean()
        return avg_strength


    def affect_metrics(self, batch_tokens, target_tokens, output_affect_embedding, target_affect_embedding, EOS_TOKEN):
        # batch_tokens, target_tokens: (seq_len, batch_size)
        # output_affect_embedding, target_affect_embedding: (seq_len, batch_size, 3)
        
        # Calculate cosine similarity
        affect_dist = 0
        affect_strength = 0
        for idx in range(output_affect_embedding.size(1)):
            # Iterate through a generated sentence
            sent1 = []
            sent2 = []
            for token_idx, token in enumerate(batch_tokens[:, idx]):
                if token == EOS_TOKEN:
                    break
                sent1.append(output_affect_embedding[token_idx][idx].numpy())
            
            # Iterate through a target sentence
            for token_idx, token in enumerate(target_tokens[:, idx]):
                if token == EOS_TOKEN:
                    break
                sent2.append(target_affect_embedding[token_idx][idx].numpy())
            
            if len(sent1) != 0 and len(sent2) != 0: 
                affect_dist += self.affect_distance(sent1, sent2)
                affect_strength += self.affect_sent_strength(sent1)

        affect_dist = affect_dist/output_affect_embedding.size(1)
        affect_strength = affect_strength/output_affect_embedding.size(1)
        
        return affect_dist, affect_strength

    # Function to compute evaluation metrics for affect #px
    def compute_affect_metrics(self, tgt, outputs):
        """
        tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch x 1]`.
        outputs (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`)
                `[tgt_len x batch x hidden]`
        self.model.generator: a MLP of size `[hidden x vocab_size]`
        """
        scores = self.model.generator(outputs) # scores: (seq_len, batch, vocab)
        # Add bias to softmax px
        # vocab_affect = self.model.decoder.embeddings.embedding_copy.weight.data[:, -3:] # FloatTensor: (vocab, 3)
        # vocab_affect_norm = 1 + self.affect_bias * torch.norm(Variable(vocab_affect - torch.FloatTensor([5, 3, 5]).cuda()), p=2, dim=1) # (vocab, )
        # print(type(scores))
        # print(scores.size())
        # print(type(vocab_affect_norm))
        # print(vocab_affect_norm.size())
        # scores = scores * vocab_affect_norm
        # Get max token ids from softmax distribution
        output_ids = torch.max(scores, dim=2)[1] # output_ids: (seq_len, batch)
        
        # Calculate distinct-1 and distinct-2
        batch_unigrams = []
        batch_bigrams = []
        for idx in range(output_ids.size(1)):
            tokens = []
            for token in output_ids.data[:, idx]:
                if token == self.tgt_vocab.stoi[onmt.io.EOS_WORD]:
                    break
                if self.tgt_vocab.itos[token] not in string.punctuation:
                    tokens.append(token)
            bi_grams = [item for item in bigrams(tokens)]
            batch_unigrams += tokens
            batch_bigrams += bi_grams
        
        # Embedding Metrics
        output_embedding = self.model.decoder.embeddings.embedding_copy(output_ids)[:,:,:-3] # output_embedding: (seq_len, batch, emb_size)
        target_embedding = self.model.decoder.embeddings.embedding_copy(tgt.squeeze(2))[:,:,:-3] # target_embedding: (seq_len, batch, emb_size)
        embed_greedy, embed_avg, embed_extrema = self.embed_similarity(output_ids.cpu().data, tgt.squeeze(2).cpu().data, \
            output_embedding.cpu().data, target_embedding.cpu().data, self.tgt_vocab.stoi[onmt.io.EOS_WORD])

        # Affect Embedding Metrics
        affect_similarity = 0
        affect_strength = 0
        output_affect_embedding = self.model.decoder.embeddings.embedding_copy(output_ids)[:,:,-3:] # output_affect_embedding: (seq_len, batch, 3)
        target_affect_embedding = self.model.decoder.embeddings.embedding_copy(tgt.squeeze(2))[:,:,-3:] # target_affect_embedding: (seq_len, batch, 3)
        affect_dist, affect_strength = self.affect_metrics(output_ids.cpu().data, tgt.squeeze(2).cpu().data, \
            output_affect_embedding.cpu().data, target_affect_embedding.cpu().data, self.tgt_vocab.stoi[onmt.io.EOS_WORD])

        affect_stats = Statistics(unigrams=batch_unigrams, bigrams=batch_bigrams, embed_greedy=embed_greedy, embed_avg=embed_avg, \
            embed_extrema=embed_extrema, affect_dist=affect_dist, affect_strength=affect_strength)
        return affect_stats
    
    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        counter = 0
        for batch in valid_iter:
            # print("Iterating validation batch no ", counter)
            # print(valid_iter.cur_iter.iterations)
            # print(valid_iter.cur_iter._iterations_this_epoch)
 
            counter += 1
            
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Compute affect metrics # px
            batch_affect_stats = self.compute_affect_metrics(tgt[:-1], outputs) # exclude last target

            # Update statistics. px
            stats.update(batch_stats)
            stats.update_affect(batch_affect_stats)

            # Break here for faster validation px
            if valid_iter.cur_iter._iterations_this_epoch == 1:
                print("Completed {0} iterations.".format(counter))
                break

        # Set model back to training mode.
        self.model.train()

        return stats, counter

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()
        
        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()
