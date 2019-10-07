"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import onmt
import onmt.io


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
        embedding_copy(nn.Embedding): `[vocab x embed_size]`
        affect_obj_coefficient(float): coefficient for affective norm of output in loss computation
        affect_softmax_scaler(float): scaling factor before softmax computation for affect strength in the loss function
        kl_loss(boolean): use kl loss for word affect norm
    """
    def __init__(self, generator, tgt_vocab, embedding_copy, affect_obj_coefficient, affect_softmax_scaler, kl_loss, vad_loss, lambda_AC, loss_log):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]
        self.eos_idx = tgt_vocab.stoi[onmt.io.EOS_WORD]

        self.embedding_copy = embedding_copy
        self.affect_obj_coefficient = affect_obj_coefficient
        self.affect_softmax_scaler = affect_softmax_scaler
        self.kl_loss = kl_loss
        self.vad_loss = vad_loss
        self.lambda_AC = lambda_AC
        self.loss_log = loss_log
        self.base_affect_embedding = torch.FloatTensor([5, 3, 5]).cuda()
        self.vocab_affect = self.embedding_copy.weight.data[:, -3:] # (vocab, 3)
        self.vocab_affect_strength = Variable(torch.norm(self.vocab_affect - self.base_affect_embedding, p=2, dim=1)) # (vocab, )
        self.vocab_affect_strength_l1 = Variable(torch.norm(self.vocab_affect - self.base_affect_embedding, p=1, dim=1)) # (vocab, ), L1 norm
        self.vocab_affect_dist = F.softmax(self.vocab_affect_strength) # (vocab, )

        # print(self.kl_loss)

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note harding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        

        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)

            loss.div(normalization).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0, embedding_copy=None, affect_obj_coefficient=0, affect_softmax_scaler=1, kl_loss=False, vad_loss=False, lambda_AC=0, loss_log=""):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab, embedding_copy, affect_obj_coefficient, affect_softmax_scaler, kl_loss, vad_loss, lambda_AC,loss_log)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)

        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            if self.affect_obj_coefficient > 0 and self.vad_loss == True:
                print("Using weighted cross-entropy loss...")
                weight = self.vocab_affect_strength.size()[0] * (1 + self.affect_obj_coefficient * \
                    self.vocab_affect_strength)/torch.sum((1 + self.affect_obj_coefficient * self.vocab_affect_strength))
                print(weight)
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing
        # self.affect_criterion = nn.KLDivLoss(size_average=False)

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        """
        batch (batch) : batch of labeled examples
        output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`

        """
        
        seq_len = output.size(0)
        batch_size = output.size(1)
        scores = self.generator(self._bottle(output)) # after softmax
        # scores: (seq_len * batch, vocab)

        gtruth = target.view(-1) # (seq_len * batch, )
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        loss = self.criterion(scores, gtruth)

        if self.confidence < 1:
            loss_data = - likelihood.sum(0)
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        if random.random() < 0.0005:
            print(loss)

        # Maximizing affective content by Asghar 2017
        if self.lambda_AC > 0:
            target_mask = target.ne(self.padding_idx).float()
            # print(target_mask)
            output_ids = torch.max(scores, dim=1)[1].view(seq_len, batch_size) # output_ids: (seq_len, batch)
            # print(output_ids)
            affect_embedding = Variable(self.vocab_affect[output_ids.view(-1).data], requires_grad=False).view(seq_len, batch_size, 3) # (seq_len, batch, 3)
            # print(affect_embedding.requires_grad)
            # print(affect_embedding)
            affect_norm = torch.norm((affect_embedding - Variable(self.base_affect_embedding, requires_grad=False)), p=2, dim=2) # (seq_len, batch)
            # print(affect_norm)
            affect_loss = torch.sum(torch.exp(scores)[torch.LongTensor(range(seq_len * batch_size)).cuda(), output_ids.view(-1).data] * target_mask.view(-1) * affect_norm.view(-1))
            # print(torch.exp(scores)[torch.LongTensor(range(seq_len * batch_size)).cuda(), output_ids.view(-1).data])
            loss = (1-self.lambda_AC)*loss - self.lambda_AC*affect_loss
            # print(self.lambda_AC, affect_loss, loss, affect_loss/loss)
            if random.random() < 0.0005:
                print(self.lambda_AC, affect_loss, loss, affect_loss/loss)

        # Add affective loss px
        if self.affect_obj_coefficient > 0 and self.vad_loss == False:
            # output_ids = torch.max(scores, dim=1)[1].view(seq_len, batch_size) # output_ids: (seq_len, batch)
            # affect_embedding = self.embedding_copy(output_ids)[:,:,-3:] # (seq_len, batch, 3)
            
            # vocab_affect_strength = Variable(torch.norm(vocab_affect - base_affect_embedding, p=2, dim=1) * \
            #     torch.exp(scores.view(seq_len, batch_size, -1)).data, requires_grad=False) # (seq_len, batch, vocab)
            target_mask = target.ne(self.padding_idx).float() # (seq_len, batch)
            # affect_loss = (target_mask * (torch.exp(scores.view(seq_len, batch_size, -1)) * vocab_affect_strength).sum(dim=2)).sum()

            # Sequence length up to EOS for output response
            # seq_lengths = torch.zeros(batch_size).long()
            # for idx in range(batch_size):
            #     seq_lengths[idx] = seq_len
            #     for seq_idx in range(seq_len):
            #         if output_ids.data[seq_idx, idx] == self.eos_idx:
            #             seq_lengths[idx] = seq_idx
            #             break
            # seq_range = torch.arange(0, seq_len).long() #seq_range: (seq_len, )
            # seq_range_expand = seq_range.unsqueeze(1).expand(seq_len, batch_size) #seq_range_expand: (seq_len, batch_size)
            # seq_length_expand = (seq_lengths.unsqueeze(0).expand_as(seq_range_expand)) #seq_length_expand: (seq_len, batch_size)
            # output_mask = Variable((seq_range_expand < seq_length_expand).float().cuda()) #output_mask: (seq_len, batch_size) of 0 and 1

            # affect norm loss
            if self.kl_loss == False:
                affect_loss = (target_mask * (torch.exp(scores.view(seq_len, batch_size, -1)) * torch.pow(self.vocab_affect_strength, self.affect_softmax_scaler)).sum(dim=2)).sum()
                # Adaptive affect_obj_coefficient
                # ratio = loss.data[0]/affect_loss.data[0]
                # affect_loss = affect_loss * ratio * 0.1
                loss -= self.affect_obj_coefficient * affect_loss
            # Use KL divergence loss
            else:
                output_dist = scores.view(seq_len, batch_size, -1) # (seq_len, batch_size, vocab), afrer log_softmax
                
                pre_scores = self.generator[0](self._bottle(output)) # before softmax, (seq_len*batch_size, vocab)
                expanded_target_dist = self.vocab_affect_dist.repeat(seq_len*batch_size, 1)
                # target_dist = F.softmax(pre_scores * (1 + expanded_target_dist)).view(seq_len, batch_size, -1)
                target_dist = F.softmax(Variable(pre_scores.data * (1 + expanded_target_dist.data), requires_grad=False)).view(seq_len, batch_size, -1) # diable backprop
                # disable backprop for target dist
                # target_dist.requires_grad = False
                affect_loss = (target_mask * (target_dist * (torch.log(target_dist) - output_dist)).sum(dim=2)).sum()
                # affect_loss_kl = self.affect_criterion(scores, expanded_target_dist.view(seq_len * batch_size, -1))
                # affect_loss_kl = affect_loss_kl * target_mask.sum()/(seq_len*batch_size)
                loss += self.affect_obj_coefficient * affect_loss
                # loss = affect_loss # Only KL loss 

            
            if random.random() < 0.0005:
                print(self.kl_loss, affect_loss, loss, affect_loss/loss)
            if random.random() < 0.1 and self.loss_log != "":
                with open(self.loss_log, "a+") as f:
                    f.write("{0}, {1}\n".format((affect_loss/target_mask.sum().data[0]), (loss/target_mask.sum().data[0])))

        return loss, stats


def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
