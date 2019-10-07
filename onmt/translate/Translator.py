import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn

import onmt.translate.Beam
import onmt.io


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, dump_beam="", min_length=0, 
                 antilm_lambda=0, antilm_eta=0, antilm_equal_src=False, lambda_ADBS=0,
                 affective_decoding=False, k=10, sort_AS=False, sort_similarity=False, penalize_repeats=False):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.beam_trace = beam_trace
        self.dump_beam = dump_beam
        self.min_length = min_length
        self.antilm_lambda = antilm_lambda
        self.antilm_eta = antilm_eta
        self.antilm_equal_src = antilm_equal_src
        self.lambda_ADBS = lambda_ADBS
        self.affective_decoding = affective_decoding
        self.word_vector = self.model.decoder.embeddings.embedding_copy.weight.data[:, :-3]
        self.base_vad = torch.FloatTensor([5, 3, 5]).cuda()
        self.vocab_affect = self.model.decoder.embeddings.embedding_copy.weight.data[:, -3:]
        self.vocab_size = self.model.decoder.embeddings.embedding_copy.weight.data.size(0)
        self.k = k
        self.sort_AS = sort_AS
        self.sort_similarity = sort_similarity
        self.cos = nn.CosineSimilarity(dim=0)
        self.penalize_repeats = penalize_repeats

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    vocab_affect=self.vocab_affect,
                                    lambda_ADBS=self.lambda_ADBS)
                for __ in range(batch_size)] # A batch of beams

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Define the similarity between two words
        def similarity_score(word1, word2):
            wv1, wv2 = self.word_vector[word1], self.word_vector[word2]
            vad1, vad2 = self.vocab_affect[word1] - self.base_vad, self.vocab_affect[word2] - self.base_vad
            return (1 + self.cos(wv1, wv2)) * (1 + self.cos(vad1, vad2))/4

        # AS loss for a word
        def AS_loss(word):
            return self.vocab_affect[word].norm()

        # check repeats during decoding
        def is_repeated(word, beam_id, batch_id):
            b = beam[batch_id]
            for step in range(len(b.next_ys)):
                if b.next_ys[step][beam_id] == word:
                    return 1
            return 0

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type) # (seq_len, batch_size, 1)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src # src_lengths: (batch_size, )
        
        # print(vocab.stoi[onmt.io.PAD_WORD], vocab.stoi[onmt.io.EOS_WORD], vocab.stoi[onmt.io.BOS_WORD]) # 1,3,2
        if self.antilm_lambda > 0:
            dummy_src = src.clone()
            dummy_src.data.fill_(vocab.stoi[onmt.io.PAD_WORD])
            dummy_src_lengths = src_lengths.clone()

            if self.antilm_equal_src is False:
                for idx, src_len in enumerate(dummy_src_lengths):
                    dummy_src[src_len-1, idx, :] = vocab.stoi[onmt.io.EOS_WORD]
            else:
                dummy_src_lengths[:] = torch.max(src_lengths)
                dummy_src[-1, :, :] = vocab.stoi[onmt.io.EOS_WORD]

            # run encoder and init decoder for dummy src
            dummy_enc_states, dummy_context = self.model.encoder(dummy_src, dummy_src_lengths)
            dummy_dec_states = self.model.decoder.init_decoder_state(
                                            dummy_src, dummy_context, dummy_enc_states)

        # normal seq2seq
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
                                        src, context, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data)\
                                                  .long()\
                                                  .fill_(context.size(0))

        
        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        context = rvar(context.data)
        context_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # Repeat for dummy src, not implemented when copy_attn = True
        if self.antilm_lambda > 0:
            dummy_context = rvar(dummy_context.data)
            dummy_context_lengths = dummy_src_lengths.repeat(beam_size)
            dummy_dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        # Add MMI-antilm decoding, for each token, get the beam score by p(Tk|S) - p(Tk),
        # where k is the index of current token in candidate response px
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, context, dec_states, context_lengths=context_lengths, src=src)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # Calculate p(T) for MMI-antilm model 
            if self.antilm_lambda > 0:
                dummy_dec_out, dummy_dec_states, dummy_attn = self.model.decoder(
                    inp, dummy_context, dummy_dec_states, context_lengths=dummy_context_lengths, src=dummy_src)
                dummy_dec_out = dummy_dec_out.squeeze(0)

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                # Add probability bias towards affective words

                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # out: (beam, batch_size, tgt_vocab)
                if self.antilm_lambda > 0:
                    dummy_out = self.model.generator.forward(dummy_dec_out).data
                    dummy_out = unbottle(dummy_out)
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()

            # true probability = p(Tk|S) - p(Tk) px
            if self.antilm_lambda > 0:
                # Weighted target probability, only the first k tokens need to be counted
                if i < self.antilm_eta:
                    out = out - self.antilm_lambda * dummy_out
            
            # Affective Decoding
            if self.affective_decoding:
                # 1. Select the most likely word
                top_words = torch.max(out, dim=2)[1] #(beam, batch)

                # 2. Select the top k likely words
                top_k_words = torch.topk(out, self.k, dim=2)[1] #(beam, batch, k)

                # 3. Add loss offset, AS and similarity loss
                for beam_id in range(top_k_words.size(0)):
                    for batch_id in range(top_k_words.size(1)):
                        for word in top_k_words[beam_id][batch_id]:
                            # 3.1 loss offset for top k words
                            out[beam_id][batch_id][word] += 100

                            # 3.2 AS for each top k word
                            if self.sort_AS:
                                out[beam_id][batch_id][word] += AS_loss(word)

                            # 3.3 similarity loss for each top word
                            if self.sort_similarity:
                                out[beam_id][batch_id][word] += similarity_score(top_words[beam_id][batch_id], word)[0]

                            # 3.4 loss for repeated words
                            if self.penalize_repeats:
                                out[beam_id][batch_id][word] -= 100 * is_repeated(word, beam_id, batch_id)

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j],
                    unbottle(attn["std"]).data[:, j, :context_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)
        
        # Save beam
        if self.beam_trace:
            with open(self.dump_beam + ".pkl", "wb") as f:
                pickle.dump(beam[0], f, pickle.HIGHEST_PROTOCOL)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        # print(ret["scores"])
        # print(ret["predictions"])
        # for i in range(len(ret["scores"])):
        #     for j in range(len(ret["scores"][i])):
        #         ret["scores"][i][j] -= (1 + ret["scores"][i][j]//100)*100                
        
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data, src=src)
        ret["batch"] = batch
        ret["src"] = src.view(src.size(1), src.size(0))

        # print(ret["scores"])
        # print(ret["predictions"])

        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data, src=None):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(src,
                                                           context, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, context, dec_states, context_lengths=src_lengths, src=src)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
