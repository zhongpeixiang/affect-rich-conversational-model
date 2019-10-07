import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import random

from onmt.modules.UtilClass import BottleLinear
from onmt.Utils import aeq, sequence_mask


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """
    def __init__(self, dim, coverage=False, attn_type="dot", affective_attention=None, affective_attn_strength=0.1, embedding_size=1027, local_weights=False):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        self.affective_attention = affective_attention
        self.affective_attn_strength = affective_attn_strength
        self.embedding_size = embedding_size
        self.local_weights = local_weights # weighted affective attention, local weights
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = BottleLinear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

        # Add affective attention params
        if self.affective_attention == "matrix_norm":
            self.affect_linear = nn.Linear(dim, 3, bias=False)
        elif self.affective_attention == "bigram_norm":
            self.affect_linear = nn.Linear(embedding_size-3, 3, bias=False)
            self.affect_linear1 = nn.Linear(3, 1, bias=False)

    def score(self, h_t, h_s, emb_s=None, emb_copy=None, word_freq=None):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
          emb_s (`FloatTensor`): sequence of sources `[batch x src_len x emb_dim]`, fixed embedding params

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        # Use original copy of embedding to compute affective attention
        # emb_s = emb_copy

        # Add affect norm to scores px
        affect_norm = 1
        base_affect_embedding = Variable(torch.FloatTensor([5, 3, 5]).cuda())

        # For translation with beam search, check batch dimension
        if emb_s is not None and tgt_batch != emb_s.size(0):
            beam_size = int(tgt_batch/emb_s.size(0))
            emb_s = emb_s.repeat(beam_size, 1, 1)
            emb_copy = emb_copy.repeat(beam_size, 1, 1)
            if word_freq is not None:
                word_freq = word_freq.repeat(beam_size, 1)
        if self.affective_attention == "norm":
            affect_norm = self.affective_attn_strength * torch.norm(emb_copy[:,:,-3:] - base_affect_embedding, p=2, dim=2).unsqueeze(1).repeat(1, tgt_len, 1)
        elif self.affective_attention == "matrix_norm":
            # Apply non-linear transformation to decoder hidden state and then multiply it with affect embedding
            transformed_affect = self.tanh(self.affect_linear(h_t)) # `[batch x tgt_len x 3]`
            affect_norm = torch.bmm(transformed_affect, torch.abs(emb_copy[:,:,-3:] - base_affect_embedding).transpose(1, 2)) 
            # (batch, t_len, 3) x (batch, 3, s_len) --> (batch, t_len, s_len)
        elif self.affective_attention == "bigram_norm":
            # Apply non-linear transformation to previous input token embedding and then multiply it with affect embedding
            # Use current embedding instead because the emb_copy is randomly initialized and does not contain useful word vectors
            # Use original embedding when emb_copy is initialized with GloVe embeddings
            # New bigram: x' = x + alpha * (x - [5,1,5])
            emb_s_copy = emb_s.clone()
            emb_s_copy[:,1:,:] = emb_s[:,:-1,:] # Shift embeddings to the right by one position, now position t refers to t-1
            emb_s_copy[:,0,:] = emb_s[:,0,:]
            transformed_affect = self.tanh(self.affect_linear(emb_s_copy[:,:,:-3])) # `[batch x src_len x 3]`
            
            l2_vad = torch.pow(((1 + transformed_affect) * (emb_copy[:,:,-3:] - base_affect_embedding)), 2).sum(dim=2)
            if word_freq is not None:
                word_freq = word_freq.unsqueeze(2).repeat(1, 1, 3) #(batch, src_len, 3)
                if self.local_weights:
                    sum_log = torch.log(1/word_freq).sum(dim=1, keepdim=True)
                    word_weights = Variable(torch.log(1/word_freq)/sum_log, requires_grad=False)
                else:
                    a = 0.001
                    word_weights = Variable(a/(a + word_freq), requires_grad=False)
                l2_vad = (word_weights * torch.pow(((1 + transformed_affect) * (emb_copy[:,:,-3:] - base_affect_embedding)), 2)).sum(dim=2)
                # print(transformed_affect.size(), word_weights.size(), l2_vad.size(), type(transformed_affect), type(word_weights), type(l2_vad))
            affect_norm = self.affective_attn_strength * l2_vad.unsqueeze(1).repeat(1, tgt_len, 1)
            # affect_norm = self.affective_attn_strength * torch.norm(((1 + transformed_affect) * (emb_copy[:,:,-3:] - base_affect_embedding)), p=1, dim=2).unsqueeze(1).repeat(1, tgt_len, 1)

            # affect_norm = self.affect_linear1(torch.pow((1 + transformed_affect) * (emb_copy[:,:,-3:] - base_affect_embedding), 2)).squeeze(2).unsqueeze(1).repeat(1, tgt_len, 1)
            
            # affect_norm = (1 + F.softmax(torch.norm((1 + transformed_affect) * (emb_copy[:,:,-3:] - base_affect_embedding), p=1, dim=2))).unsqueeze(1).repeat(1, tgt_len, 1)

            
        # Make batch_size consistent during inference due to beam_size multiplier
        if affect_norm is not 0 and tgt_batch != emb_s.size(0):
            beam_size = int(tgt_batch/emb_s.size(0))
            affect_norm = affect_norm.repeat(beam_size, 1, 1)
        
        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            # print(torch.bmm(h_t, h_s_)[0])
            # print(affect_norm[0])
            # if random.random() < 0.05:
            #     print(self.affective_attn_strength, l2_vad.unsqueeze(1).repeat(1, tgt_len, 1)[0], torch.bmm(h_t, h_s_)[0])

            return torch.bmm(h_t, h_s_) + affect_norm
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            # Add affect norm to scores
            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len) + affect_norm

    def forward(self, input, context, context_lengths=None, coverage=None, embedding_now=None, embedding_copy=None, word_freq=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`, decoder hidden state at each timestep
          context (`FloatTensor`): source vectors `[batch x src_len x dim]`, encoder hidden state at each timestep
          context_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          embedding_copy (`FloatTensor`): the original input sequence embeddings with affect `[batch x src_len x emb_dim]`
          word_freq (`FloatTensor`): the word frequency `[batch x src_len]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input for InputFeedDecoder
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            context += self.linear_cover(cover).view_as(context)
            context = self.tanh(context)

        # compute attention scores, as in Luong et al.
        # Add affective attention here px
        align = self.score(input, context, embedding_now, embedding_copy, word_freq)

        if context_lengths is not None:
            mask = sequence_mask(context_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        return attn_h, align_vectors
