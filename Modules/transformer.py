import torch.nn as nn
from Layers.transformer import EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    ''' A encoder models with self attention mechanism. '''

    # def __init__(
    #         self, embedding,
    #         len_max_seq, d_word_vec,
    #         n_layers, n_head, d_k, d_v,
    #         d_model, d_inner, dropout=0.1):
    #     super().__init__()
    #
    #
    #     self.word_embedding = nn.Embedding.from_pretrained(embedding)
    #
    #     self.position_enc = nn.Embedding.from_pretrained(
    #         get_sinusoid_encoding_table(len_max_seq, d_word_vec, padding_idx=0),
    #         freeze=True)

    def __init__(
            self, position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout=0.1, use_bottleneck=True, d_bottleneck=256):
        super().__init__()


        self.position_enc = position_encoding_layer(d_features=d_features, max_length=max_seq_length, d_meta=d_meta)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_features, n_head, dropout, use_bottleneck=use_bottleneck, d_bottleneck=d_bottleneck)
            for _ in range(n_layers)])

    def forward(self, feature_sequence, position, non_pad_mask=None, slf_attn_mask=None):
        '''
            Arguments:
                input_feature_sequence {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence
                position {Tensor, shape [batch, max_sequence_length, d_meta]} -- input feature position sequence

            Returns:
                enc_output {Tensor, shape [batch, max_sequence_length, d_features]} -- encoder output (representation)
                encoder_self_attn_list
        '''

        encoder_self_attn_list = []
        # # -- Prepare masks
        # slf_attn_mask = get_attn_key_pad_mask(seq_k=sequence, seq_q=sequence,
        #                                       padding_idx=0)  # [batch_size, seq_length, seq_length]
        #
        # non_pad_mask = get_non_pad_mask(sequence, padding_idx=0)  # [batch_size, seq_length, 1]

        # Add position information at the beginning
        enc_output = feature_sequence + self.position_enc(position)

        for enc_layer in self.layer_stack:
            enc_output, encoder_self_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            encoder_self_attn_list += [encoder_self_attn]

        return enc_output, encoder_self_attn_list


class Decoder(nn.Module):
    ''' A decoder models with self attention mechanism. '''

    # def __init__(
    #         self,
    #         n_tgt_vocab, len_max_seq, d_word_vec,
    #         n_layers, n_head, d_k, d_v,
    #         d_model, d_inner, dropout=0.1):
    #     super().__init__()
    #
    #     n_position = len_max_seq + 1
    #
    #     self.tgt_word_emb = nn.Embedding(
    #         n_tgt_vocab, d_word_vec, padding_idx=0)
    #
    #     self.position_enc = nn.Embedding.from_pretrained(
    #         get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
    #         freeze=True)

    def __init__(
            self, position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout=0.1, use_bottleneck=True, d_bottleneck=256):

        super().__init__()

        self.position_enc = position_encoding_layer(d_features=d_features, max_length=max_seq_length, d_meta=d_meta)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_features, n_head, dropout, use_bottleneck=use_bottleneck, d_bottleneck=d_bottleneck)
            for _ in range(n_layers)])

    def forward(self, target_feature_sequence, tg_position, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
            Arguments:
                target_feature_sequence {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence
                tg_position {Tensor, shape [batch, max_sequence_length, d_meta]} -- input feature position sequence
                enc_output {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence

            Returns:
                dec_output {Tensor, shape [batch, max_sequence_length, d_features]} -- decoder output (representation)
                dec_slf_attn_list
                dec_enc_attn_list
        '''

        dec_slf_attn_list, dec_enc_attn_list = [], []
        #
        # # -- Prepare masks
        # non_pad_mask = get_non_pad_mask(tgt_seq)
        #
        # slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        # slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        # slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        #
        # dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        dec_output = target_feature_sequence + self.position_enc(tg_position)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            dec_slf_attn_list += [dec_slf_attn]
            dec_enc_attn_list += [dec_enc_attn]

        return dec_output, dec_slf_attn_list, dec_enc_attn_list


class Transformer(nn.Module):
    ''' A sequence to sequence models with attention mechanism. '''

    # def __init__(
    #         self,
    #         embedding, n_tgt_vocab, len_max_seq,
    #         d_word_vec=512, d_model=512, d_inner=2048,
    #         n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
    #         tgt_emb_prj_weight_sharing=True,
    #         emb_src_tgt_weight_sharing=True):
    def __init__(
            self, position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout=0.1, use_bottleneck=True,
            d_bottleneck=256):

        super().__init__()

        self.encoder = Encoder(position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout, use_bottleneck, d_bottleneck)

        self.decoder = Decoder(position_encoding_layer, n_layers, n_head, d_features, max_seq_length, d_meta, dropout, use_bottleneck, d_bottleneck)


        # if tgt_emb_prj_weight_sharing:
        #     # Share the weight matrix between target word embedding & the final logit dense layer
        #     self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
        #     self.x_logit_scale = (d_model ** -0.5)
        # else:
        #     self.x_logit_scale = 1.

        # if emb_src_tgt_weight_sharing:
        #     # Share the weight matrix between source & target word embeddings
        #     # assert n_src_vocab == n_tgt_vocab, \
        #     "To share word embedding table, the vocabulary size of src/tgt shall be the same."
        #     self.encoder.word_embedding.weight = self.decoder.tgt_word_emb.weight

    def forward(self, input_feature_sequence, in_position, target_feature_sequence, tg_position):
        '''
            Arguments:
                input_feature_sequence {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence
                in_position {Tensor, shape [batch, max_sequence_length, d_meta]} -- input feature position sequence

            Returns:
                dec_output {Tensor, shape [batch, ]} -- decoder output (representation)
        '''

        # target_feature_sequence, tg_position = target_feature_sequence[:, :-1], tg_position[:, :-1]

        enc_output, encoder_self_attn_list = self.encoder(input_feature_sequence, in_position, non_pad_mask=None, slf_attn_mask=None)
        dec_output, dec_slf_attn_list, dec_enc_attn_list = self.decoder(target_feature_sequence, tg_position, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None)

        return dec_output


# class TransformerCls(nn.Module):
#     '''Transformer module for classification'''
#
#     def __init__(
#             self,
#             embedding, output_num, len_max_seq,
#             d_word_vec=100, d_model=512, d_inner=2048,
#             n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):
#         super().__init__()
#
#         self.encoder = Encoder(position_encoding_layer, n_layers, n_head, d_features, d_meta, dropout, use_bottleneck, d_bottleneck)
#
#
#         # encoder output: [batch_size, seq_length, w2v_length]
#
#         self.tgt_word_prj = nn.Linear(len_max_seq * d_word_vec, output_num, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)
#
#     def forward(self, src_seq, src_pos):
#         '''
#             Arguments:
#                 src_seq {Tensor, shape [batch, len_max_seq]} -- word index sequence with padding
#                 src_pos {Tensor, shape [batch, len_max_seq]} -- position index sequence with padding
#
#             Returns:
#                 seq_logit {Tensor, shape [batch, output_num]} -- logits
#         '''
#         enc_output, *_ = self.encoder(src_seq, src_pos)
#         seq_logit = self.tgt_word_prj(enc_output.view([enc_output.size(0), -1]))
#
#         return seq_logit
