from torch.nn import Transformer, TransformerEncoder, TransformerDecoder
import torch.nn as nn
import torch.nn.init as init
import torch
import math
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_sequence_length):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Create a positional encoding matrix
        pe = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        print(pe.size())
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, max_target_length,batch_size,mask):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,batch_first = True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first = True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.batch_size= batch_size
        self.max_target_length= max_target_length
        self.nhead = nhead
        self.attn_mask = mask

        init.xavier_uniform_(self.linear1.weight)
        init.zeros_(self.linear1.bias)

        init.xavier_uniform_(self.linear2.weight)
        init.zeros_(self.linear2.bias)

    def forward(self, x, enc_output):

        # Multihead self-attention with future-position masking
        attn_output, _ = self.self_attn(x, x, x, attn_mask=self.attn_mask)
        #print("decoder attention :" + str(attn_output.size()))
        #print(attn_output)


        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Multihead cross-attention with encoder output
        cross_attn_output, _ = self.multihead_attn(x, enc_output, enc_output)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)



        # Feed-forward layer
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + ff_output

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        # Multihead self-attention
        attn_output, _ = self.self_attn(x, x, x)
        #print( "encoder attention :" + str(attn_output.size()))
        #print(attn_output)

        # Residual connection and layer normalization
        x = x + attn_output
        x = self.norm1(x)

        # Feed-forward layer
        ff_output = self.linear(x)

        # Residual connection and layer normalization
        x = x + ff_output
        x = self.norm2(x)

        return x







class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, max_input_length,max_target_length, num_encoder_layers, num_decoder_layers, batch_size, output_dim, nhead,mask):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(output_dim,embedding_dim)
        self.input_position_encoder = PositionalEncoding(embedding_dim, max_input_length)
        self.target_position_encoder = PositionalEncoding(embedding_dim, max_target_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(embedding_dim, nhead) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_dim, nhead,max_target_length,batch_size,mask) for _ in range(num_decoder_layers)])
        self.linear = nn.Linear(max_input_length,max_target_length)
        self.output_linear = nn.Linear(embedding_dim,output_dim)
        self.batch_size = batch_size
        self.max_target_length = max_target_length
        self.embedding_dim = embedding_dim
            
    def forward(self, input, target):

        # positional encoding
        input = self.input_linear(input)
        target = self.input_linear(target)
        input_pos = self.input_position_encoder(input)
        target_pos = self.target_position_encoder(target)

        for layer in self.encoder_layers:
            input_pos = layer(input_pos)

            #print("encoder outputs:" + str(input.size()))
            #print(input)

        encoder_output = input_pos
        #print(encoder_output.size())
        """for i in range(self.batch_size):
          encoder_output[i] = torch.transpose(self.linear(torch.transpose(input_pos[i],0,1)),0,1)"""
         # Apply the decoder layers with the target_mask and input as context

        #print(encoder_output.device)

        for layer in self.decoder_layers:
            target_pos = layer(target_pos, encoder_output)
            #print(target.size())

        #output = self.output_linear(target_pos)
        out = self.output_linear(target_pos)
        return torch.sigmoid(torch.mul(out,10))

    def encode(self, x):
        input = self.input_linear(x)
        input = self.input_position_encoder(input)

        for layer in self.encoder_layers:
            input = layer(input)

        return input

    def decode(self,encoder_output,decoder_input):
        target = self.input_linear(decoder_input)
        target = self.target_position_encoder(target)
        for layer in self.decoder_layers:
            target = layer(target,encoder_output)

        out = self.output_linear(target)
        return torch.sigmoid(torch.mul(out,10))