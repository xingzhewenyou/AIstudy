import torch
import torch.nn as nn
import torch.optim as optim
import math

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = self.attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.out(attn_output)

        return output

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.attention(x, x, x, tgt_mask)
        x = self.norm1(x + attn_output)
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + cross_attn_output)
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, vocab_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_decoder_layers)])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(src.size(-1))
        tgt = self.embedding(tgt) * math.sqrt(tgt.size(-1))

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        output = self.fc_out(tgt)
        return output

# Example usage:
d_model = 512  # Embedding size
num_heads = 8  # Number of attention heads
num_encoder_layers = 6  # Number of encoder layers
num_decoder_layers = 6  # Number of decoder layers
d_ff = 2048  # Feed-forward network hidden size
vocab_size = 10000  # Vocabulary size

# Create a Transformer model
model = Transformer(d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, vocab_size)

# Sample data (batch_size, sequence_length)
src = torch.randint(0, vocab_size, (2, 10))  # Source sentences
tgt = torch.randint(0, vocab_size, (2, 10))  # Target sentences

# Forward pass
output = model(src, tgt)

# Print the output shape and values
print("Output shape:", output.shape)
print("Output values:", output)
