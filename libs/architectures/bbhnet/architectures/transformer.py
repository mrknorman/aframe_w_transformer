import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from typing import Callable, List, Literal, Optional

def detector_positional_enc(seq_len: int, model_dim: int, num_ifos: int) -> torch.Tensor:
    """
    Computes pre-determined positional encoding as in (Vaswani et al., 2017),
    accounting for different detectors.
    """
    pos = torch.arange(seq_len // num_ifos).unsqueeze(-1)
    dim = torch.arange(0, model_dim, step=2)

    frequencies = 1.0 / torch.pow(1000, (dim / model_dim))

    positional_encoding_table = []

    for i in range(num_ifos):
        detector_encoding = torch.zeros((seq_len // num_ifos, model_dim))
        detector_encoding[:, 0::2] = torch.sin(pos * frequencies + i * np.pi / num_ifos)
        detector_encoding[:, 1::2] = torch.cos(pos * frequencies + i * np.pi / num_ifos)
        positional_encoding_table.append(detector_encoding)

    return torch.cat(positional_encoding_table, dim=0).unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(head_size * num_heads, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout(attn_output)
        return attn_output
    
class TransformerEncoder(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length, ff_dim, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm((sequence_length, head_size * num_heads), eps=1e-6)
        self.multi_head_attention = MultiHeadAttention(head_size, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.layer_norm2 = nn.LayerNorm((sequence_length, head_size * num_heads), eps=1e-6)
        self.dense1 = nn.Linear(head_size * num_heads, ff_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(ff_dim, head_size * num_heads)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs):
        # Normalization and Attention
        x = self.layer_norm1(inputs)
        x = self.multi_head_attention(x)
        x = self.dropout1(x)
        res = x + inputs

        # Feed Forward Part
        x = self.layer_norm2(res)
        x = nn.ReLU()(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.dropout3(x)

        return x + res

class TransformerNet(nn.Module):
    def __init__(
        self,
        num_ifos: int,
        num_transformer_blocks: int = 1,
        head_size: int = 8,
        num_heads: int = 8,
        chunk_size: int = 1,
        ff_dim: int = 128
    ) -> None:
        super().__init__()

        self.embedding_size = num_heads * head_size
        self.chunk_size = chunk_size
        self.num_ifos = num_ifos
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
    
        # Chunk input data into num_chunks vectors of length chunk_size:
        self.chunking = nn.Unflatten(1, (chunk_size, -1))
        
        # Embedded input data into higher dimensionality for positional encoding:
        self.embedding = nn.Conv1d(
            chunk_size,
            self.embedding_size,
            kernel_size=1,
            stride=1
        )
        
        self.relu = nn.ReLU()

        # Average pool over each feature map to create a
        # single value for each feature map that we'll use
        # in the fully connected head
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _forward_impl(self, x: Tensor) -> Tensor:
        input_size = x.size(-1)
        self.sequence_length = ((self.num_ifos * input_size) // self.chunk_size)   
        
        # Now create transformer blocks:
        encoder_blocks = []
        for _ in range(self.num_transformer_blocks):
            block = TransformerEncoder(self.head_size, self.num_heads, self.sequence_length, self.ff_dim, dropout=0)
            encoder_blocks.append(block)
        
        device = x.device
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.fc = nn.Linear(self.sequence_length, 1).to(device)
        
        # Move the entire transformer block to the same device as the input tensor
        for block in self.encoder_blocks:
            block.to(device)
                    
        # Split the input tensor along the num_ifos dimension
        x = x.view(*x.shape[:-2], self.num_ifos * x.shape[-1])
        x = x.squeeze(dim=-2)
        x = torch.flatten(x, 1)
                
        x = self.chunking(x)
        
        x = self.embedding(x)
        x = self.relu(x)
                        
        # Add detector-specific positional encoding
        positional_encoding_table = detector_positional_enc(
            self.sequence_length, 
            self.embedding_size, 
            self.num_ifos
        ).to(device)
        
        x = x.permute(0,2,1)
        
        positional_encoding_table = positional_encoding_table.repeat(x.size(0), 1, 1)
                
        torch.add(x, positional_encoding_table)
                                                     
        for block in self.encoder_blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)