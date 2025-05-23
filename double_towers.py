import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from GNN import GNNLayer

# Set seed for reproduction
torch.manual_seed(2)
np.random.seed(3)


class TransGNNModel(nn.Module):
    """
    TransGNN Model
    """
    def __init__(self, model_config):
        """
        Initialization
        """
        super(TransGNNModel, self).__init__()
        # Basic config
        self.model_config = model_config
        self.drug_max_seq = model_config['drug_max_seq']
        self.target_max_seq = model_config['target_max_seq']
        self.emb_size = model_config['emb_size']
        self.dropout_ratio = model_config['dropout_ratio']
        self.input_drug_dim = model_config['input_drug_dim']
        self.input_target_dim = model_config['input_target_dim']
        self.layer_size = model_config['layer_size']

        # Model config
        self.interm_size = model_config['interm_size']
        self.num_attention_heads = model_config['num_attention_heads']
        self.attention_dropout_ratio = model_config['attention_dropout_ratio']
        self.hidden_dropout_ratio = model_config['hidden_dropout_ratio']
        self.hidden_size = model_config['emb_size']
        self.flatten_dim = self.hidden_size * self.drug_max_seq * self.target_max_seq

        # Enhanced embeddings
        self.drug_emb = EnhancedEmbedding(self.input_drug_dim, self.emb_size, self.drug_max_seq, self.dropout_ratio)
        self.target_emb = EnhancedEmbedding(self.input_target_dim, self.emb_size, self.target_max_seq, self.dropout_ratio)
        
        # Encoder module
        self.encoder = EncoderModule(self.layer_size, self.hidden_size, self.interm_size, self.num_attention_heads, 
                                      self.attention_dropout_ratio, self.hidden_dropout_ratio)
        
        # GNN Layer
        self.gnn_layer = GNNLayer(self.hidden_size, self.hidden_size)
        
        # Decoder module
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, d, t, d_masking, t_masking):
        """
        TransGNN forward pass
        """
        batch_size = d.size(0)
        chunk_size = 16  # 分批次处理的大小
        res_list = []

        for i in range(0, batch_size, chunk_size):
            d_chunk = d[i:i + chunk_size]
            t_chunk = t[i:i + chunk_size]
            d_masking_chunk = d_masking[i:i + chunk_size]
            t_masking_chunk = t_masking[i:i + chunk_size]

            tempd_masking = d_masking_chunk.unsqueeze(1).unsqueeze(2)
            tempt_masking = t_masking_chunk.unsqueeze(1).unsqueeze(2)

            tempd_masking = (1.0 - tempd_masking) * -10000.0
            tempt_masking = (1.0 - tempt_masking) * -10000.0

            d_embedding = self.drug_emb(d_chunk)
            t_embedding = self.target_emb(t_chunk)
            
            d_encoder = self.encoder(d_embedding.float(), tempd_masking.float())
            t_encoder = self.encoder(t_embedding.float(), tempt_masking.float())

            d_encoder_2d = d_encoder.view(-1, d_encoder.size(-1))
            t_encoder_2d = t_encoder.view(-1, t_encoder.size(-1))

            adj_d = torch.eye(d_encoder.size(1)).to(d_encoder.device)
            adj_t = torch.eye(t_encoder.size(1)).to(t_encoder.device)

            if d_encoder_2d.size(0) != adj_d.size(0):
                adj_d = torch.eye(d_encoder_2d.size(0)).to(d_encoder.device)
            if t_encoder_2d.size(0) != adj_t.size(0):
                adj_t = torch.eye(t_encoder_2d.size(0)).to(t_encoder.device)

            d_gnn = self.gnn_layer(adj_d, d_encoder_2d)
            t_gnn = self.gnn_layer(adj_t, t_encoder_2d)

            drug_res = d_gnn.view(d_chunk.size(0), -1, self.hidden_size).unsqueeze(2)
            target_res = t_gnn.view(t_chunk.size(0), -1, self.hidden_size).unsqueeze(1)

            i_score = (drug_res * target_res).sum(dim=-1)
            i_score = i_score.view(i_score.size(0), -1)

            if i_score.size(1) != self.decoder[0].in_features:
                self.decoder[0] = nn.Linear(i_score.size(1), self.decoder[0].out_features).to(i_score.device)

            res_chunk = self.decoder(i_score)
            res_list.append(res_chunk)

        res = torch.cat(res_list, dim=0)
        return res


class EnhancedEmbedding(nn.Module):
    """
    Enhanced Embeddings of drug, target
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_ratio):
        """
        Initialization
        """
        super(EnhancedEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_id):
        """
        Embeddings
        """
        seq_len = input_id.size(1)
        position_id = torch.arange(seq_len, dtype=torch.long, device=input_id.device)
        position_id = position_id.unsqueeze(0).expand_as(input_id)

        word_embeddings = self.word_embedding(input_id)
        position_embeddings = self.position_embedding(position_id)

        embedding = word_embeddings + position_embeddings
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class EncoderModule(nn.Module):
    """
    Encoder Module with multiple layers
    """
    def __init__(self, layer_size, hidden_size, interm_size, num_attention_heads, 
                 attention_dropout_ratio, hidden_dropout_ratio):
        """
        Initialization
        """
        super(EncoderModule, self).__init__()
        module = Encoder(hidden_size, interm_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio)
        self.module = nn.ModuleList([module for _ in range(layer_size)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Multiple encoders
        """
        for layer_module in self.module:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states

  
class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, hidden_size, interm_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio):
        """
        Initialization
        """
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio)
        self.latent = LatentModule(hidden_size, interm_size)
        self.output = Output(interm_size, hidden_size, hidden_dropout_ratio)

    def forward(self, hidden_states, attention_mask):
        """
        Encoder block
        """
        attention_temp = self.attention(hidden_states, attention_mask)
        latent_temp = self.latent(attention_temp)
        module_output = self.output(latent_temp, attention_temp)
        return module_output


class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio):
        """
        Initialization
        """
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_dropout_ratio)
        self.output = SelfOutput(hidden_size, hidden_dropout_ratio)

    def forward(self, input_tensor, attention_mask):
        """
        Attention block
        """
        attention_output = self.self(input_tensor, attention_mask)
        self_output = self.output(attention_output, input_tensor)
        return self_output


class LatentModule(nn.Module):
    """
    Intermediate Layer
    """
    def __init__(self, hidden_size, interm_size):
        """
        Initialization
        """
        super(LatentModule, self).__init__()
        self.connecter = nn.Linear(hidden_size, interm_size)

    def forward(self, hidden_states):
        """
        Latent block
        """
        hidden_states = self.connecter(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class Output(nn.Module):
    """
    Output Layer
    """
    def __init__(self, interm_size, hidden_size, hidden_dropout_ratio):
        """
        Initialization
        """
        super(Output, self).__init__()
        self.connecter = nn.Linear(interm_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_ratio)

    def forward(self, hidden_states, input_tensor):
        """
        Output block
        """
        hidden_states = self.connecter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SelfOutput(nn.Module):
    """
    Self-Output Layer
    """
    def __init__(self, hidden_size, hidden_dropout_ratio):
        """
        Initialization
        """
        super(SelfOutput, self).__init__()
        self.connecter = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_ratio)

    def forward(self, hidden_states, input_tensor):
        """
        Self-output block
        """
        hidden_states = self.connecter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SelfAttention(nn.Module):
    """
    Self-Attention
    """
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_ratio):
        """
        Initialization
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                 "The hidden size (%d) is not a product of the number of attention heads (%d)" % 
                 (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_size

        self.q = nn.Linear(hidden_size, self.all_head_size)
        self.k = nn.Linear(hidden_size, self.all_head_size)
        self.v = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout_ratio)

    def score_transpose(self, x):
        """
        Score transpose
        """
        temp = x.size()[:-1] + (self.num_attention_heads, self.head_size)
        x = x.view(*temp)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """
        Self-Attention block
        """
        temp_q = self.q(hidden_states)
        temp_k = self.k(hidden_states)
        temp_v = self.v(hidden_states)

        q_layer = self.score_transpose(temp_q)
        k_layer = self.score_transpose(temp_k)
        v_layer = self.score_transpose(temp_v)

        attention_score = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.head_size)
        attention_score = attention_score + attention_mask

        attention_prob = nn.Softmax(dim=-1)(attention_score)
        attention_prob = self.dropout(attention_prob)

        attention_layer = torch.matmul(attention_prob, v_layer)
        attention_layer = attention_layer.permute(0, 2, 1, 3).contiguous()

        temp_attention_layer = attention_layer.size()[:-2] + (self.all_head_size,)
        attention_layer = attention_layer.view(*temp_attention_layer)
        return attention_layer