import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat
import math


##########################################################################################################################

class FeatureMLPEncoder(nn.Module):
    """
    Simple MLP encoder for feature-based input data
    """
    def __init__(self, in_dim, out_dim):
        super(FeatureMLPEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        self.out_dim = out_dim
        self.in_dim = in_dim

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x


#########################################################################################################################
class FCN(nn.Module):
    """
    Fully Convolutional Network encoder for time series data
    Adapted from https://doi.org/10.1145/3534678.3539134
    """
    def __init__(self, n_channels, out_channels=128, out_size=None):
        super(FCN, self).__init__()


        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(), 
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.out_len = 18

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels
        if out_size is not None:
            self.out_dim = out_size * out_channels


    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        B, C, T = x.shape

        # Choose pooling size to match desired_output_dim = C * pooled_T
        pooled_T = max(1, self.out_dim // C)

        # Apply adaptive pooling to reach that
        x = F.adaptive_avg_pool1d(x, pooled_T)
        x = x.view(B, -1)

        return x

##########################################################################################################################

class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM encoder for time series data
    Adapted from https://doi.org/10.1145/3534678.3539134
    """
    def __init__(self, n_channels, conv_kernels=64, kernel_size=5, LSTM_units=128):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        return x
    
##########################################################################################################################
# Transformer Encoder


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Ensure max_len is an integer
        max_len = int(max_len)

        pe = torch.zeros(max_len, d_model)  # No error now
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        self.attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', self.attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class Seq_Transformer(nn.Module):
    def __init__(self, n_channel, len_sw, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(n_channel, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position = PositionalEncoding(d_model=dim, max_len=len_sw)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        x = self.position(x)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for time series data
    Adapted from https://doi.org/10.1145/3534678.3539134
    """
    def __init__(self, n_channels, len_sw, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.transformer(x)
        return x

##########################################################################################################################
        
def get_encoder_from_name(cfg):
    name = cfg.ENCODER_NAME.lower()
    if name.lower() == 'fcn':
        from common.encoders import FCN
        out_size = cfg.OUT_DIM // 128
        return FCN(cfg.UCIHAR_NUM_CHANNELS, out_size=out_size)
    elif cfg.ENCODER_NAME.lower() == 'deepconvlstm':
        from common.encoders import DeepConvLSTM
        return DeepConvLSTM(cfg.UCIHAR_NUM_CHANNELS)
    elif cfg.ENCODER_NAME.lower() == 'transformer':
        from common.encoders import TransformerEncoder
        return TransformerEncoder(cfg.UCIHAR_NUM_CHANNELS, seq_len=cfg.UCIHAR_SEQ_LEN)
    else:
        raise ValueError(f"Unknown encoder name: {cfg.ENCODER_NAME}. Supported: 'FCN', 'DeepConvLSTM', 'Transformer'")




# create all models and print the number of parameters
def print_model_params(model, input_shape):
    num_params = sum(p.numel() for p in model.parameters())
    dummy_input = torch.randn(*input_shape)  # Create a dummy input tensor with the specified shape
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(dummy_input)
    output_shape = output[1].shape if isinstance(output, tuple) else output.shape  # Handle tuple outputs
    print(f"Model: {model.__class__.__name__}, Parameters: {num_params} ({num_params / 1e6:.2f}M), Output Shape: {output_shape}")

if __name__ == "__main__":
    # Example usage
    n_channels = 9
    n_classes = 6
    batch_size = 64
    input_shape = (batch_size, 128, n_channels)  # Example input shape (batch_size, sequence_length, n_channels)
    print(f"Input shape: {input_shape}")

    model = FCN(n_channels, out_size=1)
    print_model_params(model, input_shape)

    model = DeepConvLSTM(n_channels)
    print_model_params(model, input_shape)

    model = TransformerEncoder(n_channels, 128)
    print_model_params(model, input_shape)
