"""Source-circuit encoder for equivalence conditioning."""


# file exports these two main things:
# - SourceCircuitEncoderConfig: dataclass for the config of the source circuit encoder
# - SourceCircuitEncoder: the actual encoder module, which takes in embedded source circuit tensors and outputs conditioning embeddings for the diffusion model. 
# It consists of a convolutional projection followed by a transformer encoder
__all__ = ["SourceCircuitEncoderConfig", "SourceCircuitEncoder"]

# %% ../../src/models/source_circuit_encoder.ipynb
from ..imports import *
from .config_model import ConfigModel
from dataclasses import dataclass


@dataclass
class SourceCircuitEncoderConfig: # configuration container
    in_channels: int = 8
    cond_emb_size: int = 512
    hidden_channels: int = 256
    num_heads: int = 8
    depth: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0
    add_positional_encoding: bool = True

# The class assumes that source_latents = embedder.embed(source_tokens) was already called, so the input to the source encoder is already in the continuous embedding space. 
class SourceCircuitEncoder(ConfigModel):
    """
    Encodes an embedded source-circuit tensor into a conditioning sequence.

    Expected input:
        x: [batch, clr_dim, num_qubits, max_gates]

    Output:
        s_emb: [batch, seq, cond_emb_size]
    """

    def __init__( 
        self,
        in_channels: int = 8,
        cond_emb_size: int = 512,
        hidden_channels: int = 256,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        add_positional_encoding: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.cond_emb_size = cond_emb_size
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.add_positional_encoding = add_positional_encoding

        # CONVOLUTION BLOCK -> verify logic
        # 1x1 conv to project input to hidden dimension, then conv blocks to get to cond_emb_size, then flatten and transformer encoder layers, 
        # then layer norm. Positional encoding is added after the conv blocks, before flattening, using a custom 2D sin/cos encoding based on 
        # the height and width of the feature map (i.e. num_qubits and max_gates dimensions).
        self.proj_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(hidden_channels, cond_emb_size, kernel_size=1),
        )
        
        # TRANSFORMER ENCODER BLOCK: -> verify logic
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cond_emb_size,
            nhead=num_heads,
            dim_feedforward=cond_emb_size * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(cond_emb_size)

        # store config for saving and loading
        self.params_config = SourceCircuitEncoderConfig(
            in_channels=in_channels,
            cond_emb_size=cond_emb_size,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            add_positional_encoding=add_positional_encoding,
        )

    # VERIFY! 
    # build 2d sinusoidal positional embedding based on height and width of the feature map, 
    # with separate sin/cos for x and y dimensions, and interleaved in the channel dimension.
    @staticmethod
    def _build_2d_sincos_pos_emb(h: int, w: int, c: int, device, dtype):
        """
        Returns positional embedding of shape [1, c, h, w].
        Requires c % 4 == 0.
        """
        if c % 4 != 0:
            raise ValueError(
                f"`cond_emb_size` must be divisible by 4 for 2D sin/cos encoding, got {c}."
            )

        quarter = c // 4

        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # [h, w]

        omega = torch.arange(quarter, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / max(quarter, 1)))

        out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
        out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)

        pos = torch.cat(
            [
                torch.sin(out_y),
                torch.cos(out_y),
                torch.sin(out_x),
                torch.cos(out_x),
            ],
            dim=1,
        )  # [h*w, c]

        pos = pos.reshape(h, w, c).permute(2, 0, 1).unsqueeze(0)  # [1, c, h, w]
        return pos

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        x: [batch, in_channels, num_qubits, max_gates]
        returns: [batch, seq, cond_emb_size]
        """
        if x.dim() != 4:
            raise ValueError(
                f"`SourceCircuitEncoder` expects 4D input [b,c,s,t], got shape {tuple(x.shape)}."
            )

        x = self.proj_in(x)
        x = self.conv(x)

        if self.add_positional_encoding:
            _, c, h, w = x.shape
            x = x + self._build_2d_sincos_pos_emb(h, w, c, x.device, x.dtype)

        x = x.flatten(2).transpose(1, 2)  # [b, h*w, cond_emb_size]
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x