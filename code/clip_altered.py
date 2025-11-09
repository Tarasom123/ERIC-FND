import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from feature_extraction import BertEncoder,ResNetEncoder,VggEncoder
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class ImagePatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, d_model=128, in_channels=3):
        super(ImagePatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)  # (img_size, img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv_layer = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        B, C, H, W = image.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        image = self.conv_layer(image).flatten(2).transpose(1, 2)  # (B, H*W, D)
        return image
class CLIP(nn.Module):
    def __init__(self, out_channels,
        embeddings,
        data_name,
        shared_image_dim=128,
        shared_text_dim=128):
        super(CLIP, self).__init__()


        self.img_patch_embed = ImagePatchEmbed(224, 16, 256)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, 256))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (224 // 16) * (224 // 16)
        n = img_len // 2 + 1
        # Text
        self.text_embed = nn.Embedding(32)
        self.text_embed.weight = nn.Parameter(torch.from_numpy(W))
        self.text_encoder = nn.Sequential(nn.Linear(d_text, d_model),
                                          nn.LayerNorm(d_model),
                                          TextPositionEmbed(seq_len, d_model, dropout))
        s = seq_len // 2 + 1


        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )
        self.img_fc = nn.Linear(shared_image_dim, out_channels)

        self.text_enc = BertEncoder(embeddings)
        if data_name == 'twitter':
            self.vision_enc = VggEncoder()
        else:
            self.vision_enc = ResNetEncoder()
        self.shared_text_linear = nn.Sequential(
            nn.Linear(300, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        ntoken, ninp, nhead, nhid, nlayers, dropout = 49408, 768, 8, 2048, 12, 0.5
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.text_fc = nn.Linear(ninp, out_channels)

    def forward(self, text, mask, image):

        text_encoding = self.text_enc(text,mask)


        img = self.vision_enc(image)
        img_out = self.shared_image(img)

        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1) 

        
        text_shared = self.shared_text_linear(text_encoding)
        text_shared = text_shared.long()
        src = self.encoder(text_shared) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        text_out = self.transformer_encoder(src, None)

        text_out = text_out[:, -1, :]
        text_out = self.text_fc(text_out)
        text_out = F.normalize(text_out, p=2, dim=-1)

        return img_out, text_out

    def encode_image(self, image):
        n_batch = image.size(0)

        out = self.img_model(image)
        out = self.avg_pool(out)
        out = out.view(n_batch, -1)
        out = self.img_fc(out)

        return out

    def encode_text(self, text):
        src = self.encoder(text) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, None)

        out = out[:, -1, :]
        out = self.text_fc(out)

        return out