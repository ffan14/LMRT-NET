# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from scipy import ndimage
from . import transformer_configs as configs
import torch.fft as fft
from models import DSF


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size##paraphrase

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch and position embeddings dynamically based on input tensor dimensions."""
    def __init__(self, config, img_size=None, in_channels=3, input_dim=3, old=1):
        super(Embeddings, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.transformer["dropout_rate"]

        if img_size is not None:
            h, w = img_size
        else:
            h, w = config.image_size  # or any default size you choose

        grid_size = self.config.patches["grid"]
        patch_size = (h // grid_size[0], w // grid_size[1])
        patch_size_real = (patch_size[0], patch_size[1])
        n_patches = (h // patch_size_real[0]) * (w // patch_size_real[1])

        # Learnable patch embeddings
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=self.hidden_size,
                                          kernel_size=patch_size_real,
                                          stride=patch_size_real)
        
        # Learnable positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, self.hidden_size))

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        # print(self.positional_encoding.shape)
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel").replace('\\', '/')]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel").replace('\\', '/')]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel").replace('\\', '/')]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel").replace('\\', '/')]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias").replace('\\', '/')]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias").replace('\\', '/')]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias").replace('\\', '/')]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias").replace('\\', '/')]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel").replace('\\', '/')]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel").replace('\\', '/')]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias").replace('\\', '/')]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias").replace('\\', '/')]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale").replace('\\', '/')]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias").replace('\\', '/')]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale").replace('\\', '/')]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias").replace('\\', '/')]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self,config, img_size, vis,in_channels=3,old = 1):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config,img_size=img_size,input_dim=in_channels,old = old)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class DART_block(nn.Module):
    def __init__(self,config, input_dim, img_size=224,transformer = None, up=0):
        super(DART_block, self).__init__()
        self.up = up
        self.transformer = transformer
        self.config = config
        ngf = 64
        mult = 4
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        if self.transformer:
            # Downsample
            model = [nn.Conv2d(ngf * 4, 1024, kernel_size=1,
                               stride=1, padding=0, bias=use_bias),
                     norm_layer(1024),
                     nn.ReLU(True)]
            setattr(self, 'downsample', nn.Sequential(*model))
            #Patch embedings
            self.embeddings = Embeddings(config, img_size=img_size, input_dim=input_dim, in_channels= 1024)
            # Upsampling block
            if self.up == 0:
                model = [nn.ConvTranspose2d(self.config.hidden_size, ngf * 8,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(ngf * 8),
                        nn.ReLU(True)]
                model += [nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(ngf * 4),
                        nn.ReLU(True)]
                setattr(self, 'upsample', nn.Sequential(*model))
            elif self.up == 1:
                model = [nn.ConvTranspose2d(self.config.hidden_size, ngf * 4,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(ngf * 4),
                        nn.ReLU(True)]
                setattr(self, 'upsample', nn.Sequential(*model))
            elif self.up == 2:
                model = [nn.ConvTranspose2d(self.config.hidden_size, ngf * 4,
                                            kernel_size=1, stride=1,
                                            padding=0, output_padding=0,
                                            bias=use_bias),
                        norm_layer(ngf * 4),
                        nn.ReLU(True)]
                setattr(self, 'upsample', nn.Sequential(*model))
            #Channel compression
            self.cc = channel_compression(ngf * 8, ngf * 4)
        # Residual CNN
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        if self.transformer:
            # downsample
            down_sampled = self.downsample(x)
            # embed
            embedding_output = self.embeddings(down_sampled)
            # feed to transformer
            transformer_out, attn_weights = self.transformer(embedding_output)
            B, n_patch, hidden = transformer_out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            transformer_out = transformer_out.permute(0, 2, 1)
            transformer_out = transformer_out.contiguous().view(B, hidden, h, w)
            # upsample transformer output
            # print(transformer_out.shape)
            transformer_out = self.upsample(transformer_out)
            x = torch.cat([transformer_out, x], dim=1)
            # channel compression
            x = self.cc(x)
        # residual CNN
        x = self.residual_cnn(x)
        return x

########Generator############
class LMRT(nn.Module):
    def __init__(self,config, input_dim, img_size=224, output_dim=3, vis=False):
        super(LMRT, self).__init__()
        self.transformer_encoder = Encoder(config, vis)
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4
        self.dsf1 = DSF.DSF(256) # channel
        self.dsf2 = DSF.DSF(256)

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ############################################################################################
        # Layer4-Encoder4
        model = []
        model = [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_4', nn.Sequential(*model))
        ############################################################################################
        # Layer5-Encoder5
        model = []
        model = [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_5', nn.Sequential(*model))
        ####################################DART Blocks##############################################
        mult = 4
        # self.dart_1_1 = DART_block(self.config, input_dim, img_size = (64, 64),transformer = self.transformer_encoder, up=0)
        self.dart_2_1 = DART_block(self.config, input_dim, img_size = (32, 32),transformer = self.transformer_encoder, up=1)
        self.dart_3_1 = DART_block(self.config, input_dim, img_size = (16, 16), transformer= self.transformer_encoder, up=2)
        # self.dart_1_2 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_2_2 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_3_2 = DART_block(self.config, input_dim, img_size, transformer= None)
        # self.dart_1_3 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_2_3 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_3_3 = DART_block(self.config, input_dim, img_size, transformer= None)

        # 2 * 4
        self.dart_2_4 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_3_4 = DART_block(self.config, input_dim, img_size, transformer= None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, ngf * mult,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(ngf * mult),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, ngf * mult,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(ngf * mult),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_3', nn.Sequential(*model))
        ############################################################################################
        # Layer16-Decoder4
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_4', nn.Sequential(*model))
        ############################################################################################
        # Layer17-Decoder5
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_5', nn.Sequential(*model))

    ############################################################################################
        
    def forward(self, x):
        # Pass input through cnn encoder of ResViT
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x_1 = self.encoder_3(x) # 对应1_1
        x_2 = self.encoder_4(x_1) # 对应2_1
        x_3 = self.encoder_5(x_2) # 对应3_1
    
        # 2 * 4
        # 第一列
        x_2_1 = self.dart_2_1(x_2)
        x_3_1 = self.dart_3_1(x_3)

        # 第二列
        x_3_1_u = F.interpolate(x_3_1, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_2_2 = self.dart_2_2(x_2_1 + x_3_1_u)
        x_3_2 = self.dart_3_2(x_3_1) 

        # 第三列
        x_3_2_u = F.interpolate(x_3_2, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_2_3 = self.dart_2_3(x_2_2 + x_3_2_u)
        x_3_3 = self.dart_3_3(x_3_2) 

        # 第四列
        x_3_3_u = F.interpolate(x_3_3, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_2_4 = self.dart_2_4(x_2_3 + x_3_3_u)
        x_3_4 = self.dart_3_4(x_3_3) 

        #decoder
        x = self.decoder_1(x_3_4)
        x = self.dsf1(x, x_3_4, x_2_4)
        x = self.decoder_2(x)
        x = self.dsf2(x, x_2_4, x_1)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.decoder_5(x)
        return x

    def load_from(self, weights):
        # return
        with torch.no_grad():

            res_weight = weights
            # print(self.dart_1_1.embeddings.patch_embeddings.weight.shape)
            print(np2th(res_weight["embedding/kernel"], conv=True).shape)
            if self.config.name == 'b16':
                # self.dart_1_1.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                # self.dart_1_1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

                self.dart_2_1.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.dart_2_1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

                self.dart_3_1.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.dart_3_1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer_encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            # posemb_new = self.dart_1_1.embeddings.positional_encoding
            # if posemb.size() == posemb_new.size():
            #     self.dart_1_1.embeddings.positional_encoding.copy_(posemb)
            # elif posemb.size()[1] - 1 == posemb_new.size()[1]:
            #     posemb = posemb[:, 1:]
            #     self.dart_1_1.embeddings.positional_encoding1.copy_(posemb)
            # else:
            #     logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
            #     ntok_new = posemb_new.size(1)
            #     _, posemb_grid = posemb[:, :1], posemb[0, 1:]
            #     gs_old = int(np.sqrt(len(posemb_grid)))
            #     gs_new = int(np.sqrt(ntok_new))
            #     print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
            #     posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
            #     zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            #     posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
            #     posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            #     posemb = posemb_grid
            #     self.dart_1_1.embeddings.positional_encoding.copy_(np2th(posemb))

            #############
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.dart_2_1.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.dart_2_1.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.dart_2_1.embeddings.positional_encoding.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.dart_2_1.embeddings.positional_encoding.copy_(np2th(posemb))

            #############
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.dart_3_1.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.dart_3_1.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.dart_3_1.embeddings.positional_encoding.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.dart_3_1.embeddings.positional_encoding.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer_encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


class Res_CNN(nn.Module):
    def __init__(self, config, input_dim, img_size=224, output_dim=3, vis=False):
        super(Res_CNN, self).__init__()
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4
        # MIF
        self.dsf1 = DSF.DSF(256)
        self.dsf2 = DSF.DSF(256)
        # self.efc1 = EFC(256, 256)
        # self.efc2 = EFC(256, 256)
        # self.eff1 = EFF(256, 256)
        # self.eff2 = EFF(256, 256)

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ############################################################################################
        # Layer4-Encoder4
        model = []
        model = [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_4', nn.Sequential(*model))
        ############################################################################################
        # Layer5-Encoder5
        model = []
        model = [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_5', nn.Sequential(*model))
        ####################################DART Blocks##############################################
        mult = 4
        # self.dart_1_1 = DART_block(self.config, input_dim, img_size = (64, 64),transformer = self.transformer_encoder, up=0)
        self.dart_2_1 = DART_block(self.config, input_dim, img_size = (32, 32),transformer = None, up=1)
        self.dart_3_1 = DART_block(self.config, input_dim, img_size = (16, 16), transformer= None, up=2)
        # self.dart_1_2 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_2_2 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_3_2 = DART_block(self.config, input_dim, img_size, transformer= None)
        # self.dart_1_3 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_2_3 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_3_3 = DART_block(self.config, input_dim, img_size, transformer= None)

        # 2 * 4
        self.dart_2_4 = DART_block(self.config, input_dim, img_size,transformer = None)
        self.dart_3_4 = DART_block(self.config, input_dim, img_size, transformer= None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, ngf * mult,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(ngf * mult),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, ngf * mult,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(ngf * mult),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_3', nn.Sequential(*model))
        ############################################################################################
        # Layer16-Decoder4
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_4', nn.Sequential(*model))
        ############################################################################################
        # Layer17-Decoder5
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_5', nn.Sequential(*model))

    ############################################################################################

    def forward(self, x):
        # Pass input through cnn encoder of ResViT
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x_1 = self.encoder_3(x) # 对应1_1
        x_2 = self.encoder_4(x_1) # 对应2_1
        x_3 = self.encoder_5(x_2) # 对应3_1

        # 2 * 4
        # 第一列
        x_2_1 = self.d_2_1(x_2)
        x_3_1 = self.dart_3_1(x_3)

        # 第二列
        x_3_1_u = F.interpolate(x_3_1, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_2_2 = self.dart_2_2(x_2_1 + x_3_1_u)
        x_3_2 = self.dart_3_2(x_3_1) 

        # 第三列
        x_3_2_u = F.interpolate(x_3_2, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_2_3 = self.dart_2_3(x_2_2 + x_3_2_u)
        x_3_3 = self.dart_3_3(x_3_2) 

        # 第四列
        x_3_3_u = F.interpolate(x_3_3, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_2_4 = self.dart_2_4(x_2_3 + x_3_3_u)
        x_3_4 = self.dart_3_4(x_3_3) 

        #decoder
        x = self.decoder_1(x_3_4)
        x = self.dsf1(x, x_3_4, x_2_4)
        # x = self.efc1((x_2_4, x))
        # x = self.eff1(x, x_2_4)
        x = self.decoder_2(x)
        x = self.dsf2(x, x_2_4, x_1)
        # x = self.efc2((x_1,x))
        # x = self.eff2(x, x_1)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.decoder_5(x)
        return x




class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'Res-ViT-B_16': configs.get_resvit_b16_config(),
    'Res-ViT-L_16': configs.get_resvit_l16_config(),
}


