# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

from models.VIT.vit import Transformer

class MAE_Padded(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        pos_mode = "none",
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:] # includes extra pos and embedding 
        self.max_patches = num_patches

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        self.pixel_values_per_patch = encoder.patch_dim
        self.pad_token = nn.Parameter(torch.randn(encoder_dim))

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches + (self.encoder.n_pos - 1), decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, self.pixel_values_per_patch)
        self.pos_mode=pos_mode

    def process_tokens(self, tokens, embedding, num_patches, batch):
        device = tokens.device
        # add embedding token
        if embedding is not None:
            embedding = embedding.to(device)
            embedding_token = self.encoder.embedding_to_token(embedding) # [b, d]
            embedding_token = repeat(embedding_token, 'b d -> b 1 d')
            tokens = torch.cat((tokens, embedding_token), dim=1) # appended to end to avoid being masked
                
        # Add pad tokens to tokens if needed
        num_extra_tokens = (self.max_patches - self.encoder.n_pos) - num_patches
        if num_extra_tokens > 0: 
            extra_tokens = repeat(self.pad_token, 'd -> b n d', b = batch, n = num_extra_tokens)
            tokens = torch.cat((tokens, extra_tokens), dim = 1)
        
        return tokens

    def add_indices(self, unmasked_indices, embedding, num_patches, batch, device):

        if embedding is not None:
            embedding_indices = torch.tensor([num_patches], device = device).repeat(batch, 1)
            unmasked_indices = torch.cat((unmasked_indices, embedding_indices), dim = 1) # add embedding index to unmasked indices
        
        num_extra_tokens = (self.max_patches - self.encoder.n_pos) - num_patches
        if num_extra_tokens > 0:
            offset = self.encoder.n_pos - 1
            extra_indices = torch.arange(num_patches+offset, num_patches + num_extra_tokens + offset, device = device).repeat(batch, 1)
            unmasked_indices = torch.cat((unmasked_indices, extra_indices), dim = 1)
        
        return unmasked_indices
    
    def remove_indices(self, tokens, num_patches):
        if num_patches < self.max_patches - 1:
            tokens = tokens[:, :num_patches]
        return tokens

    def forward(self, img, embedding=None, normalizer=None, features=False):

        if normalizer is not None:
            img = normalizer.normalize(img)
        img = img.unsqueeze(1) # add channel dimension
        device = img.device

        # get patches

        patches = self.to_patch(img) # b c nt nx ny -> b (f h w) (p1 p2 pf c)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches) # b (f h w) (p1 p2 pf c) -> b (f h w) d

        # add embedding token or padding if used
        tokens = self.process_tokens(tokens, embedding, num_patches, batch)

        if self.encoder.pool == "cls":
            pos = self.encoder.pos_embedding[:, 1:(self.max_patches)]
        elif self.encoder.pool == "mean":
            pos = self.encoder.pos_embedding
        tokens = tokens + pos

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # Add extra unmasked indices to prevent pad and embedding tokens from being masked
        unmasked_indices = self.add_indices(unmasked_indices, embedding, num_patches, batch, device)

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices] # b unmasked d

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens) # b unmasked d

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens) # b unmasked decoder_dim

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, self.max_patches - 1 , self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens) # b (f h w) d

        decoded_tokens = self.remove_indices(decoded_tokens, num_patches)

        if features:
            # construct mask based on masked indices
            mask = torch.zeros(batch, num_patches, self.pixel_values_per_patch)
            mask[batch_range, masked_indices] = 1
            mask = mask.to(device)
                    
            # project decoded tokens to pixel space
            decoded_patches = self.to_pixels(decoded_tokens) # b (f h w) (p1 p2 pf c)

            x_orig = self.encoder.unpatchify(patches).squeeze() # b (f h w) (p1 p2 pf c) -> b c (pf f) (p1 h) (p2 w)
            x_rec = self.encoder.unpatchify(decoded_patches).squeeze()
            mask = self.encoder.unpatchify(mask).squeeze()

            if normalizer is not None:
                x_orig = normalizer.denormalize(x_orig)
                x_rec = normalizer.denormalize(x_rec)

            return x_orig, x_rec, mask
        
        else: # does fewer computations on decoded patches
            masked_patches = patches[batch_range, masked_indices] # b num_masked (p1 p2 pf c)

            mask_tokens = decoded_tokens[batch_range, masked_indices] # b num_masked d
            pred_pixel_values = self.to_pixels(mask_tokens) # b num_masked (p1 p2 pf c)

            recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
            return recon_loss