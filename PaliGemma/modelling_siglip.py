from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size=768,
            intermediate_size = 3072,
            num_hidden_layers = 12,
            num_attention_heads = 12,
            num_channels = 3,
            image_size = 224,
            patch_size = 16,
            layer_norm_eps = 1e-12,
            attention_dropout = 0.0,
            num_image_tokens : int = None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size 
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens 


class SiglipVisionEmbeddings(nn.Modeule):
    def __init__(self, config:SiglipVisionconfig):
        super().__init__()
        self.config = config 
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_position = self.num_patches
        self.position_embeddings = nn.Embedding(
            self.num_position,
            self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _,_, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedddings(pixel_values) # [batch_size, embed_dim, num_patches_h, num_pacthes_w] where num_patches_h * num_patches_w = num_patches
        patch_embeds = patch_embeds.flatten(2) # [batch_size, embed_dim, num_patchs]
        patch_embeds = patch_embeds.transpose(1, 2) # [batch_Size, num_patches, embed_dim]
        
        # Add position embeddings to the patch embeddings
        embeddings = patch_embeds + self.position_embeddings(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)  # GELU activation
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipAttention(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init_()
        

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states  = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states  = self.layer_norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual # [Batch_Size, Num_Patches, Embed_Dim]

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) #generate embedding from images
        self.encoder = SiglipEncoder(config) #send images into transformer encoder layers
        self.post_normlayer = nn.LayerNorm(embed_dim, eps=config.layer)

    def forward(self, pixel_values : torch.Tensor ) -> torch.Tensor:

        hidden_state = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_state)
        last_hidden_state = self.post_normlayer(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches(num_img_tokens), Embed_Dim]
        return self.vision_model(pixel_values = pixel_values) # output is a batch of list of embeddings of length embedding vector for each image
    
    
    
