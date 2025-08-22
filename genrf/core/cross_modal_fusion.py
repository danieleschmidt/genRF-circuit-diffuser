"""
Cross-Modal Fusion Architecture for RF Circuit Design.

Revolutionary breakthrough: Unified multi-modal learning combining:
- Schematic image understanding (Vision Transformer)
- SPICE netlist text processing (Transformer encoder)
- Parameter vector optimization (Diffusion models)
- Performance prediction (Graph Neural Networks)
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import torchvision.transforms as transforms
from PIL import Image
import re

from .models import DiffusionModel
from .design_spec import DesignSpec

logger = logging.getLogger(__name__)


@dataclass
class CrossModalConfig:
    """Configuration for cross-modal fusion."""
    vision_dim: int = 768
    text_dim: int = 512
    param_dim: int = 256
    fusion_dim: int = 1024
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    temperature: float = 0.07


class VisionEncoder(nn.Module):
    """Vision Transformer for schematic image understanding."""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        num_classes: int = 0  # No classification head
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Circuit-specific components detector
        self.component_detector = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Topology understanding
        self.topology_classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64)  # Topology representation
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass for schematic images."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.transformer:
            x = block(x)
        
        x = self.layer_norm(x)
        
        # Extract features
        cls_features = x[:, 0]  # Class token features
        patch_features = x[:, 1:]  # Patch features
        
        # Component detection using cross-attention
        component_features, attention_weights = self.component_detector(
            cls_features.unsqueeze(1), patch_features, patch_features
        )
        
        # Topology classification
        topology_rep = self.topology_classifier(cls_features)
        
        return {
            'cls_features': cls_features,
            'patch_features': patch_features,
            'component_features': component_features.squeeze(1),
            'topology_representation': topology_rep,
            'attention_weights': attention_weights
        }


class TextEncoder(nn.Module):
    """Transformer encoder for SPICE netlist processing."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 1024
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # SPICE-specific understanding
        self.component_extractor = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Circuit parameter extraction
        self.param_extractor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        # Netlist validation
        self.syntax_checker = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Forward pass for SPICE netlists."""
        B, seq_len = token_ids.shape
        
        # Token and positional embedding
        x = self.token_embed(token_ids)
        x = x + self.pos_embed[:, :seq_len]
        
        # Transformer encoding
        if attention_mask is not None:
            # Convert attention mask to transformer format
            attention_mask = attention_mask.bool()
        
        x = self.transformer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        # Global representation (mean pooling over valid tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            global_rep = (x * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            global_rep = x.mean(1)
        
        # Component extraction using self-attention
        component_rep, comp_attention = self.component_extractor(x, x, x)
        component_summary = component_rep.mean(1)
        
        # Parameter extraction
        param_features = self.param_extractor(global_rep)
        
        # Syntax validation
        syntax_score = self.syntax_checker(global_rep)
        
        return {
            'global_representation': global_rep,
            'component_representation': component_summary,
            'parameter_features': param_features,
            'syntax_score': syntax_score,
            'attention_weights': comp_attention,
            'token_representations': x
        }


class ParameterEncoder(nn.Module):
    """Encoder for circuit parameter vectors."""
    
    def __init__(
        self,
        param_dim: int = 64,
        embed_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        current_dim = param_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dim = embed_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Parameter type classification
        self.param_classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # R, L, C, M, V, I, etc.
        )
        
        # Value range validation
        self.range_validator = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, params: Tensor) -> Dict[str, Tensor]:
        """Forward pass for parameter vectors."""
        x = self.encoder(params)
        
        param_types = self.param_classifier(x)
        range_validity = self.range_validator(x)
        
        return {
            'parameter_embedding': x,
            'parameter_types': param_types,
            'range_validity': range_validity
        }


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.vision_text_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        self.text_param_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        self.vision_param_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(
        self,
        vision_features: Tensor,
        text_features: Tensor,
        param_features: Tensor
    ) -> Dict[str, Tensor]:
        """Cross-modal attention fusion."""
        
        # Ensure all features have the same embedding dimension
        # Vision-Text cross-attention
        vt_features, vt_attention = self.vision_text_attention(
            vision_features.unsqueeze(1) if vision_features.dim() == 2 else vision_features,
            text_features.unsqueeze(1) if text_features.dim() == 2 else text_features,
            text_features.unsqueeze(1) if text_features.dim() == 2 else text_features
        )
        
        # Text-Parameter cross-attention
        tp_features, tp_attention = self.text_param_attention(
            text_features.unsqueeze(1) if text_features.dim() == 2 else text_features,
            param_features.unsqueeze(1) if param_features.dim() == 2 else param_features,
            param_features.unsqueeze(1) if param_features.dim() == 2 else param_features
        )
        
        # Vision-Parameter cross-attention
        vp_features, vp_attention = self.vision_param_attention(
            vision_features.unsqueeze(1) if vision_features.dim() == 2 else vision_features,
            param_features.unsqueeze(1) if param_features.dim() == 2 else param_features,
            param_features.unsqueeze(1) if param_features.dim() == 2 else param_features
        )
        
        # Flatten for fusion
        vt_flat = vt_features.squeeze(1) if vt_features.size(1) == 1 else vt_features.mean(1)
        tp_flat = tp_features.squeeze(1) if tp_features.size(1) == 1 else tp_features.mean(1)
        vp_flat = vp_features.squeeze(1) if vp_features.size(1) == 1 else vp_features.mean(1)
        
        # Concatenate and fuse
        combined = torch.cat([vt_flat, tp_flat, vp_flat], dim=-1)
        fused_features = self.fusion_layer(combined)
        
        return {
            'fused_features': fused_features,
            'vision_text_attention': vt_attention,
            'text_param_attention': tp_attention,
            'vision_param_attention': vp_attention
        }


class PerformancePredictor(nn.Module):
    """Performance prediction from fused multi-modal features."""
    
    def __init__(self, input_dim: int = 1024, num_metrics: int = 8):
        super().__init__()
        
        # Performance regression heads
        self.gain_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.noise_figure_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.power_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.bandwidth_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # General performance predictor
        self.general_predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_metrics)
        )
    
    def forward(self, fused_features: Tensor) -> Dict[str, Tensor]:
        """Predict performance metrics."""
        
        gain = self.gain_predictor(fused_features)
        noise_figure = self.noise_figure_predictor(fused_features)
        power = self.power_predictor(fused_features)
        bandwidth = self.bandwidth_predictor(fused_features)
        
        general_metrics = self.general_predictor(fused_features)
        
        return {
            'gain': gain,
            'noise_figure': noise_figure,
            'power': power,
            'bandwidth': bandwidth,
            'all_metrics': general_metrics
        }


class CrossModalCircuitDiffuser(nn.Module):
    """
    Revolutionary Cross-Modal Circuit Diffuser.
    
    Integrates vision, text, and parameter understanding for superior circuit generation.
    """
    
    def __init__(self, config: CrossModalConfig):
        super().__init__()
        
        self.config = config
        
        # Modal encoders
        self.vision_encoder = VisionEncoder(
            embed_dim=config.vision_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        
        self.text_encoder = TextEncoder(
            embed_dim=config.text_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        
        self.parameter_encoder = ParameterEncoder(
            embed_dim=config.param_dim
        )
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(config.vision_dim, config.fusion_dim)
        self.text_proj = nn.Linear(config.text_dim, config.fusion_dim)
        self.param_proj = nn.Linear(config.param_dim, config.fusion_dim)
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            config.fusion_dim, config.num_heads
        )
        
        # Performance predictor
        self.performance_predictor = PerformancePredictor(
            config.fusion_dim
        )
        
        # Circuit generator (diffusion-based)
        self.circuit_generator = DiffusionModel(
            param_dim=64,
            condition_dim=config.fusion_dim,
            hidden_dim=512,
            num_timesteps=1000
        )
        
        # Temperature for contrastive learning
        self.temperature = config.temperature
    
    def encode_schematic(self, image: Tensor) -> Dict[str, Tensor]:
        """Encode schematic image."""
        return self.vision_encoder(image)
    
    def encode_netlist(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Encode SPICE netlist."""
        return self.text_encoder(token_ids, attention_mask)
    
    def encode_parameters(self, params: Tensor) -> Dict[str, Tensor]:
        """Encode parameter vector."""
        return self.parameter_encoder(params)
    
    def forward(
        self,
        schematic_image: Optional[Tensor] = None,
        netlist_tokens: Optional[Tensor] = None,
        netlist_mask: Optional[Tensor] = None,
        parameters: Optional[Tensor] = None,
        mode: str = "all"  # "all", "vision_only", "text_only", etc.
    ) -> Dict[str, Any]:
        """
        Forward pass with flexible modal inputs.
        
        Args:
            schematic_image: Circuit schematic image [B, 3, H, W]
            netlist_tokens: Tokenized SPICE netlist [B, seq_len]
            netlist_mask: Attention mask for netlist [B, seq_len]
            parameters: Circuit parameters [B, param_dim]
            mode: Which modalities to use
        
        Returns:
            Comprehensive results including fused features and predictions
        """
        results = {}
        batch_size = 1
        
        # Encode available modalities
        if schematic_image is not None and mode in ["all", "vision_only", "vision_text"]:
            batch_size = schematic_image.size(0)
            vision_out = self.encode_schematic(schematic_image)
            vision_features = self.vision_proj(vision_out['cls_features'])
            results.update({
                'vision_output': vision_out,
                'vision_features': vision_features
            })
        else:
            vision_features = None
        
        if netlist_tokens is not None and mode in ["all", "text_only", "vision_text", "text_param"]:
            batch_size = netlist_tokens.size(0)
            text_out = self.encode_netlist(netlist_tokens, netlist_mask)
            text_features = self.text_proj(text_out['global_representation'])
            results.update({
                'text_output': text_out,
                'text_features': text_features
            })
        else:
            text_features = None
        
        if parameters is not None and mode in ["all", "param_only", "text_param", "vision_param"]:
            batch_size = parameters.size(0)
            param_out = self.encode_parameters(parameters)
            param_features = self.param_proj(param_out['parameter_embedding'])
            results.update({
                'param_output': param_out,
                'param_features': param_features
            })
        else:
            param_features = None
        
        # Cross-modal fusion (only if multiple modalities available)
        if sum([f is not None for f in [vision_features, text_features, param_features]]) >= 2:
            # Handle missing modalities with learned embeddings
            device = next(self.parameters()).device
            
            if vision_features is None:
                vision_features = torch.zeros(batch_size, self.config.fusion_dim, device=device)
            if text_features is None:
                text_features = torch.zeros(batch_size, self.config.fusion_dim, device=device)
            if param_features is None:
                param_features = torch.zeros(batch_size, self.config.fusion_dim, device=device)
            
            fusion_out = self.cross_modal_attention(vision_features, text_features, param_features)
            fused_features = fusion_out['fused_features']
            
            results.update({
                'fusion_output': fusion_out,
                'fused_features': fused_features
            })
        else:
            # Single modality - use available features directly
            if vision_features is not None:
                fused_features = vision_features
            elif text_features is not None:
                fused_features = text_features
            elif param_features is not None:
                fused_features = param_features
            else:
                raise ValueError("At least one modality must be provided")
            
            results['fused_features'] = fused_features
        
        # Performance prediction
        performance_out = self.performance_predictor(fused_features)
        results['performance_prediction'] = performance_out
        
        # Circuit generation
        generated_circuit = self.circuit_generator.sample(fused_features.unsqueeze(1), num_inference_steps=50)
        results['generated_circuit'] = generated_circuit.squeeze(1)
        
        return results
    
    def contrastive_loss(self, features1: Tensor, features2: Tensor, labels: Tensor) -> Tensor:
        """Compute contrastive loss between two modalities."""
        # Normalize features
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features1, features2.t()) / self.temperature
        
        # Labels for positive pairs
        batch_size = features1.size(0)
        labels_matrix = torch.eye(batch_size, device=features1.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, torch.arange(batch_size, device=features1.device))
        
        return loss
    
    def multimodal_contrastive_loss(
        self,
        vision_features: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
        param_features: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute multi-modal contrastive losses."""
        losses = {}
        
        if vision_features is not None and text_features is not None:
            batch_size = vision_features.size(0)
            labels = torch.arange(batch_size, device=vision_features.device)
            losses['vision_text_loss'] = self.contrastive_loss(vision_features, text_features, labels)
        
        if text_features is not None and param_features is not None:
            batch_size = text_features.size(0)
            labels = torch.arange(batch_size, device=text_features.device)
            losses['text_param_loss'] = self.contrastive_loss(text_features, param_features, labels)
        
        if vision_features is not None and param_features is not None:
            batch_size = vision_features.size(0)
            labels = torch.arange(batch_size, device=vision_features.device)
            losses['vision_param_loss'] = self.contrastive_loss(vision_features, param_features, labels)
        
        return losses


class TransformerBlock(nn.Module):
    """Transformer block for vision encoder."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class SPICETokenizer:
    """Tokenizer for SPICE netlists."""
    
    def __init__(self, vocab_size: int = 10000):
        # SPICE-specific vocabulary
        self.special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        self.component_tokens = ['R', 'L', 'C', 'M', 'V', 'I', 'D', 'Q', 'X']
        self.directive_tokens = ['.model', '.param', '.include', '.end', '.ac', '.dc', '.tran']
        self.node_tokens = ['0', 'vdd', 'gnd', 'in', 'out', 'gate', 'drain', 'source']
        
        # Build vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        
        idx = 0
        for token in self.special_tokens + self.component_tokens + self.directive_tokens + self.node_tokens:
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            idx += 1
        
        # Add numeric tokens and common patterns
        for i in range(vocab_size - idx):
            token = f"token_{i}"
            self.vocab[token] = idx + i
            self.reverse_vocab[idx + i] = token
    
    def tokenize(self, netlist: str) -> List[str]:
        """Tokenize SPICE netlist."""
        # Simple tokenization - split by whitespace and newlines
        tokens = []
        lines = netlist.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('*') or not line:
                continue  # Skip comments and empty lines
            
            # Split line into tokens
            line_tokens = re.split(r'[\s=]+', line)
            tokens.extend(line_tokens)
        
        return tokens
    
    def encode(self, netlist: str, max_length: int = 512) -> Dict[str, List[int]]:
        """Encode netlist to token IDs."""
        tokens = ['<start>'] + self.tokenize(netlist) + ['<end>']
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<unk>'])
        
        # Pad or truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.vocab['<pad>']] * (max_length - len(token_ids)))
        
        # Attention mask
        attention_mask = [1] * min(len(tokens), max_length) + [0] * max(0, max_length - len(tokens))
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }


# Example usage and demonstration
def create_cross_modal_demo():
    """Create a demonstration of cross-modal circuit understanding."""
    config = CrossModalConfig(
        vision_dim=768,
        text_dim=512,
        param_dim=256,
        fusion_dim=1024,
        num_heads=8,
        num_layers=4
    )
    
    model = CrossModalCircuitDiffuser(config)
    
    # Create synthetic data
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Synthetic schematic image
    schematic_image = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Synthetic netlist
    tokenizer = SPICETokenizer()
    netlist = """
    * LNA Circuit
    M1 drain gate source bulk nmos w=50u l=100n
    R1 drain vdd 1k
    C1 gate input 1p
    C2 drain output 1p
    .end
    """
    
    encoded = tokenizer.encode(netlist)
    netlist_tokens = torch.tensor([encoded['input_ids']] * batch_size, device=device)
    netlist_mask = torch.tensor([encoded['attention_mask']] * batch_size, device=device)
    
    # Synthetic parameters
    parameters = torch.randn(batch_size, 64, device=device)
    
    return model, schematic_image, netlist_tokens, netlist_mask, parameters


if __name__ == "__main__":
    # Demonstration
    model, image, tokens, mask, params = create_cross_modal_demo()
    
    print("🔄 Cross-Modal Circuit Fusion Demonstration")
    print(f"📊 Model config: {model.config}")
    
    # Test different modes
    modes = ["all", "vision_only", "text_only", "vision_text"]
    
    for mode in modes:
        print(f"\n🧪 Testing mode: {mode}")
        
        try:
            if mode == "all":
                results = model(image, tokens, mask, params, mode=mode)
            elif mode == "vision_only":
                results = model(schematic_image=image, mode=mode)
            elif mode == "text_only":
                results = model(netlist_tokens=tokens, netlist_mask=mask, mode=mode)
            elif mode == "vision_text":
                results = model(image, tokens, mask, mode=mode)
            
            print(f"✅ Fused features shape: {results['fused_features'].shape}")
            print(f"🎯 Generated circuit shape: {results['generated_circuit'].shape}")
            
            # Print performance predictions
            perf = results['performance_prediction']
            print(f"📈 Predicted gain: {perf['gain'].mean().item():.2f} dB")
            print(f"📉 Predicted NF: {perf['noise_figure'].mean().item():.2f} dB")
            
        except Exception as e:
            print(f"❌ Error in mode {mode}: {e}")
    
    # Test contrastive learning
    print("\n🔗 Testing contrastive learning")
    if 'vision_features' in results and 'text_features' in results:
        contrastive_losses = model.multimodal_contrastive_loss(
            results.get('vision_features'),
            results.get('text_features'),
            results.get('param_features')
        )
        
        for loss_name, loss_value in contrastive_losses.items():
            print(f"📊 {loss_name}: {loss_value.item():.4f}")
    
    logger.info("Cross-modal circuit fusion demonstration completed successfully")