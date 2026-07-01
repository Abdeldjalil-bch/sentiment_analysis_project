"""
src/models.py
==============
Architectures de modèles pour l'analyse de sentiment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
#  MODEL BIDIRECTIONAL LSTM
# ======================

class SentimentLSTM(nn.Module):
    """LSTM bidirectionnelle optimisée pour sentiment analysis (3 classes)"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, 
                 num_layers=2, output_dim=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 pour bidirectionnel
    
    def forward(self, input_ids, attention_mask=None):
        embedded = self.dropout(self.embedding(input_ids))
        
        # Gestion du padding avec pack_padded_sequence (optionnel mais recommandé)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(packed)
            # hidden shape: [num_layers * 2, batch, hidden_dim]
        else:
            lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Concaténer les états cachés finaux des deux directions
        hidden_forward = hidden[-2, :, :]  # dernière couche, direction forward
        hidden_backward = hidden[-1, :, :] # dernière couche, direction backward
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        return self.fc(self.dropout(hidden_concat))
    
# ======================
# BILSTM + ATTENTION ARCHITECTURE
# ======================

class BiLSTMAttention(nn.Module):
    """
    BiLSTM bidirectionnelle avec couche d'attention additive.
    → Contrairement à une LSTM unidirectionnelle, la BiLSTM capture le contexte 
      passé ET futur pour chaque mot (ex: "not" influence les mots suivants ET précédents).
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, 
                 num_layers=2, output_dim=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM : bidirectional=True → 2 directions (forward + backward)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # ← CRITIQUE : active la bidirectionnalité
        )
        
        # Attention sur les 2 directions concaténées (d'où hidden_dim * 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, input_ids, attention_mask):
        # Embedding
        embedded = self.dropout(self.embedding(input_ids))  # [batch, seq, emb_dim]
        
        # BiLSTM
        bilstm_out, _ = self.bilstm(embedded)  # [batch, seq, hidden_dim*2]
        
        # Attention additive améliorée
        attn_scores = self.attention(bilstm_out).squeeze(-1)  # [batch, seq]
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch, 1, seq]
        
        # Context vector pondéré
        context_vector = torch.bmm(attn_weights, bilstm_out).squeeze(1)  # [batch, hidden*2]
        
        # Classification
        return self.fc(self.dropout(context_vector))


# ======================
# CNN 1D ARCHITECTURE - Multi-filtres
# ======================
class SentimentCNN(nn.Module):
    """
    CNN 1D avec filtres multi-tailles (3,4,5) pour détecter les motifs locaux :
    → Filtre 3 : bigrammes/trigrammes critiques ("not good", "very bad")
    → Filtre 4 : expressions courtes ("love this", "hate it")
    → Filtre 5 : phrases courtes ("this is great")
    
    Avantage vs BiLSTM : 
    - Plus rapide à entraîner
    - Excellent pour les motifs locaux (négations, intensifieurs)
    - Moins sensible au bruit de longue portée
    """
    def __init__(self, vocab_size, embedding_dim=256, n_filters=128, 
                 filter_sizes=[3, 4, 5], output_dim=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutions 1D avec différentes tailles de filtres
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim, 
                out_channels=n_filters, 
                kernel_size=fs
            )
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        # Initialisation des poids (Xavier pour stabilité)
        self._init_weights()
    
    def _init_weights(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding : [batch, seq_len] → [batch, seq_len, emb_dim]
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Permutation pour Conv1d : [batch, emb_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        # Appliquer chaque convolution + ReLU + Max pooling global
        conved = [
            F.relu(conv(embedded))  # [batch, n_filters, seq_len - kernel_size + 1]
            for conv in self.convs
        ]
        
        # Max pooling global sur la dimension temporelle
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)  # [batch, n_filters]
            for conv in conved
        ]
        
        # Concaténer les features de tous les filtres
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch, n_filters * len(filter_sizes)]
        
        # Classification
        return self.fc(cat)
    
# ======================
# HYBRID CNN + BiLSTM + ATTENTION
# ======================

class HybridCNNBiLSTMAttention(nn.Module):
    """
    Architecture hybride combinant :
    → CNN 1D : extraction de motifs locaux (n-grammes critiques comme "not_good")
    → BiLSTM : modélisation du contexte global bidirectionnel
    → Attention : pondération des représentations les plus discriminantes
    
    Stratégie de fusion PARALLÈLE (plus stable que séquentielle) :
    - Branche CNN : max pooling global → vecteur de features locales
    - Branche BiLSTM : attention → vecteur de contexte global
    - Concaténation → classification finale
    
    Pourquoi parallèle ?
    ✅ Évite la perte de séquence après max pooling CNN
    ✅ Les deux branches apprennent des représentations COMPLÉMENTAIRES
    ✅ Meilleure stabilité d'entraînement vs architecture séquentielle
    """
    def __init__(self, vocab_size, embedding_dim=256, n_filters=64, 
                 filter_sizes=[3, 4, 5], hidden_dim=128, num_layers=2, 
                 output_dim=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # ============ BRANCHE CNN ============
        # Convolutions 1D multi-tailles pour capturer les n-grammes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim, 
                out_channels=n_filters, 
                kernel_size=fs,
                padding=fs//2  # Padding pour préserver la longueur temporelle
            )
            for fs in filter_sizes
        ])
        
        # ============ BRANCHE BILSTM + ATTENTION ============
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention additive améliorée
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ============ FUSION ET CLASSIFICATION ============
        self.dropout = nn.Dropout(dropout)
        # Dimension totale = features CNN + features BiLSTM
        total_features = (n_filters * len(filter_sizes)) + (hidden_dim * 2)
        self.fc = nn.Linear(total_features, output_dim)
        
        # Initialisation Xavier pour stabilité
        self._init_weights()
    
    def _init_weights(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, input_ids, attention_mask):
        # Embedding partagé pour les deux branches
        embedded = self.embedding(input_ids)  # [batch, seq_len, emb_dim]
        embedded = self.dropout(embedded)
        
        # ========== BRANCHE CNN ==========
        # Permute pour Conv1d: [batch, emb_dim, seq_len]
        embedded_cnn = embedded.permute(0, 2, 1)
        
        # Appliquer convolutions + ReLU
        conved = [F.relu(conv(embedded_cnn)) for conv in self.convs]  # [batch, n_filters, seq_len]
        
        # Max pooling global sur chaque convolution
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [batch, n_filters]
        
        # Concaténer les features CNN
        cnn_features = torch.cat(pooled, dim=1)  # [batch, n_filters * len(filter_sizes)]
        cnn_features = self.dropout(cnn_features)
        
        # ========== BRANCHE BILSTM + ATTENTION ==========
        # BiLSTM sur les embeddings originaux
        bilstm_out, _ = self.bilstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # Calcul de l'attention
        attn_scores = self.attention(bilstm_out).squeeze(-1)  # [batch, seq_len]
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch, 1, seq_len]
        
        # Vecteur de contexte pondéré
        lstm_features = torch.bmm(attn_weights, bilstm_out).squeeze(1)  # [batch, hidden_dim*2]
        lstm_features = self.dropout(lstm_features)
        
        # ========== FUSION DES DEUX BRANCHES ==========
        # Concaténation des représentations complémentaires
        combined = torch.cat([cnn_features, lstm_features], dim=1)  # [batch, total_features]
        
        # Classification finale
        return self.fc(combined)
    

#-------------------------------------------------------------------------------------------------

# ======================
# SENTIMENT TRANSFORMER (ViT-style pour texte) 
# ======================

class SentimentTransformer(nn.Module):
    """
    Transformer Encoder avec token CLS pour classification de sentiment.
    CORRECTION CRITIQUE : utilisation de key_padding_mask au lieu de attn_mask
    pour éviter les erreurs de dimension avec nn.MultiheadAttention.
    """
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, 
                 mlp_dim=512, transformer_layers=4, max_length=60, 
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.max_length = max_length
        
        # Embedding de tokens + token CLS
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embeddings appris (pour séquence + CLS token)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length + 1, embed_dim))
        
        # Dropout régularisation
        self.dropout = nn.Dropout(dropout)
        
        # Couches Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(transformer_layers)
        ])
        
        # Tête de classification sur le token CLS
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialisation des poids (Xavier pour stabilité)
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier pour stabilité de convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - indices des tokens
            attention_mask: [batch_size, seq_len] - masque de padding (1=token réel, 0=padding)
        
        Returns:
            logits: [batch_size, num_classes] - scores de classification
        """
        B = input_ids.size(0)
        
        # Embedding des tokens
        x = self.token_embedding(input_ids)  # [B, seq_len, embed_dim]
        
        # Ajout du token CLS en position 0
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, seq_len+1, embed_dim]
        
        # Ajout des positional embeddings (tronquage si séquence trop longue)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        
        # 🔑 CORRECTION CRITIQUE : key_padding_mask pour nn.MultiheadAttention
        # Format: [batch_size, seq_len+1] avec True = token à masquer (padding)
        key_padding_mask = None
        if attention_mask is not None:
            # Étendre le masque pour inclure le token CLS (toujours actif = False)
            cls_mask = torch.zeros(B, 1, device=attention_mask.device, dtype=torch.bool)
            padding_mask = (attention_mask == 0)  # True où il y a du padding
            key_padding_mask = torch.cat((cls_mask, padding_mask), dim=1)  # [B, seq_len+1]
        
        # Passage dans les couches Transformer
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        
        # Classification sur le token CLS
        cls_output = x[:, 0]  # [B, embed_dim]
        logits = self.mlp_head(cls_output)  # [B, num_classes]
        
        return logits


class TransformerBlock(nn.Module):
    """Bloc Transformer standard avec pré-normalisation (Pre-LN) + gestion correcte du masque."""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        # Pré-normalisation (meilleure stabilité que Post-LN)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        
        # MLP avec GELU
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            key_padding_mask: [batch_size, seq_len] - booléen (True = masquer ce token)
        """
        # === Attention avec résidu (pré-normalisation) ===
        normed = self.layer_norm_1(x)
        
        # 🔑 CORRECTION : utiliser key_padding_mask (pas attn_mask) pour le padding
        attn_out, _ = self.attention(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,  # ✅ Format correct pour le padding
            attn_mask=None  # Pas de masque causal pour classification
        )
        x = x + attn_out
        
        # === MLP avec résidu (pré-normalisation) ===
        normed = self.layer_norm_2(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out
        
        return x

# ======================
# EXPORTS
# ======================
__all__ = [
    'SentimentLSTM',
    'BiLSTMAttention',
    'SentimentCNN',
    'HybridCNNBiLSTMAttention',
    'TransformerEncoder',
    'SentimentTransformer'
]