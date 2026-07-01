"""
src/__init__.py
================
Package principal pour l'analyse de sentiment.
Permet des imports simplifiés de tous les modules clés.
"""

# Import des fonctions utilitaires et prétraitement
from .preprocessing import (
    preprocess_text,
    preprocess_full_dataset,
    expand_contractions,
    expand_slang,
    replace_emojis,
    preprocess_full_dataset_lemmatized
)

# Import des fonctions de dataset et dataloaders
from .dataset import (
    create_dataloaders
)

# Import des fonctions d'entraînement
from .trainer import (
    train_model,
    train_roberta
)

# Import des fonctions d'évaluation
from .evaluate import (
    evaluate_model,
    compare_models
)

# Import des architectures de modèles
from .models import (
    SentimentLSTM,
    BiLSTMAttention,
    SentimentCNN,
    HybridCNNBiLSTMAttention,
    SentimentTransformer
)


# Définir les symboles exportés par le package
__all__ = [
    # Prétraitement
    'preprocess_text', 'preprocess_full_dataset', 
    'expand_contractions', 'expand_slang', 'replace_emojis',
    'preprocess_full_dataset_lemmatized',
    # Dataset
    'create_dataloaders',  
    
    # Entraînement
    'train_model', 'train_roberta',  
    
    # Évaluation
    'evaluate_model', 'compare_models',  
    
    # Modèles
    'SentimentLSTM', 'BiLSTMAttention', 
    'SentimentCNN', 'HybridCNNBiLSTMAttention', 
    'TransformerEncoder', 'SentimentTransformer',
 
]

# Version du package
__version__ = "1.0.0"