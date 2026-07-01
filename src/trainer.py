"""
trainer.py
==========
Entraînement universel pour modèles PyTorch avec monitoring complet :
- Affichage détaillé par epoch (comme demandé)
- Early stopping intelligent basé sur F1 macro
- Génération automatique des courbes d'apprentissage
- Sauvegarde des meilleurs modèles + historique d'entraînement
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import json
import os
from pathlib import Path

# Configuration du style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 30,
    patience: int = 4,
    save_dir: str = "models/trained",
    model_name: str = "model",
    scheduler=None,
    verbose: bool = True
):
    """
    Entraîne un modèle PyTorch avec monitoring complet et early stopping.
    
    Args:
        model: Modèle PyTorch à entraîner
        train_loader: DataLoader pour l'entraînement
        val_loader: DataLoader pour la validation
        criterion: Fonction de perte
        optimizer: Optimiseur PyTorch
        device: Device ('cuda' ou 'cpu')
        num_epochs: Nombre maximum d'epochs
        patience: Patience pour l'early stopping (basé sur F1 macro val)
        save_dir: Dossier de sauvegarde des résultats
        model_name: Nom du modèle (pour les fichiers de sortie)
        scheduler: Scheduler d'apprentissage optionnel (ex: ReduceLROnPlateau)
        verbose: Afficher les logs en temps réel
    
    Returns:
        dict: Historique d'entraînement contenant les métriques par epoch
        str: Chemin du meilleur modèle sauvegardé
    """
    # Créer le dossier de sauvegarde
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialisation
    model = model.to(device)
    best_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
        'lr': [], 'epochs': []
    }
    
    if verbose:
        print("\n" + "="*80)
        print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT : {model_name}")
        print("="*80)
        print(f"{'Epoch':<6} | {'Train Loss':<12} {'Train F1':<10} {'Val Loss':<12} "
              f"{'Val F1':<10} {'Val Acc':<10} | {'Time':<6} | {'Status':<12}")
        print("-"*80)
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Entraînement
        train_loss, train_f1 = _train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Évaluation
        val_loss, val_f1, val_acc, _, _ = _evaluate_epoch(model, val_loader, criterion, device)
        
        # Learning rate courant
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Mise à jour du scheduler si présent
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()
        
        # Enregistrement de l'historique
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        elapsed = time.time() - start_time
        
        # Détection du meilleur modèle
        status = "→"
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            status = "★ BEST"
            
            # Sauvegarder le meilleur modèle
            model_path = save_path / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_f1': val_f1,
                'history': history
            }, model_path)
            
            # Sauvegarder l'historique complet
            with open(save_path / "training_history.json", 'w') as f:
                json.dump(history, f, indent=2)
        else:
            patience_counter += 1
        
        # Affichage formaté
        if verbose:
            print(f"{epoch+1:6d} | {train_loss:<12.4f} {train_f1:<10.4f} {val_loss:<12.4f} "
                  f"{val_f1:<10.4f} {val_acc:<10.4f} | {elapsed:<6.0f}s | {status:<12}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\n🛑 Early stopping déclenché à l'epoch {epoch+1} (patience={patience})")
            break
    
    if verbose:
        print("="*80)
        print(f"✅ ENTRAÎNEMENT TERMINÉ - Meilleur F1 Macro : {best_f1:.4f} (epoch {len(history['val_f1']) - patience_counter})")
        print(f"💾 Modèle sauvegardé : {save_path / 'best_model.pth'}")
        print(f"📊 Courbes générées : {save_path / 'training_curves.png'}")
        print("="*80)
    
    # Générer les courbes d'entraînement
    _plot_training_curves(history, save_path, model_name)
    
    return history, str(save_path / "best_model.pth")


def _train_epoch(model, dataloader, optimizer, criterion, device):
    """Entraîne le modèle sur une epoch et retourne loss + F1 macro."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stabilité numérique
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1


def _evaluate_epoch(model, dataloader, criterion, device):
    """Évalue le modèle et retourne loss, F1 macro, accuracy + prédictions."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    
    return avg_loss, f1_macro, acc, all_preds, all_labels


def _plot_training_curves(history, save_path, model_name):
    """Génère et sauvegarde les courbes d'entraînement professionnelles."""
    epochs = history['epochs']
    
    # Créer la figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Entraînement - {model_name}', fontsize=16, fontweight='bold')
    
    # Courbe de Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_title('Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Courbe de F1 Macro
    ax2.plot(epochs, history['train_f1'], 'b-o', label='Train F1', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_f1'], 'r-o', label='Val F1', linewidth=2, markersize=4)
    ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Objectif F1=0.7')
    ax2.set_title('F1 Macro', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Courbe Learning Rate (si applicable)
    if 'lr' in history and len(history['lr']) > 1:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history['lr'], 'purple', marker='o', linewidth=2)
        plt.title(f'Learning Rate - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / "learning_rate.png", dpi=150, bbox_inches='tight')
        plt.close()


# ======================
# UTILITAIRE : Chargement du meilleur modèle
# ======================
def load_best_model(model_class, model_path, device, **model_kwargs):
    """
    Charge le meilleur modèle sauvegardé avec ses poids.
    
    Args:
        model_class: Classe du modèle (ex: SentimentLSTMAttention)
        model_path: Chemin vers best_model.pth
        device: Device PyTorch
        **model_kwargs: Arguments pour initialiser le modèle
    
    Returns:
        model: Modèle chargé et prêt pour l'inférence
        history: Historique d'entraînement
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialiser le modèle
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('history', None)




# ============================================================
# FONCTIONS SPÉCIALISÉES POUR ROBERTA (sans modifier train_model)
# ============================================================

def train_roberta(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=4,
    patience=2,
    save_path='best_model_roberta.pth',
    model_name='RoBERTa-base',
    verbose=True
):
    """
    Fine-tuning RoBERTa avec le format d'affichage exact de votre code original.
    
    Returns:
        dict: Historique d'entraînement {'train_loss', 'train_f1', 'val_loss', 'val_f1', 'val_acc'}
        str: Chemin du meilleur modèle sauvegardé
    """
    from sklearn.metrics import f1_score
    import time
    import torch
    
    best_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [], 'val_acc': []
    }
    
    if verbose:
        print("\n" + "="*60)
        print(f"DÉBUT DU FINE-TUNING - {model_name}")
        print("="*60)
        print(f"⚙️  Configuration : LR={optimizer.param_groups[0]['lr']:.0e} | "
              f"Batch={train_loader.batch_size} | Epochs={num_epochs}")
        print(f"⚠️  Texte BRUT utilisé (pas de prétraitement custom)\n")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Entraînement
        train_loss, train_f1 = _train_epoch_roberta(model, train_loader, optimizer, scheduler, device)
        
        # Évaluation
        val_loss, val_f1_macro, val_f1_weighted, val_acc, _, _ = _evaluate_roberta(model, val_loader, device)
        
        elapsed = time.time() - start_time
        
        # Enregistrement historique
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1_macro)
        history['val_acc'].append(val_acc)
        
        # Affichage formaté IDENTIQUE à votre code original
        if verbose:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val F1 (macro): {val_f1_macro:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Time: {elapsed:.0f}s")
        
        # Early stopping + sauvegarde
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"   → ✅ Meilleur modèle sauvegardé (F1 macro: {best_f1:.4f})")
        else:
            patience_counter += 1
            if verbose:
                print(f"   → ⏳ Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                if verbose:
                    print(f"\n🛑 Early stopping déclenché à l'epoch {epoch+1}")
                break
    
    if verbose:
        print("\n" + "="*60)
        print(f"RÉSULTATS FINAUX - {model_name}")
        print("="*60)
        print(f"F1 Macro      : {best_f1:.4f}")
        print(f"Meilleur modèle sauvegardé : '{save_path}'")
        print("="*60 + "\n")
    
    return history, save_path


def _train_epoch_roberta(model, dataloader, optimizer, scheduler, device):
    """Entraînement d'une epoch RoBERTa (interne, non appelée directement)."""
    from sklearn.metrics import f1_score
    import torch
    
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1


def _evaluate_roberta(model, dataloader, device):
    """Évaluation RoBERTa (interne, non appelée directement)."""
    from sklearn.metrics import f1_score, accuracy_score
    import torch
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1_macro, f1_weighted, acc, all_preds, all_labels