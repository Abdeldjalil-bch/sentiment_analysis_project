"""
evaluate.py
===========
Évaluation complète des modèles avec métriques avancées et visualisations :
- Matrice de confusion (heatmap professionnelle)
- Courbes ROC multiclasse (micro/macro + par classe)
- Rapport de classification détaillé
- Analyse des erreurs par classe
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score, 
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize  # Pour ROC multiclasse


import json 
import pandas as pd  

import os
from pathlib import Path

# Configuration du style
sns.set_style("white")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
# Configuration du style
sns.set_style("white")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list = ['Négatif', 'Neutre', 'Positif'],
    save_dir: str = "reports/figures",
    model_name: str = "model",
    plot_roc: bool = True,
    plot_confusion: bool = True
):
    """
    Évalue un modèle PyTorch et génère des visualisations professionnelles.
    
    Args:
        model: Modèle PyTorch entraîné
        dataloader: DataLoader de test
        device: Device PyTorch
        class_names: Noms des classes pour les labels
        save_dir: Dossier de sauvegarde des figures
        model_name: Nom du modèle (pour les fichiers de sortie)
        plot_roc: Générer les courbes ROC (True par défaut)
        plot_confusion: Générer la matrice de confusion (True par défaut)
    
    Returns:
        dict: Dictionnaire complet des métriques
        dict: Prédictions et labels bruts pour analyse ultérieure
    """
    # Créer le dossier de sauvegarde
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Obtenir les prédictions
    all_preds, all_labels, all_probs = _get_predictions(model, dataloader, device, len(class_names))
    
    # Calculer les métriques principales
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'f1_per_class': f1_score(all_labels, all_preds, average=None).tolist(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    # Calculer le ROC AUC multiclasse
    if len(class_names) == 2:
        # Binaire : ROC AUC simple
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        # Multiclasse : ROC AUC micro/macro
        metrics['roc_auc_micro'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='micro')
        metrics['roc_auc_macro'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    
    # Afficher le rapport de classification
    print("\n" + "="*70)
    print(f"📊 ÉVALUATION DU MODÈLE : {model_name}")
    print("="*70)
    print(f"\nAccuracy      : {metrics['accuracy']:.4f}")
    print(f"F1 Macro      : {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted   : {metrics['f1_weighted']:.4f}")
    
    if len(class_names) == 2:
        print(f"ROC AUC       : {metrics['roc_auc']:.4f}")
    else:
        print(f"ROC AUC Micro : {metrics['roc_auc_micro']:.4f}")
        print(f"ROC AUC Macro : {metrics['roc_auc_macro']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Générer les visualisations
    if plot_confusion:
        _plot_confusion_matrix(all_labels, all_preds, class_names, save_path, model_name)
    
    if plot_roc and len(class_names) > 1:
        _plot_roc_curves(all_labels, all_probs, class_names, save_path, model_name)
    
    # Sauvegarder les métriques
    _save_metrics(metrics, save_path, model_name)
    
    # Retourner les résultats
    results = {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    print(f"\n✅ Évaluation terminée - Résultats sauvegardés dans : {save_path}")
    print("="*70)
    
    return results


def _get_predictions(model, dataloader, device, num_classes):
    """Obtient les prédictions, labels et probabilités (compatible RoBERTa + modèles custom)."""
    import torch
    import numpy as np
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # ✅ GESTION UNIVERSELLE : extraire les logits selon le type de sortie
            outputs = model(input_ids, attention_mask)
            
            # Cas 1 : Modèle Hugging Face (RoBERTa) → outputs.logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            # Cas 2 : Modèle custom (BiLSTM, CNN) → outputs est déjà un tenseur
            else:
                logits = outputs
            
            # Calcul des probabilités et prédictions
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def _plot_confusion_matrix(labels, preds, class_names, save_path, model_name):
    """Génère une heatmap professionnelle de la matrice de confusion."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12}, cbar_kws={'label': 'Pourcentage (%)'})
    
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Prédiction', fontsize=13, fontweight='bold')
    plt.ylabel('Vérité Terrain', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Version textuelle pour le rapport
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(save_path / "confusion_matrix.csv")
    print(f"   → Matrice de confusion sauvegardée : confusion_matrix.png/csv")


def _plot_roc_curves(labels, probs, class_names, save_path, model_name):
    """Génère les courbes ROC multiclasse avec micro/macro moyennes (corrigé pour multiclasse)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    
    # Binariser les labels pour le calcul ROC multiclasse
    y_bin = label_binarize(labels, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calculer ROC pour chaque classe (one-vs-rest)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculer micro moyenne (tous les échantillons considérés comme un seul problème binaire)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calculer macro moyenne (moyenne non pondérée des courbes par classe)
    # Interpoler toutes les courbes aux mêmes points FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Interpoler les TPR pour chaque classe
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Moyenne sur les classes
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Créer la figure
    plt.figure(figsize=(10, 8))
    
    # Couleurs pour les classes
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:0.3f})')
    
    # Micro moyenne
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro moyenne (AUC = {roc_auc["micro"]:0.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    # Macro moyenne
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro moyenne (AUC = {roc_auc["macro"]:0.3f})',
             color='navy', linestyle=':', linewidth=3)
    
    # Ligne de hasard
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Aléatoire (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs', fontsize=13, fontweight='bold')
    plt.ylabel('Taux de Vrais Positifs', fontsize=13, fontweight='bold')
    plt.title(f'Courbes ROC Multiclasse - {model_name}', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "roc_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"   → Courbes ROC sauvegardées : roc_curves.png")

def _save_metrics(metrics, save_path, model_name):
    """Sauvegarde les métriques dans un fichier JSON lisible et un résumé textuel."""
    import json  # Sécurité supplémentaire (au cas où l'import global échoue)
    
    # Nettoyer les valeurs numpy pour la sérialisation JSON
    metrics_clean = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_clean[k] = v.tolist()
        elif isinstance(v, np.generic):
            metrics_clean[k] = v.item()
        elif isinstance(v, (int, float, str, list, dict)):
            metrics_clean[k] = v
        else:
            metrics_clean[k] = str(v)  # Conversion de secours
    
    # Sauvegarder en JSON
    try:
        with open(save_path / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde JSON : {e}")
    
    # Créer un résumé textuel pour le rapport
    try:
        with open(save_path / "metrics_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"RÉSUMÉ DES MÉTRIQUES - {model_name}\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy      : {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Macro      : {metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Weighted   : {metrics['f1_weighted']:.4f}\n")
            
            if 'roc_auc' in metrics:
                f.write(f"ROC AUC       : {metrics['roc_auc']:.4f}\n")
            elif 'roc_auc_micro' in metrics and 'roc_auc_macro' in metrics:
                f.write(f"ROC AUC Micro : {metrics['roc_auc_micro']:.4f}\n")
                f.write(f"ROC AUC Macro : {metrics['roc_auc_macro']:.4f}\n")
            
            f.write("\nF1 par classe :\n")
            class_names = ['Négatif', 'Neutre', 'Positif']
            for i, (name, f1) in enumerate(zip(class_names, metrics['f1_per_class'])):
                f.write(f"  {name:12s}: {f1:.4f}\n")
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde résumé textuel : {e}")

# ======================
# UTILITAIRE : Comparaison de modèles
# ======================
def compare_models(results_dict, save_dir="reports/figures/comparison"):
    """
    Compare plusieurs modèles et génère un tableau de métriques côte à côte.
    
    Args:
        results_dict: Dict {nom_modèle: résultats_evaluate_model}
        save_dir: Dossier de sauvegarde
    
    Exemple d'utilisation:
        results = {
            "LSTM": evaluate_model(model1, ...),
            "RoBERTa": evaluate_model(model2, ...),
        }
        compare_models(results)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Préparer les données pour le tableau
    metrics_list = []
    for model_name, results in results_dict.items():
        metrics = results['metrics']
        metrics_list.append({
            'Modèle': model_name,
            'Accuracy': metrics['accuracy'],
            'F1 Macro': metrics['f1_macro'],
            'F1 Weighted': metrics['f1_weighted'],
            'ROC AUC': metrics.get('roc_auc', metrics.get('roc_auc_macro', 'N/A'))
        })
    
    # Créer le DataFrame
    df = pd.DataFrame(metrics_list)
    df = df.sort_values('F1 Macro', ascending=False)
    
    # Sauvegarder en CSV
    df.to_csv(save_path / "model_comparison.csv", index=False)
    
    # Créer un tableau visuel
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(
        cellText=df.values.round(4),
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style du tableau
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorer la meilleure ligne
    best_idx = df['F1 Macro'].idxmax()
    for j in range(len(df.columns)):
        table[(best_idx + 1, j)].set_facecolor('#E7E6E6')
    
    plt.title('Comparaison des Modèles - Métriques Clés', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path / "model_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Comparaison des modèles sauvegardée :")
    print(f"   - Tableau CSV : {save_path / 'model_comparison.csv'}")
    print(f"   - Visualisation : {save_path / 'model_comparison.png'}")
    print("\n" + df.to_string(index=False))