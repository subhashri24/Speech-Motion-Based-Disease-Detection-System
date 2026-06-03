import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Healthy", "Parkinson", "Alzheimer", "Essential Tremor"]

def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(7,4))
    counts = df["label"].value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in counts.index]
    ax.bar(labels, counts.values)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig

def plot_histogram(df, col):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(df[col].dropna(), bins=25)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def plot_box(df, col):
    fig, ax = plt.subplots(figsize=(8,4))
    groups = [df[df["label"] == i][col].dropna() for i in sorted(df["label"].unique())]
    ax.boxplot(groups, tick_labels=CLASS_NAMES[:len(groups)])
    ax.set_title(f"{col} by Class")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig

def plot_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12,8))
    cax = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(cax)
    fig.tight_layout()
    return fig

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm)
    fig.colorbar(im)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    return fig

def plot_ovr_roc(y_true, y_proba):
    y_bin = label_binarize(y_true, classes=[0,1,2,3])
    fig, ax = plt.subplots(figsize=(7,5))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def plot_ovr_pr(y_true, y_proba):
    y_bin = label_binarize(y_true, classes=[0,1,2,3])
    fig, ax = plt.subplots(figsize=(7,5))
    for i, name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def plot_feature_importance(feature_names, importances, top_n=15):
    idx = np.argsort(importances)[::-1][:top_n]
    names = np.array(feature_names)[idx]
    vals = np.array(importances)[idx]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(names[::-1], vals[::-1])
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig

def plot_metric_bars(metric_dict):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    fig, ax = plt.subplots(figsize=(8,4))
    vals = [metric_dict[m] for m in metrics]
    ax.bar(metrics, vals)
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Evaluation Metrics")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig
