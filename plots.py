import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm




def bootstrap_and_plot(model, n_iterations, X_test_l1_selected, y_test, small=False):
    n_size = len(y_test)

    tprs = []
    aucs_roc = []
    precisions = []
    aucs_pr = []

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall_levels = np.linspace(0, 1, 100)

    # for _ in range(n_iterations):
    for _ in tqdm(range(n_iterations)):

        X_resample, y_resample = resample(X_test_l1_selected, y_test, n_samples=n_size)
        y_prob_resample = model.predict_proba(X_resample)[:, 1]

        fpr, tpr, _ = roc_curve(y_resample, y_prob_resample)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs_roc.append(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(y_resample, y_prob_resample)
        precisions.append(np.interp(mean_recall_levels, recall[::-1], precision[::-1]))
        aucs_pr.append(auc(recall, precision))

    def confidence_interval(data, confidence=0.95):
        std = np.std(data, axis=0)
        mean = np.mean(data, axis=0)
        n = len(data)
        width = std * 1.96 / np.sqrt(n)
        return mean - width, mean + width

    mean_auroc = np.mean(aucs_roc)
    mean_auprc = np.mean(aucs_pr)

    auroc_ci = np.percentile(aucs_roc, [2.5, 97.5])
    auprc_ci = np.percentile(aucs_pr, [2.5, 97.5])

 
    if small:
        figsize = (8, 3.5)  # Half the width when 'small' is True
    else:
        figsize = (16, 7)  # Default size
    
    plt.figure(figsize=figsize)

    # Plot AUROC Curve
    plt.subplot(1, 2, 1)
    for i in range(n_iterations):
        plt.plot(mean_fpr, tprs[i], color='lightgrey', lw=0.5, alpha=0.1)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.plot(mean_fpr, np.mean(tprs, axis=0), color='blue', lw=2, label=f'Mean AUROC: {mean_auroc:.4f}')
    plt.fill_between(mean_fpr, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0), color='grey', alpha=0.2, label=f'95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.grid(which='major', linestyle='-', linewidth=.5, color='gray')
    plt.grid(which='minor', linestyle='-', linewidth=0.1, color='gray')
    plt.tick_params(which='minor', length=0)

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for i in range(n_iterations):
        plt.plot(mean_recall_levels, precisions[i], color='lightgrey', lw=0.5, alpha=0.1)
    plt.plot(mean_recall_levels, np.mean(precisions, axis=0), color='blue', lw=2, label=f'Mean AUPRC: {mean_auprc:.4f}')
    plt.fill_between(mean_recall_levels, np.percentile(precisions, 2.5, axis=0), np.percentile(precisions, 97.5, axis=0), color='grey', alpha=0.2, label=f'95% CI: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.grid(which='major', linestyle='-', linewidth=.5, color='gray')
    plt.grid(which='minor', linestyle='-', linewidth=0.1, color='gray')
    plt.tick_params(which='minor', length=0)

    plt.tight_layout()
    plt.show()

# Usage example
# bootstrap_and_plot(model, n_iterations, X_test_l1_selected, y_test)

def plot_top_feature_importances(model, n_features, small=False):
    # Get feature importance of type 'gain'
    feature_importance = model.get_booster().get_score(importance_type='gain')

    # Take the top n features by gain
    top_features = [k for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:n_features]]
    top_gains = [feature_importance[f] for f in top_features]

    # Sort the top n features and their gains in descending order
    top_features.reverse()
    top_gains.reverse()

    # Plotting the top n feature importances as horizontal bars in descending order
    # Adjust plot size based on 'small' parameter
    if small:
        figsize = (5, 5)  # Half the width when 'small' is True
    else:
        figsize = (10, 20)  # Default size
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_gains)), top_gains, color='skyblue')
    plt.yticks(ticks=range(len(top_features)), labels=top_features)
    plt.xlabel('Gain')
    plt.ylabel('Features')
    plt.title(f'Top {n_features} Feature Importance by Gain')
    plt.show()

# Usage example
# plot_top_feature_importances(model, 20)



def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.show()

def plot_roc_pr_curves(y_test, y_prob, small=False):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    if small:
        figsize = (7, 3)  # Half the width when 'small' is True
    else:
        figsize = (10, 4)  # Default size
    
    # plt.figure(figsize=figsize)
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve plot
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")

    # Precision-Recall Curve plot
    ax2.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.4f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")

    plt.show()

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm

def bootstrap_and_plot_results(model, n_iterations, X_test_l1_selected, y_test, small=False):
    n_size = len(y_test)

    tprs = []
    aucs_roc = []
    precision_curves = []
    aucs_pr = []
    f1_scores = {"0": [], "1": []}
    recall_scores = {"0": [], "1": []}
    precision_scores = {"0": [], "1": []}
    specificities = {"0": [], "1": []}

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall_levels = np.linspace(0, 1, 100)

    for _ in tqdm(range(n_iterations)):
        X_resample, y_resample = resample(X_test_l1_selected, y_test, n_samples=n_size)
        y_pred_resample = model.predict(X_resample)
        y_prob_resample = model.predict_proba(X_resample)[:, 1]

        fpr, tpr, _ = roc_curve(y_resample, y_prob_resample)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs_roc.append(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(y_resample, y_prob_resample)
        precision_curves.append(np.interp(mean_recall_levels, recall[::-1], precision[::-1]))
        aucs_pr.append(auc(recall, precision))

        tn, fp, fn, tp = confusion_matrix(y_resample, y_pred_resample).ravel()

        f1_scores["0"].append(f1_score(y_resample, y_pred_resample, pos_label=0))
        f1_scores["1"].append(f1_score(y_resample, y_pred_resample, pos_label=1))

        recall_scores["0"].append(recall_score(y_resample, y_pred_resample, pos_label=0))
        recall_scores["1"].append(recall_score(y_resample, y_pred_resample, pos_label=1))

        precision_scores["0"].append(precision_score(y_resample, y_pred_resample, pos_label=0))
        precision_scores["1"].append(precision_score(y_resample, y_pred_resample, pos_label=1))

        specificity_0 = tn / (tn + fp) if tn + fp > 0 else 0
        specificity_1 = tp / (tp + fn) if tp + fn > 0 else 0
        specificities["0"].append(specificity_0)
        specificities["1"].append(specificity_1)

    mean_auroc = np.mean(aucs_roc)
    mean_auprc = np.mean(aucs_pr)
    auroc_ci = np.percentile(aucs_roc, [2.5, 97.5])
    auprc_ci = np.percentile(aucs_pr, [2.5, 97.5])

    mean_metrics = {
        "f1_scores": {label: np.mean(f1_scores[label]) for label in ["0", "1"]},
        "recall_scores": {label: np.mean(recall_scores[label]) for label in ["0", "1"]},
        "precision_scores": {label: np.mean(precision_scores[label]) for label in ["0", "1"]},
        "specificities": {label: np.mean(specificities[label]) for label in ["0", "1"]}
    }

    ci_metrics = {
        "f1_scores": {label: np.percentile(f1_scores[label], [2.5, 97.5]) for label in ["0", "1"]},
        "recall_scores": {label: np.percentile(recall_scores[label], [2.5, 97.5]) for label in ["0", "1"]},
        "precision_scores": {label: np.percentile(precision_scores[label], [2.5, 97.5]) for label in ["0", "1"]},
        "specificities": {label: np.percentile(specificities[label], [2.5, 97.5]) for label in ["0", "1"]}
    }

    if small:
        figsize = (8, 3.5)
    else:
        figsize = (16, 7)
    
    plt.figure(figsize=figsize)

    # Plot AUROC Curve
    plt.subplot(1, 2, 1)
    for i in range(n_iterations):
        plt.plot(mean_fpr, tprs[i], color='lightgrey', lw=0.5, alpha=0.1)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.plot(mean_fpr, np.mean(tprs, axis=0), color='blue', lw=2, label=f'Mean AUROC: {mean_auroc:.4f}')
    plt.fill_between(mean_fpr, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0), color='grey', alpha=0.2, label=f'95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.grid(which='major', linestyle='-', linewidth=.5, color='gray')
    plt.grid(which='minor', linestyle='-', linewidth=0.1, color='gray')
    plt.tick_params(which='minor', length=0)

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for i in range(n_iterations):
        plt.plot(mean_recall_levels, precision_curves[i], color='lightgrey', lw=0.5, alpha=0.1)
    plt.plot(mean_recall_levels, np.mean(precision_curves, axis=0), color='blue', lw=2, label=f'Mean AUPRC: {mean_auprc:.4f}')
    plt.fill_between(mean_recall_levels, np.percentile(precision_curves, 2.5, axis=0), np.percentile(precision_curves, 97.5, axis=0), color='grey', alpha=0.2, label=f'95% CI: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.grid(which='major', linestyle='-', linewidth=.5, color='gray')
    plt.grid(which='minor', linestyle='-', linewidth=0.1, color='gray')
    plt.tick_params(which='minor', length=0)

    plt.tight_layout()
    plt.show()

    return {
        "mean_auroc": mean_auroc,
        "auroc_ci": auroc_ci,
        "mean_auprc": mean_auprc,
        "auprc_ci": auprc_ci,
        "mean_fpr": mean_fpr,
        "mean_tpr": np.mean(tprs, axis=0),
        "mean_recall_levels": mean_recall_levels,
        "mean_precision": np.mean(precision_curves, axis=0),
        "mean_metrics": mean_metrics,
        "ci_metrics": ci_metrics
    }

# Use the function in your code by passing the required arguments.
# results = bootstrap_and_plot(model, n_iterations, X_test_l1_selected, y_test, small=False)
# This will plot the graphs and also return the relevant values, including confidence intervals for F1 score, recall, precision, and specificity.
