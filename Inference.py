import os
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset_GCN import WSIFeatureDataset
from models_gcn.GCN_main_TB import TripleStreamGCN

CHECKPOINT_PATH_TEMPLATE = './models_log/{fold}/checkpoints/best_model.pth'

FEATURE_DIR_TEMPLATE = './features/tf_efficientnetv2_b0.in1k_ft/{fold}'

INFERENCE_SPLIT = 'Test'

NUM_FOLDS = 5

NUM_CLASSES = 8

NUM_WORKERS = 0
PIN_MEMORY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, device):
    model = TripleStreamGCN(
        in_channels=1280,
        hidden_channels=128,
        out_channels=128,
        momentum=0.3,
        momentum_decay=0.1,
        decay_steps=3,
        afm_threshold=0.75,
        beta=0.4,
        gamma=0.1,
        num_classes=NUM_CLASSES,
        tf_num_layers=2,
        tf_num_heads=4
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def inference_one_fold(model, data_loader, device):
    y_pred = []
    y_true = []
    score_list = []

    with torch.no_grad():
        for data, target, pos in tqdm(data_loader, desc="Inference"):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            data = data.squeeze(0)
            target = target.squeeze(0)

            output = model(data)

            score_list.extend(output.detach().cpu().numpy())

            _, pred = torch.max(output.detach(), 1)

            y_pred.extend(pred.view(-1).detach().cpu().numpy())
            y_true.extend(target.view(-1).detach().cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(score_list)


def calculate_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Inference split: {INFERENCE_SPLIT}")
    print(f"Number of folds: {NUM_FOLDS}")
    print("=" * 50)

    all_metrics = []
    all_y_true = []
    all_y_pred = []
    all_scores = []

    for fold in range(NUM_FOLDS):
        print(f"\n>>> Fold {fold}")

        checkpoint_path = CHECKPOINT_PATH_TEMPLATE.format(fold=fold)
        feature_dir = FEATURE_DIR_TEMPLATE.format(fold=fold)

        if not os.path.exists(checkpoint_path):
            print(f"Warning: checkpoint not found: {checkpoint_path}")
            print("Skipping this fold...")
            continue

        print(f"Loading model: {checkpoint_path}")
        model = load_model(checkpoint_path, DEVICE)

        dataset = WSIFeatureDataset(feature_dir=feature_dir, split=INFERENCE_SPLIT)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

        print(f"Number of samples: {len(dataset)}")

        y_true, y_pred, scores = inference_one_fold(model, data_loader, DEVICE)

        fold_metrics = calculate_metrics(y_true, y_pred)
        all_metrics.append(fold_metrics)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_scores.append(scores)

        print(f"Fold {fold} Results:")
        print(f"  Accuracy:  {fold_metrics['accuracy']:.4f}")
        print(f"  Precision: {fold_metrics['precision']:.4f}")
        print(f"  Recall:    {fold_metrics['recall']:.4f}")
        print(f"  F1:        {fold_metrics['f1']:.4f}")
        print(f"  MCC:       {fold_metrics['mcc']:.4f}")

    print("\n" + "=" * 50)
    print("Average Performance Across All Folds")
    print("=" * 50)

    if len(all_metrics) == 0:
        print("No valid fold results!")
        return

    df_metrics = pd.DataFrame(all_metrics)

    mean_metrics = df_metrics.mean()
    std_metrics = df_metrics.std()

    print(f"\n{'Metric':<12} {'Mean':>10} {'Std':>10}")
    print("-" * 35)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc']:
        print(f"{metric:<12} {mean_metrics[metric]:>10.4f} {std_metrics[metric]:>10.4f}")

    output_dir = './inference_results'
    os.makedirs(output_dir, exist_ok=True)

    df_metrics.index = [f'fold_{i}' for i in range(len(df_metrics))]
    df_metrics.loc['mean'] = mean_metrics
    df_metrics.loc['std'] = std_metrics

    csv_path = os.path.join(output_dir, f'metrics_{INFERENCE_SPLIT}.csv')
    df_metrics.to_csv(csv_path, index=True)
    print(f"\nMetrics saved to: {csv_path}")

    np.save(os.path.join(output_dir, f'y_true_{INFERENCE_SPLIT}.npy'), np.array(all_y_true, dtype=object))
    np.save(os.path.join(output_dir, f'y_pred_{INFERENCE_SPLIT}.npy'), np.array(all_y_pred, dtype=object))
    np.save(os.path.join(output_dir, f'scores_{INFERENCE_SPLIT}.npy'), np.array(all_scores, dtype=object))

    print("Detailed prediction results saved as npy files")
    print("\nInference completed!")


if __name__ == '__main__':
    main()
