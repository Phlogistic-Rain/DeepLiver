# DeepLiver

## Data Preparation

Please download the original image dataset and the processed feature files from [Zenodo](link).

Then place the feature files in the `./features` directory.

## Inference

Run the inference script:

```bash
python inference.py
```

## Expected Results

| Metric    | Mean   | Std    |
|-----------|--------|--------|
| Accuracy  | 0.9542 | 0.0078 |
| Precision | 0.9414 | 0.0096 |
| Recall    | 0.9433 | 0.0096 |
| F1        | 0.9415 | 0.0099 |
| MCC       | 0.9473 | 0.0089 |
