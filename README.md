# DeepLiver

## Data Preparation

Please download the original image dataset and the processed feature files from [Zenodo](https://zenodo.org/records/18479640?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5N2FmZDJmLWQxM2YtNDhiMy05NzE0LTZjYzNiNmI1MzAzZCIsImRhdGEiOnt9LCJyYW5kb20iOiJmYWMyNmExOTZkOTRkZmVjY2M5ODIwM2FhNTk2N2NkNyJ9.JVNYoxYZWrKEX74V5yMZXNKy5Dg1dFTCa1ZYcN12hIoDxxTYN0ciazrgMYAyu_ZAWbdHtmfz1Yn3v_AxkNFNxQ).

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
