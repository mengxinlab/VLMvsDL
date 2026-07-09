# Metadata-control summary

## AUC

| condition     |   n |      auc |    ci_lo |    ci_hi |
|:--------------|----:|---------:|---------:|---------:|
| F2 image-only | 917 | 0.699465 | 0.639409 | 0.75432  |
| F3 image+text | 917 | 0.739419 | 0.683709 | 0.799317 |
| F0 text-only  | 917 | 0.69268  | 0.633753 | 0.74615  |

## Paired DeLong

| a             | b             |   n |    auc_a |    auc_b |   delta_a_minus_b |         p |
|:--------------|:--------------|----:|---------:|---------:|------------------:|----------:|
| F3 image+text | F0 text-only  | 917 | 0.739419 | 0.69268  |         0.0467393 | 0.0165084 |
| F3 image+text | F2 image-only | 917 | 0.739419 | 0.699465 |         0.0399536 | 0.340695  |

## Association audit

| condition     | feature_space   |   n_diameter |   spearman_diameter |   spearman_diameter_p |   n_brock |   spearman_brock |   spearman_brock_p |   mean_score_spiculated_or_irregular |   mean_score_other_margin |   n_spiculated_or_irregular |   n_other_margin |
|:--------------|:----------------|-------------:|--------------------:|----------------------:|----------:|-----------------:|-------------------:|-------------------------------------:|--------------------------:|----------------------------:|-----------------:|
| F2 image-only | true_metadata   |          701 |            0.165194 |           1.10169e-05 |       701 |         0.146769 |       9.61473e-05  |                             0.416265 |                  0.429315 |                         166 |              540 |
| F3 image+text | true_metadata   |          701 |            0.569189 |           1.97165e-61 |       701 |         0.619486 |       1.59714e-75  |                             0.558012 |                  0.208037 |                         166 |              540 |
| F0 text-only  | true_metadata   |          701 |            0.661405 |           2.17642e-89 |       701 |         0.722767 |       2.85516e-114 |                             0.658253 |                  0.225074 |                         166 |              540 |
