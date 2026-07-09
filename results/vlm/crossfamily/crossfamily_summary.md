# Cross-family metadata-reliance summary

## Metadata lift = AUC(image+text) - AUC(image-only)

| family                            |   n_paired |   auc_image_only |   auc_image_text |   metadata_lift |   lift_ci_lo |   lift_ci_hi |   lift_delong_p |   auc_text_only |   text_only_vs_imgtext_delong_p |   recovery_frac |
|:----------------------------------|-----------:|-----------------:|-----------------:|----------------:|-------------:|-------------:|----------------:|----------------:|--------------------------------:|----------------:|
| Gemini-3-Flash (ref)              |        917 |         0.682269 |         0.729591 |       0.0473226 |   -0.0284866 |     0.121979 |     0.230236    |      nan        |                      nan        |      nan        |
| anthropic_claude-opus-4-8@default |        917 |         0.496699 |         0.72965  |       0.232951  |    0.172051  |     0.29313  |     6.12843e-14 |        0.718833 |                        0.266074 |        0.953564 |
| google_gemini-3-flash-preview     |        917 |         0.508078 |         0.722422 |       0.214344  |    0.126258  |     0.299886 |     1.96347e-06 |        0.72062  |                        0.896108 |        0.991595 |
| google_gemini-3.1-pro-preview     |        917 |         0.515506 |         0.728971 |       0.213465  |    0.12532   |     0.294821 |     8.8735e-07  |        0.712594 |                        0.1636   |        0.923279 |
| openai_gpt-5.5-2026-04-23         |        917 |         0.5      |         0.711523 |       0.211523  |    0.152079  |     0.273837 |     2.86782e-11 |        0.702965 |                        0.224548 |        0.959542 |

## Per-condition AUC

| family                            | condition   |   n |      auc |    ci_lo |    ci_hi |
|:----------------------------------|:------------|----:|---------:|---------:|---------:|
| Gemini-3-Flash (ref)              | image-only  | 917 | 0.682269 | 0.614272 | 0.752073 |
| Gemini-3-Flash (ref)              | image-text  | 917 | 0.729591 | 0.670373 | 0.793225 |
| anthropic_claude-opus-4-8@default | image-only  | 917 | 0.496699 | 0.474393 | 0.523988 |
| anthropic_claude-opus-4-8@default | image-text  | 917 | 0.72965  | 0.668669 | 0.790787 |
| anthropic_claude-opus-4-8@default | text-only   | 917 | 0.718833 | 0.657545 | 0.77437  |
| google_gemini-3-flash-preview     | image-only  | 917 | 0.508078 | 0.444311 | 0.562055 |
| google_gemini-3-flash-preview     | image-text  | 917 | 0.722422 | 0.655292 | 0.77832  |
| google_gemini-3-flash-preview     | text-only   | 917 | 0.72062  | 0.656524 | 0.779956 |
| google_gemini-3.1-pro-preview     | image-only  | 917 | 0.515506 | 0.454502 | 0.57556  |
| google_gemini-3.1-pro-preview     | image-text  | 917 | 0.728971 | 0.668716 | 0.785503 |
| google_gemini-3.1-pro-preview     | text-only   | 917 | 0.712594 | 0.655148 | 0.767183 |
| openai_gpt-5.5-2026-04-23         | image-only  | 917 | 0.5      | 0.5      | 0.5      |
| openai_gpt-5.5-2026-04-23         | image-text  | 917 | 0.711523 | 0.646279 | 0.76988  |
| openai_gpt-5.5-2026-04-23         | text-only   | 917 | 0.702965 | 0.640038 | 0.762999 |

## Score-vs-structured-predictor association (Spearman)

| family                            | condition   |   spearman_diameter |   n_diameter |   spearman_brock |   n_brock |
|:----------------------------------|:------------|--------------------:|-------------:|-----------------:|----------:|
| Gemini-3-Flash (ref)              | image-only  |         0.190404    |          701 |        0.181427  |       701 |
| Gemini-3-Flash (ref)              | image-text  |         0.513303    |          701 |        0.637065  |       701 |
| anthropic_claude-opus-4-8@default | image-only  |        -0.000524091 |          701 |       -0.0134474 |       701 |
| anthropic_claude-opus-4-8@default | image-text  |         0.751348    |          701 |        0.843488  |       701 |
| anthropic_claude-opus-4-8@default | text-only   |         0.715091    |          701 |        0.824398  |       701 |
| google_gemini-3-flash-preview     | image-only  |        -0.0159624   |          701 |       -0.0139652 |       701 |
| google_gemini-3-flash-preview     | image-text  |         0.652043    |          701 |        0.767733  |       701 |
| google_gemini-3-flash-preview     | text-only   |         0.685691    |          701 |        0.796739  |       701 |
| google_gemini-3.1-pro-preview     | image-only  |         0.0605397   |          701 |        0.0632175 |       701 |
| google_gemini-3.1-pro-preview     | image-text  |         0.733219    |          701 |        0.78862   |       701 |
| google_gemini-3.1-pro-preview     | text-only   |         0.741662    |          701 |        0.801198  |       701 |
| openai_gpt-5.5-2026-04-23         | image-only  |       nan           |          701 |      nan         |       701 |
| openai_gpt-5.5-2026-04-23         | image-text  |         0.702042    |          701 |        0.823213  |       701 |
| openai_gpt-5.5-2026-04-23         | text-only   |         0.704479    |          701 |        0.82192   |       701 |
