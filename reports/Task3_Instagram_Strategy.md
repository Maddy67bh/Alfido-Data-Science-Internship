# Task 3: Instagram Engagement Analysis

## Dataset and methodology

- Source: locally available CSV in `data/raw/`.
- Findings describe this dataset only; they are not automatically generalizable.

## Findings

- Rows: 29,999; columns: 15.
- Duplicate rows: 0.
- Cells missing values: 0.
- engagement_rate: mean 14.42; median 9.19; valid rows 29,999.
- likes: mean 99,912.66; median 99,580.00; valid rows 29,999.
- comments: mean 5,017.78; median 5,042.00; valid rows 29,999.
- shares: mean 2,502.91; median 2,498.00; valid rows 29,999.
- saves: mean 7,490.12; median 7,483.00; valid rows 29,999.
- reach: mean 996,561.60; median 992,181.00; valid rows 29,999.
- impressions: mean 1,246,668.83; median 1,241,026.00; valid rows 29,999.
- followers_gained: mean 502.15; median 501.00; valid rows 29,999.
- Highest average engagement rate by content category: Beauty (15.664).
- Highest average engagement rate by media type: Video (14.620).

## Outputs

Charts and supporting CSV files are saved in `outputs/`.

## Posting-time limitation

The current hourly summary contains only one observed hour (9) across 29,999 records. This does not provide a meaningful comparison of posting times. Verify the source timestamps and test multiple hours before recommending an optimal posting time.
