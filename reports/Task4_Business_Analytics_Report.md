# Task 4: Business Sales and Customer Analytics

## Dataset and methodology

- Source: `data/raw/business_sales_customers.csv`.
- Records: 30,000; columns: 41.
- Duplicate full rows: 0.
- Missing cells across the dataset: 0.
- Revenue metric: sum of `Order_Amount`, grouped by the dataset's Year and Month fields.
- Profit metric: sum of `Profit_Amount`, grouped by `Product_Category`.
- Currency and units were not independently verified; monetary figures are reported in source units.

## Findings

- Total recorded order amount: 11,370,043.99.
- Average recorded order amount: 379.00.
- Median recorded order amount: 171.03.
- Total recorded profit amount: 2,215,916.27.
- Monthly revenue series contains 48 month groups with valid year/month fields.
- Unique customers according to `Customer_ID`: 8,683.
- Highest recorded revenue category: Electronics (6,320,059.31).
- Highest recorded total profit category: Electronics (1,040,688.00).

## Business recommendations

- Monitor monthly revenue and profit separately; do not use discount amount as a proxy for revenue.
- Compare category revenue with category profit before allocating promotions.
- Validate currency, units, return handling, order statuses and duplicate order identifiers before operational decisions.
- Treat the results as descriptive analytics, not proof that a category or promotion caused better performance.
- Validate any forecast on a chronological holdout period before using it for planning.

## Outputs

- `outputs/task4_monthly_revenue.csv`
- `outputs/task4_monthly_revenue.png`
- `outputs/task4_sales_by_category_summary.csv`
- `outputs/task4_profit_by_category_summary.csv`
- `outputs/task4_revenue_by_category.png`
- `outputs/task4_profit_by_category.png`
