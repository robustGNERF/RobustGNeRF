import pandas as pd

# Read the CSV data
df = pd.read_csv('/disk1/chanho/3d/MetaFool/eval/summarized_results.csv')

# Extract components from the 'Model' column
df[['TrainingMethod', 'EvaluationMethod', 'NearbyNumber']] = df['Model'].str.split('_', expand=True).iloc[:, 1:4]

# If 'NearbyNumber' is NaN, set it to 1
df['NearbyNumber'].fillna('1', inplace=True)

# Group by the extracted components and calculate the mean only for numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
grouped = df.groupby(['TrainingMethod', 'EvaluationMethod', 'NearbyNumber'])[numeric_columns].mean().reset_index()

# Create a 'Model' column for the averaged data
grouped['Model'] = 'AVG_' + grouped['TrainingMethod'] + '_' + grouped['EvaluationMethod'] + '_' + grouped['NearbyNumber']

# Append the average data to the main DataFrame
df_avg = grouped[['Model'] + list(numeric_columns)]
df = pd.concat([df, df_avg], sort=False)

# Convert the augmented DataFrame to a LaTeX table
latex_table = df.to_latex(index=False, escape=False, column_format='|c'*len(df.columns) + '|')

# Add table caption and label
latex_table = latex_table.replace("\\end{tabular}",
                                  "\\hline\n\\end{tabular}\n\\caption{Summary of Results with Averages}\n\\label{tab:results_with_avg}")

# Print the LaTeX table
print(latex_table)
