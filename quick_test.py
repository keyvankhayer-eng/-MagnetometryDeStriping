import pandas as pd
from src.line_leveling import hooshang_corr4

df = pd.read_excel('example/sample_magnetic_data.xlsx')
df_corr = hooshang_corr4(df)
df_corr.to_excel('example/output_CORR4.xlsx', index=False)
print(df_corr.head())
