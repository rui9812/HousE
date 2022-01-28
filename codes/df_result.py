import torch
import optuna
import pandas as pd


study = optuna.create_study(
    study_name='example-study',
    storage='sqlite:///KGE-HousD-FB15K-Adam-1000-XXXXXX.db',
    load_if_exists=True,
    direction='maximize'
)
# optuna.delete_study('example-study', 'sqlite:///KGE.db')
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df)

df = df.sort_values(by="value" , ascending=False)
print(df)

writer = pd.ExcelWriter('test.xlsx')
df.to_excel(writer, sheet_name='data')
writer.save()
writer.close()
