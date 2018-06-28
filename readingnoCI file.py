import os
import numpy as np
import pandas as pd
df = pd.read_csv('noCIwithdateFunctionality2.csv')
df2 = pd.read_csv('Function-noCIwithdatefinal.csv')
# path = 'C:/Users/M193053/Documents/fldr'
# files = os.listdir(path)
#
# df = pd.DataFrame(files, columns=['file_name'])
# df['Clinic Number'] = (df.file_name.str[1:8]).astype(int)
# df['latestdate'] = df.file_name.str.extract('(\d{4}-\d{2}-\d{2})', expand=True)
df['note date'] = pd.to_datetime(df['note date'])
df2['note date'] = pd.to_datetime(df2['note date'])
df2['date'] = pd.to_datetime(df2['date'])
sorted = df.sort_values(by=['Clinic Number','note date'])
sorted['date'] = df.groupby('Clinic Number')['note date'].transform('max')
common = pd.merge(sorted,df2,how='inner',on=['Clinic Number'])
uncommondf=sorted[~sorted['Clinic Number'].isin(common['Clinic Number'])]
uncommondf.insert(1,'norm', 'X')
uncommondf.insert(4,'Answer Text', 'No')
frame = [uncommondf,df2]
finaldf = pd.concat(frame)
#finaldf = pd.concat(uncommondf,df2)
# #
print(finaldf.to_string())
finaldf.to_csv('noCIFunctionality-final-withdiagdate')
# result = sorted.drop_duplicates('Clinic Number', keep='last')
#
# dffinal = df_list.join(result.set_index('Clinic Number'),on='Clinic Number')
# print(dffinal)
