import os

import pandas as pd
df_list = pd.read_csv('biobank_65up_NoCI_random_new.txt',names=['Clinic Number'])
#print(df_list.head())

path = '/infodev1/phi-data/sohn/biobank/biobank_65up_CI_dx_CN'
files = os.listdir(path)

df = pd.DataFrame(files, columns=['file_name'])
df['Clinic Number'] = (df.file_name.str[1:8]).astype(int)
df['date'] = df.file_name.str.extract('(\d{4}-\d{2}-\d{2})', expand=True)

sorted = df.sort_values(by='date')
#result = sorted.drop_duplicates('Clinic Number', keep='all')

#dffinal = df_list.join(sorted.set_index('Clinic Number'),on='Clinic Number')
sorted.to_csv('CIwithdateFunctionality2.csv')
# print(dffinal)
#
# pd1 = pd.read_csv('/infodev1/phi-data/sohn/biobank/ADL/ADL_noCI.txt')
#
# pd2=pd1.join(dffinal.set_index('Clinic Number'),on='Clinic Number')
# pd2.to_csv('noCIwithdate.csv')
# print(pd2.head())
#
