import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pddate = pd.read_csv('CIdiag.csv')

pd1 = pd.read_csv('CIwithdateFunctionality2.csv')
print(pd1.to_string())
pd2=pd1.join(pddate.set_index('Clinic Number'),on='Clinic Number')
print(pd2.to_string())
pd3 =pd.read_csv('Function-CIwithdatefinal.csv')
pd2['note date'] =pd.to_datetime(pd2['note date'])
pd2['diagdate'] =pd.to_datetime(pd2['diagdate'])
common = pd.merge(pd2,pd3,how='inner',on=['Clinic Number'])
uncommondf = pd2[~pd2['Clinic Number'].isin(common['Clinic Number'])]
uncommondf.insert(1,'norm', 'X')
uncommondf.insert(4,'Answer Text', 'No')
frame = [uncommondf,pd3]
finaldf = pd.concat(frame)
print(finaldf.to_string())
finaldf.to_csv('CIFunctionality-final-withdiagdate.csv')
#pd2=pd2.drop_duplicates(['Clinic Number','norm','note date','date','Answer Text'])
#pd2.to_csv('CIFunctionality-final-wi2thdiagdate.csv')
# pddateci = pd.read_csv('Function-CIwithdatefinal.csv')
#
# pd1ci = pd.read_csv('CIwithdateFunctionality2.csv')
#
# pd2ci=pd1ci.join(pddateci.set_index('Clinic Number'),on='Clinic Number')
# pd2ci=pd2ci.drop_duplicates(['Clinic Number','norm','note date','diagdate','Answer Text'])
#
# pd2ci.to_csv('noCIFunctionality-final-withanswer.csv')
# #CIwithdateFunctionality.csv
# #print(pd2.to_string())
# #ci =2544
# #noci = 2611
