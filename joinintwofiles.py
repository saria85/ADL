import pandas as pd

pddate = pd.read_csv('Function-noCIwithdatefinal.csv')

pd1 = pd.read_csv('noCIwithdateFunctionality2.csv')

pd2=pd1.join(pddate.set_index('Clinic Number'),on='Clinic Number')
pd2=pd2.drop_duplicates(['Clinic Number','norm','note date','date','Answer Text'])
pd2.to_csv('noCIFunctionality-final-withanswer.csv')
pddateci = pd.read_csv('Function-CIwithdatefinal.csv')

pd1ci = pd.read_csv('CIwithdateFunctionality2.csv')

pd2ci=pd1ci.join(pddateci.set_index('Clinic Number'),on='Clinic Number')
pd2ci=pd2ci.drop_duplicates(['Clinic Number','norm','note date','diagdate','Answer Text'])

pd2ci.to_csv('noCIFunctionality-final-withanswer.csv')
#CIwithdateFunctionality.csv
#print(pd2.to_string())
#ci =2544
#noci = 2611
