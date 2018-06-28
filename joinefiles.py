import pandas as pd

#namefunc =['Clinic Number','Question Text','Answer Text','Answer Date','diagdate']
pdfunc = pd.read_csv('noCIFunctionality-final-withdiagdate.csv')

#nameadl = ['Clinic Number','Question Text','Answer Text','Answer Date','diagdate']
pdadl = pd.read_csv('noCIwithdatefinal.csv')

frame = [pdadl,pdfunc]
result = pd.concat(frame)

result.to_csv('noCIgrouping.csv')


