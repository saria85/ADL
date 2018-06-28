import pandas as pd

names = ['Clinic Number','Question Text','Answer Text','Answer Date','diagdate']
data = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/CIFunctionality-final-withdiagdate.csv',names=names)

data['year'] = data['Answer Date'].str.split('/').str[2]
data['month'] = data['Answer Date'].str.split('/').str[0]

data['diagyear'] = data['diagdate'].str.split('/').str[2]
data['diagmonth'] = data['diagdate'].str.split('/').str[0]

data['Answer Date'] = pd.to_datetime(data['Answer Date'],errors='coerce')
data['diagdate'] = pd.to_datetime(data['diagdate'],errors='coerce')

data = data.dropna()
# fs ='CI_CONCENTRATING'
# data = data[data['Question Text']==fs]

data['6month'] = (pd.np.ceil((data['diagdate']-pd.Timedelta(days=1)-data['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
data = data[(data['6month'] > 0) & (data['6month']<11)]
def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    #x['all']= len(x)
    return x

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df=data.groupby(['Clinic Number','diagyear','6month']).apply(func)
print(df.sort_values(['6month','Clinic Number']))
mask = df['yes']>1
#the rows with yes>1 should be converted to 1 according to the formula of dr sohn
df.loc[mask,'yes']=1
#df1['formula_applied']=(df1['yes'])

df = df.dropna()
df = df[['6month','yes','Question Text','Clinic Number']].drop_duplicates()
df['sum'] = (df.groupby(['6month','Question Text'])['yes']
                      .transform('sum'))
df['numPatients6month'] = (df.groupby(['6month'])['Clinic Number']
                      .transform('nunique'))
df['final-formula'] = (df['sum']/df['numPatients6month'])

# ######noCI part I know this way is dirty I have to show the result in 1 hour :D

names = ['Clinic Number','Question Text','Answer Text','Answer Date','date']
nocidata = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/noCIFunctionality-final-withdiagdate.csv',names=names)

nocidata['date'].fillna('NA')
nocidata['Answer Date'].fillna('NA')
nocidata=nocidata.dropna()

nocidata['year'] = nocidata['Answer Date'].str.split('/').str[2]
nocidata['month'] = nocidata['Answer Date'].str.split('/').str[0]

nocidata['diagyear'] = nocidata['date'].str.split('/').str[2]
nocidata['diagmonth'] = nocidata['date'].str.split('/').str[0]

nocidata['Answer Date'] = pd.to_datetime(nocidata['Answer Date'],errors='coerce')
nocidata['diagdate'] = pd.to_datetime(nocidata['date'],errors='coerce')
nocidata =nocidata.dropna()
#nocidata = nocidata[nocidata['Question Text']==fs]
nocidata['6month'] = (pd.np.ceil((nocidata['diagdate']-pd.Timedelta(days=1)-nocidata['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
nocidata = nocidata[(nocidata['6month'] > 0) & (nocidata['6month']<11)]
# def func(x):
#     x['yes']= len(x[x['Answer Text']=='Yes'])
#     x['all']= len(x)
#     return x

nocidf=nocidata.groupby(['Clinic Number','diagyear','6month']).apply(func)
mask = nocidf['yes']>1
#the rows with yes>1 should be converted to 1 according to the formula of dr sohn
nocidf.loc[mask,'yes']=1
#df1['formula_applied']=(df1['yes'])

nocidf = nocidf.dropna()
nocidf = nocidf[['6month','yes','Question Text','Clinic Number']].drop_duplicates()
nocidf['sum'] = (nocidf.groupby(['6month','Question Text'])['yes']
                      .transform('sum'))
nocidf['numPatients6month'] = (nocidf.groupby(['6month'])['Clinic Number']
                      .transform('nunique'))
nocidf['final-formula'] = (nocidf['sum']/nocidf['numPatients6month'])
nocidf = nocidf[nocidf['Question Text'] !='X']
df = df[df['Question Text'] !='X']
nocidffinal = nocidf[['6month','final-formula','numPatients6month']].drop_duplicates().sort_values(['6month'])
dffinal = df[['6month','final-formula','numPatients6month']].drop_duplicates().sort_values(['6month'])
print(dffinal.to_string())
print(nocidffinal.to_string())
dffinal['6month'].replace(1, '6 month',inplace=True)
dffinal['6month'].replace(2, '1 year',inplace=True)
dffinal['6month'].replace(3, '1.5 year',inplace=True)
dffinal['6month'].replace(4, '2 year',inplace=True)
dffinal['6month'].replace(5, '2.5 year',inplace=True)
dffinal['6month'].replace(6, '3 year',inplace=True)
dffinal['6month'].replace(7, '3.5 year',inplace=True)
dffinal['6month'].replace(8, '4 year',inplace=True)
dffinal['6month'].replace(9, '4.5 year',inplace=True)
dffinal['6month'].replace(10, '5 year',inplace=True)

nocidffinal['6month'].replace(1, '6 month',inplace=True)
nocidffinal['6month'].replace(2, '1 year',inplace=True)
nocidffinal['6month'].replace(3, '1.5 year',inplace=True)
nocidffinal['6month'].replace(4, '2 year',inplace=True)
nocidffinal['6month'].replace(5, '2.5 year',inplace=True)
nocidffinal['6month'].replace(6, '3 year',inplace=True)
nocidffinal['6month'].replace(7, '3.5 year',inplace=True)
nocidffinal['6month'].replace(8, '4 year',inplace=True)
nocidffinal['6month'].replace(9, '4.5 year',inplace=True)
nocidffinal['6month'].replace(10, '5 year',inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['axes.prop_cycle'] = ("cycler('color', 'rg')")

dffinal['CI-noCI']='Cognitive Impairement'
nocidffinal['CI-noCI']='Non Cognitive Impairement'
res=pd.concat([dffinal,nocidffinal])
# This will merge columns in order of the chart
vals = pd.concat([nocidffinal['numPatients6month'], dffinal['numPatients6month']], axis=1)
print(vals)
vals = vals.stack().reset_index(level=[0,1], drop=True)
print(vals)
# Plot the chart
ax = sns.barplot(x='6month', y='final-formula', data=res, hue='CI-noCI')
print(ax)
_ = plt.xticks(fontsize=8, rotation=45)

# Add the values on top of each correct bar
for idx, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            height + height*.01,
            vals[idx])
plt.savefig('C:/Users/M193053/PycharmProjects/ADL-distribution/binary_evaluation/binary-files-plot/newFunctionality-mental/binary-newFunctionality-CI-noCI.png')
plt.show()





