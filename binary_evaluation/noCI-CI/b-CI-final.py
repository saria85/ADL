import pandas as pd

names = ["Clinic Number","Question Text","Answer Text","Answer Date",'diagdate']
data = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/CIdiagjoin.csv',names=names)
fs = 'CI function'
def mapping():
    data['Question Text']=data['Question Text'].replace({'Do you have difficulty bathing by yourself?':'bathing'})
    data['Question Text']= data['Question Text'].replace({'Do you have difficulty dressing by yourself?':'dressing'})
    data['Question Text']= data['Question Text'].replace({'Do you have difficulty eating by yourself?':'feeding'})
    data['Question Text']= data['Question Text'].replace({'Do you have difficulty housekeeping by yourself?':'housekeeping'})
    data['Question Text']= data['Question Text'].replace({'Do you have difficulty taking medications by yourself?':'medications'})
    data['Question Text']= data['Question Text'].replace({'Do you have difficulty transportation by yourself?':'transportation'})
    data['Question Text']= data['Question Text'].replace({'Do you have difficulty using the toilet by yourself?':'toileting'})
    data['Question Text']= data['Question Text'].replace({'Do you have difficulty walking by yourself?':'transferring'})
    data['Question Text']= data['Question Text'].replace({'Do you have diffuclty getting in and out of bed by yourself?':'transferring'})
    data['Question Text']= data['Question Text'].replace({'Do you have diffuclty preparing meals by yourself?':'preparing food'})
    data['Question Text']= data['Question Text'].replace({'Could you do the grocery shopping on your own?':'shopping'})
    data['Question Text']= data['Question Text'].replace({'Trouble concentrating on things, such as reading the newspaper or watching television':'CI function'})
    data['Question Text']= data['Question Text'].replace({'Have you had difficulty concentrating?':'CI function'})
    data['Question Text']= data['Question Text'].replace({'Does the patient have - difficulty concentrating?':'CI function'})
    data['Question Text']= data['Question Text'].replace({'Walking':'transferring'})
    data['Question Text']= data['Question Text'].replace({'Moving or speaking so slowly that other people could have notices. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual':'CI function'})
    data['Answer Text']= data['Answer Text'].replace({'Not at all':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Several days':'Yes'})
    data['Answer Text']= data['Answer Text'].replace({'More than half the days':'Yes'})
    data['Answer Text']= data['Answer Text'].replace({'Nearly every day':'Yes'})
    data['Answer Text']= data['Answer Text'].replace({'No Trouble':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Yes Easy':'No'})
    data['Answer Text']= data['Answer Text'].replace({'LitTrouble':'No'})
    data['Answer Text']= data['Answer Text'].replace({'ModTrouble':'No'})
    data['Answer Text']= data['Answer Text'].replace({'SlightPrbm':'No'})
    data['Answer Text']= data['Answer Text'].replace({'No Problem':'Yes'})
    data['Answer Text']= data['Answer Text'].replace({'ModProblem':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Mod Diffic':'No'})
    data['Answer Text']= data['Answer Text'].replace({'VeryDiffic':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Lit Diffic':'No'})
    data['Answer Text']= data['Answer Text'].replace({'SeverPrblm':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Impossible':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Per Self':'Yes'})
    data['Answer Text']= data['Answer Text'].replace({'< 100Yards':'Yes'})
    data['Answer Text']= data['Answer Text'].replace({'9':'Yes'})
    data['Answer Text']= data['Answer Text'].replace({'Pain/Slow':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Some Help':'No'})
    data['Answer Text']= data['Answer Text'].replace({'Unable':'No'})
    data['Answer Text']= data['Answer Text'].replace({'NoImpact':'No'})
mapping()
data = data[~data['Question Text'].isin(["knee","USUAL ACTIVITIES (e.g. work, study, housework, family or leisure activities)","Moving or speaking so slowly that other people could have notices. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual","Personal care (washing, dressing, etc.)"])]
searchfor = ['knee','trouble getting in and out','Question Text']
data = data[data['Question Text'].str.contains('|'.join(searchfor))==False]
data['Answer Text'].fillna('NA')
#print(data.head(4))

data['year'] = data['Answer Date'].str.split('/').str[2]
data['month'] = data['Answer Date'].str.split('/').str[0]

data['diagyear'] = data['diagdate'].str.split('/').str[2]
data['diagmonth'] = data['diagdate'].str.split('/').str[0]

data['Answer Date'] = pd.to_datetime(data['Answer Date'])
data['diagdate'] = pd.to_datetime(data['diagdate'])

data = data[data['Question Text']==fs]

#data['6month'] = pd.date_range(end=data['diagnosisdate'],periods=2, freq='6M',closed='left')
data['6month'] = (pd.np.ceil((data['diagdate']-pd.Timedelta(days=1)-data['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
data = data[(data['6month'] > 0) & (data['6month']<11)]
#print(data.to_string())
def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    #x['all']= len(x)
    return x

df=data.groupby(['Clinic Number','diagyear','6month']).apply(func)
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

######noCI part I know this way is dirty I have to show the result in 1 hour :D

names = ["Clinic Number","Question Text","Answer Text","Answer Date",'date']
nocidata = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/noCIwithdatefinal.csv',names=names)

def mapping():
    nocidata['Question Text']=nocidata['Question Text'].replace({'Do you have difficulty bathing by yourself?':'bathing'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have difficulty dressing by yourself?':'dressing'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have difficulty eating by yourself?':'feeding'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have difficulty housekeeping by yourself?':'housekeeping'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have difficulty taking medications by yourself?':'medications'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have difficulty transportation by yourself?':'transportation'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have difficulty using the toilet by yourself?':'toileting'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have difficulty walking by yourself?':'transferring'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have diffuclty getting in and out of bed by yourself?':'transferring'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Do you have diffuclty preparing meals by yourself?':'preparing food'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Could you do the grocery shopping on your own?':'shopping'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Trouble concentrating on things, such as reading the newspaper or watching television':'CI function'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Have you had difficulty concentrating?':'CI function'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Does the patient have - difficulty concentrating?':'CI function'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Walking':'transferring'})
    nocidata['Question Text']= nocidata['Question Text'].replace({'Moving or speaking so slowly that other people could have notices. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual':'CI function'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Not at all':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Several days':'Yes'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'More than half the days':'Yes'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Nearly every day':'Yes'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'No Trouble':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Yes Easy':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'LitTrouble':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'ModTrouble':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'SlightPrbm':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'No Problem':'Yes'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'ModProblem':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Mod Diffic':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'VeryDiffic':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Lit Diffic':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'SeverPrblm':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Impossible':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Per Self':'Yes'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'< 100Yards':'Yes'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'9':'Yes'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Pain/Slow':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Some Help':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'Unable':'No'})
    nocidata['Answer Text']= nocidata['Answer Text'].replace({'NoImpact':'No'})
mapping()
nocidata = nocidata[~nocidata['Question Text'].isin(["knee","USUAL ACTIVITIES (e.g. work, study, housework, family or leisure activities)","Moving or speaking so slowly that other people could have notices. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual","Personal care (washing, dressing, etc.)"])]
searchfor = ['knee','trouble getting in and out','Question Text']
nocidata = nocidata[nocidata['Question Text'].str.contains('|'.join(searchfor))==False]
nocidata['Answer Text'].fillna('NA')
nocidata['date'].fillna('NA')
nocidata['Answer Date'].fillna('NA')
nocidata=nocidata.dropna()
#print(nocidata.head(4))

nocidata['year'] = nocidata['Answer Date'].str.split('/').str[2]
nocidata['month'] = nocidata['Answer Date'].str.split('/').str[0]

nocidata['diagyear'] = nocidata['date'].str.split('/').str[2]
nocidata['diagmonth'] = nocidata['date'].str.split('/').str[0]

nocidata['Answer Date'] = pd.to_datetime(nocidata['Answer Date'])
nocidata['diagdate'] = pd.to_datetime(nocidata['date'])


nocidata = nocidata[nocidata['Question Text']==fs]

#nocidata['6month'] = pd.date_range(end=nocidata['diagnosisdate'],periods=2, freq='6M',closed='left')
nocidata['6month'] = (pd.np.ceil((nocidata['diagdate']-pd.Timedelta(days=1)-nocidata['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
nocidata = nocidata[(nocidata['6month'] > 0) & (nocidata['6month']<11)]
#print(nocidata.to_string())
def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    #x['all']= len(x)
    return x

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

dffinal = df[['6month','final-formula','numPatients6month']].drop_duplicates().sort_values(['6month'])
nocidffinal = nocidf[['6month','final-formula','numPatients6month']].drop_duplicates().sort_values(['6month'])
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

dffinal[fs]='Cognitive Impairement'
nocidffinal[fs]='Non Cognitive Impairement'
res=pd.concat([dffinal,nocidffinal])
# This will merge columns in order of the chart
vals = pd.concat([nocidffinal['numPatients6month'], dffinal['numPatients6month']], axis=1)
vals = vals.stack().reset_index(level=[0,1], drop=True)

# Plot the chart
ax = sns.barplot(x='6month', y='final-formula', data=res, hue=fs)
_ = plt.xticks(fontsize=8, rotation=45)

# Add the values on top of each correct bar
for idx, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2,
            height + height*.01,
            vals[idx],
            ha="center")
plt.savefig('C:/Users/M193053/PycharmProjects/ADL-distribution/binary_evaluation/binary-files-plot/noCI-CI/binary-CI-noCI-'+fs+'.png')
plt.show()





