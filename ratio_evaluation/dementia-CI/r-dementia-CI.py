import pandas as pd

names = ["Clinic Number","Question Text","Answer Text","Answer Date",'CIdate','dementiadate']
data = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/dementia-CIwithdate.csv',names=names)
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

data['year'] = data['Answer Date'].str.split('/').str[2]
data['month'] = data['Answer Date'].str.split('/').str[0]

data['Answer Date'] = pd.to_datetime(data['Answer Date'])
data['CIdate'] = pd.to_datetime(data['CIdate'],errors='coerce')
data['dementiadate'] = pd.to_datetime(data['dementiadate'],errors='coerce')
data = data.dropna()


data = data[data['Question Text']==fs]
data = data[data['Answer Date']>data['CIdate']]
data['6month'] = (pd.np.ceil((data['dementiadate']-pd.Timedelta(days=1)-data['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
data = data[((data['6month'] > 0) & (data['6month']<13))]

def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    x['all']= len(x)
    return x

df=data.groupby(['Clinic Number','6month']).apply(func)
df['formula_applied']=(df['yes']/df['all'])

df = df.dropna().sort_values(['Clinic Number','Answer Date'])
df['sum'] = (df.groupby(['6month'])['formula_applied']
                      .transform('sum'))
df['numPatients6month'] = (df.groupby(['6month'])['Clinic Number']
                      .transform('count'))
df['final-formula'] = (df['sum']/df['numPatients6month'])
import seaborn as sns
import matplotlib.pyplot as plt

dffinal = df[['6month','final-formula','numPatients6month']].drop_duplicates().sort_values(['6month'])
plt.rcParams['axes.prop_cycle'] = ("cycler('color', 'rg')")
dffinal['CI-noCI']='Cognitive Impairement'

sns.barplot(x='6month',y='final-formula',data=dffinal,hue='CI-noCI').set_title(fs)
plt.xticks(range(0,12), ['-6month','-1 year','-1.5 year','-2 year','-2.5 year','-3 year','-3.5 year','-4 year','-4.5 year','-5 year','-5.5 year','-6 year'], fontsize=8, rotation=45)
plt.savefig('C:/Users/M193053/PycharmProjects/ADL-distribution/ratio_evaluation/ratio-files-plot/dementia-CI/Ratio-dementia-CI-'+fs+'.png')
plt.show()
