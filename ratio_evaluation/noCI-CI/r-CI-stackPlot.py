import pandas as pd

names = ["Clinic Number","Question Text","Answer Text","Answer Date",'diagdate']
data = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/CIdiagjoin.csv',names=names)

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

data['6month'] = (pd.np.ceil((data['diagdate']-pd.Timedelta(days=1)-data['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
data = data[(data['6month'] > 0) & (data['6month']<11)]
def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    x['all']= len(x)
    return x

df=data.groupby(['Clinic Number','diagyear','6month','Question Text']).apply(func)
df['formula_applied']=(df['yes']/df['all'])

df = df.dropna().sort_values(['Clinic Number','Answer Date'])
df['sum'] = (df.groupby(['6month','Question Text'])['formula_applied']
                      .transform('sum'))
df['numPatients6month'] = (df.groupby(['6month'])['Clinic Number']
                      .transform('count'))
df['final-formula'] = (df['sum']/df['numPatients6month'])
dffinal = df[['6month','final-formula','Question Text','numPatients6month']].drop_duplicates().sort_values(['6month'])
print(dffinal.to_string())
df = dffinal.groupby(['6month','Question Text']).sum().unstack('Question Text')

# df.columns = df.columns.droplevel()
# print(df.to_string())
ax=df.plot(kind='bar', stacked=True)
import matplotlib.pyplot as plt

plt.xticks(range(0,10), ['6month','1 year','1.5 year','2 year','2.5 year','3 year','3.5 year','4 year','4.5 year','5 year'], fontsize=8, rotation=45)
# dffinal.plot(x='6month', y='numPatients6month',visible=False)
plt.title('Cognitive Impairement-Stack bar')

# This will merge columns in order of the chart
vals =  dffinal['numPatients6month']
#vals = vals.stack().reset_index(level=[0,1], drop=True)

for idx, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + height*.01,
            vals[idx],
            ha="center")
plt.savefig('C:/Users/M193053/PycharmProjects/ADL-distribution/ratio_evaluation/ratio-files-plot/noCI-CI/stack-Ratio-CI.png')
plt.show()

