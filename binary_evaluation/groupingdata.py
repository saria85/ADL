import pandas as pd

#names = ['Clinic Number','Answer Date','Question Text','date','Answer Text']
data = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/CIgrouping.csv')
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
def groupmapping():
    data['new'] = 'b-ADL'
    data.loc[data['Question Text']=='bathing','new']='b-ADL'
    data.loc[data['Question Text']=='dressing','new']='b-ADL'
    data.loc[data['Question Text']=='transferring','new']='b-ADL'
    data.loc[data['Question Text']=='toileting','new']='b-ADL'
    data.loc[data['Question Text']=='feeding','new']='b-ADL'
    data.loc[data['Question Text']=='transportation','new']='i-ADL'
    data.loc[data['Question Text']=='shopping','new']='i-ADL'
    data.loc[data['Question Text']=='preparing food','new']='i-ADL'
    data.loc[data['Question Text']=='housekeeping','new']='i-ADL'
    data.loc[data['Question Text']=='medications','new']='i-ADL'
    data.loc[data['Question Text']=='CI_FINANCE','new']='i-ADL'
    data.loc[data['Question Text']=='CI_CONCENTRATING','new']='MJO-Concentrating'
    data.loc[data['Question Text']=='CI function','new']='MJO-Concentrating'
    data.loc[data['Question Text']=='CI_MJO','new']='MJO-Concentrating'
groupmapping()

data = data[~data['Question Text'].isin(["knee","USUAL ACTIVITIES (e.g. work, study, housework, family or leisure activities)","Moving or speaking so slowly that other people could have notices. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual","Personal care (washing, dressing, etc.)"])]
searchfor = ['knee','trouble getting in and out','Question Text']
data = data[data['Question Text'].str.contains('|'.join(searchfor))==False]
#data['Answer Text'].fillna('NA')

data['year'] = data['Answer Date'].str.split('/').str[2]
data['month'] = data['Answer Date'].str.split('/').str[0]

data['diagyear'] = data['diagdate'].str.split('/').str[2]
data['diagmonth'] = data['diagdate'].str.split('/').str[0]

data['Answer Date'] = pd.to_datetime(data['Answer Date'])
data['diagdate'] = pd.to_datetime(data['diagdate'])

data =data.dropna()
data['6month'] = (pd.np.ceil((data['diagdate']-pd.Timedelta(days=1)-data['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
data = data[(data['6month'] > 0) & (data['6month']<11)]
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    #x['all']= len(x)
    return x

df1=data.groupby(['Clinic Number','diagyear','6month','new']).apply(func)
mask = df1['yes']>1
#the rows with yes>1 should be converted to 1 according to the formula of dr sohn
df1.loc[mask,'yes']=1
#df1['formula_applied']=(df1['yes'])

df1 = df1.dropna()
df1 = df1[['6month','yes','new','Clinic Number']].drop_duplicates()
df1['sum'] = (df1.groupby(['6month','new'])['yes']
                      .transform('sum'))
df1['numPatients6month'] = (df1.groupby(['6month'])['Clinic Number']
                      .transform('nunique'))
df1['final-formula'] = (df1['sum']/df1['numPatients6month'])
#df1 = (df1[df1['Question Text'] !='X'])
#dftest=df1.loc[(df1['6month']==1) & (df1['new']=='b-ADL') & (df1['Answer Text']=='No')]
#print(df1.to_string())
#df1.to_csv('test2.csv')
# # ######noCI part I know this way is dirty I have to show the result in 1 hour :D


names = ["Clinic Number","Question Text","Answer Text","Answer Date",'date']
nocidata = pd.read_csv('C:/Users/M193053/PycharmProjects/ADL-distribution/noCIgrouping.csv')

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
def groupmapping():
    nocidata['new'] = 'b-ADL'
    nocidata.loc[nocidata['Question Text']=='bathing','new']='b-ADL'
    nocidata.loc[nocidata['Question Text']=='dressing','new']='b-ADL'
    nocidata.loc[nocidata['Question Text']=='transferring','new']='b-ADL'
    nocidata.loc[nocidata['Question Text']=='toileting','new']='b-ADL'
    nocidata.loc[nocidata['Question Text']=='feeding','new']='b-ADL'
    nocidata.loc[nocidata['Question Text']=='transportation','new']='i-ADL'
    nocidata.loc[nocidata['Question Text']=='shopping','new']='i-ADL'
    nocidata.loc[nocidata['Question Text']=='preparing food','new']='i-ADL'
    nocidata.loc[nocidata['Question Text']=='housekeeping','new']='i-ADL'
    nocidata.loc[nocidata['Question Text']=='medications','new']='i-ADL'
    nocidata.loc[nocidata['Question Text']=='CI_FINANCE','new']='i-ADL'
    nocidata.loc[nocidata['Question Text']=='CI_CONCENTRATING','new']='MJO-Concentrating'
    nocidata.loc[nocidata['Question Text']=='CI function','new']='MJO-Concentrating'
    nocidata.loc[nocidata['Question Text']=='CI_MJO','new']='MJO-Concentrating'
groupmapping()

nocidata = nocidata[~nocidata['Question Text'].isin(["knee","USUAL ACTIVITIES (e.g. work, study, housework, family or leisure activities)","Moving or speaking so slowly that other people could have notices. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual","Personal care (washing, dressing, etc.)"])]
searchfor = ['knee','trouble getting in and out','Question Text']
nocidata = nocidata[nocidata['Question Text'].str.contains('|'.join(searchfor))==False]
nocidata['Answer Text'].fillna('NA')
nocidata['date'].fillna('NA')
nocidata['Answer Date'].fillna('NA')
nocidata=nocidata.dropna()

nocidata['year'] = nocidata['Answer Date'].str.split('/').str[2]
nocidata['month'] = nocidata['Answer Date'].str.split('/').str[0]

nocidata['diagyear'] = nocidata['date'].str.split('/').str[2]
nocidata['diagmonth'] = nocidata['date'].str.split('/').str[0]

nocidata['Answer Date'] = pd.to_datetime(nocidata['Answer Date'])
nocidata['date'] = pd.to_datetime(nocidata['date'])

nocidata['6month'] = (pd.np.ceil((nocidata['date']-pd.Timedelta(days=1)-nocidata['Answer Date'])
                                             /pd.np.timedelta64(6, 'M')).astype(int))
nocidata = nocidata[(nocidata['6month'] > 0) & (nocidata['6month']<11)]
#print(nocidata.to_string())
def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    #x['all']= len(x)
    return x

nocidf=nocidata.groupby(['Clinic Number','diagyear','6month','new']).apply(func)
mask = nocidf['yes']>1
#the rows with yes>1 should be converted to 1 according to the formula of dr sohn
nocidf.loc[mask,'yes']=1
#df1['formula_applied']=(df1['yes'])

nocidf = nocidf.dropna()
nocidf = nocidf[['6month','yes','new','Clinic Number']].drop_duplicates()
nocidf['sum'] = (nocidf.groupby(['6month','new'])['yes']
                      .transform('sum'))
nocidf['numPatients6month'] = (nocidf.groupby(['6month'])['Clinic Number']
                      .transform('nunique'))
nocidf['final-formula'] = (nocidf['sum']/nocidf['numPatients6month'])
#nocidf = (nocidf[nocidf['Question Text'] !='X'])
nocidffinal = nocidf[['6month','final-formula','new','numPatients6month']].drop_duplicates().sort_values(['6month'])
# print(nocidffinal.to_string())
nocidffinal = nocidffinal.drop('numPatients6month', 1).groupby(['6month','new']).sum().unstack('new')
# print(nocidffinal.to_string())
nocidffinal.columns = nocidffinal.columns.droplevel()
dffinal = df1[['6month','final-formula','new','numPatients6month']].drop_duplicates().sort_values(['6month'])
# print(dffinal.to_string())
dffinal = dffinal.drop('numPatients6month', 1).groupby(['6month','new']).sum().unstack('new')
# print(dffinal.to_string())
#
dffinal.columns = dffinal.columns.droplevel()
import seaborn as sns
import matplotlib.pyplot as plt
#ax=df.plot(kind='bar', stacked=True)
plt.xticks(range(0,10), ['6month','1 year','1.5 year','2 year','2.5 year','3 year','3.5 year','4 year','4.5 year','5 year'], fontsize=8, rotation=45)
#dffinal.plot(x='6month', y='final-formula')
# # plt.title('Cognitive Impairement-Stack bar')
# # plt.savefig('C:/Users/M193053/PycharmProjects/ADL-distribution/ratio_evaluation/ratio-files-plot/noCI-CI/stack-Ratio-CI.png')
# # plt.show()
# plt.rcParams['axes.prop_cycle'] = ("cycler('color', 'rg')")
# dffinal['CI-noCI']='Cognitive Impairement'
# nocidffinal['CI-noCI']='Non Cognitive Impairement'
# res=pd.concat([dffinal,nocidffinal])
#
# ax=sns.barplot(x='6month',y='final-formula',data=res,hue='CI-noCI',stacked=True).set_title('s')
# # plt.xticks(fontsize=8, rotation=45)

df_both = pd.concat(dict(CI = dffinal, noCI = nocidffinal),axis = 0)
df_both.swaplevel(0,1).sort_index().plot(kind="bar", stacked=True)

plt.savefig('C:/Users/M193053/PycharmProjects/ADL-distribution/binary_evaluation/binary-files-plot/grouping/group-CI-noCI.png')
plt.show()
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

# def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
#     """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
# labels is a list of the names of the dataframe, used for the legend
# title is a string for the title of the plot
# H is the hatch used for identification of the different dataframe"""
#
#     n_df = len(dfall)
#     n_col = len(dfall[0].columns)
#     n_ind = len(dfall[0].index)
#     axe = plt.subplot(111)
#
#     for df in dfall : # for each data frame
#         axe = df.plot(kind="bar",
#                       linewidth=0,
#                       stacked=True,
#                       ax=axe,
#                       legend=False,
#                       grid=False,
#                       **kwargs)  # make bar plots
#
#     h,l = axe.get_legend_handles_labels() # get the handles we want to modify
#     for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
#         for j, pa in enumerate(h[i:i+n_col]):
#             for rect in pa.patches: # for each index
#                 rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
#                 rect.set_hatch(H * int(i / n_col)) #edited part
#                 rect.set_width(1 / float(n_df + 1))
#
#     axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
#     axe.set_xticklabels(dffinal.index, rotation = 0)
#     axe.set_title(title)
#
#     # Add invisible data to add another legend
#     n=[]
#     for i in range(n_df):
#         n.append(axe.bar(0, 0, color="gray", hatch=H * i))
#
#     l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
#     if labels is not None:
#         l2 = plt.legend(n, labels, loc=[1.01, 0.1])
#     axe.add_artist(l1)
#     return axe
#
# # Then, just call :
# plot_clustered_stacked([dffinal, nocidffinal],["CI", "noCI"])







