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

#print(data.to_string())
data['6month'] = (pd.np.ceil((data['diagdate']-pd.Timedelta(days=1)-data['Answer Date'])
                                              /pd.np.timedelta64(6, 'M')).astype(int))
data = data[(data['6month'] > 0) & (data['6month']<11)]
def func(x):
    x['yes']= len(x[x['Answer Text']=='Yes'])
    #x['all']= len(x)
    return x

df=data.groupby(['Clinic Number','diagyear','6month','Question Text']).apply(func)
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

df = df[df['Question Text'] !='X']
dffinal = df[['6month','final-formula','Question Text','numPatients6month']].drop_duplicates().sort_values(['6month'])

df = dffinal.drop('numPatients6month', 1).groupby(['6month','Question Text']).sum().unstack('Question Text')
import matplotlib.pyplot as plt
df.columns = df.columns.droplevel()
ax=df.plot(kind='bar', stacked=True)

list_values = (dffinal.drop('final-formula', 1).groupby(['6month','Question Text']).sum()
               .unstack('Question Text').fillna(0).astype(int).values.flatten('F'))
for rect, value in zip(ax.patches, list_values):
    if value != 0:
        h = rect.get_height() /2.
        w = rect.get_width() /2.
        x, y = rect.get_xy()
        ax.text(x+w, y+h,value,horizontalalignment='center',verticalalignment='center')

plt.xticks(range(0,10), ['6month','1 year','1.5 year','2 year','2.5 year','3 year','3.5 year','4 year','4.5 year','5 year'], fontsize=8, rotation=45)

plt.title('Cognitive Impairement-Stack bar')

plt.savefig('C:/Users/M193053/PycharmProjects/ADL-distribution/binary_evaluation/binary-files-plot/newFunctionality-mental/binary-stack-mental-CI.png')
plt.show()
