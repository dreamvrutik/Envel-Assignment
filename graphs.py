import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

year={}
acb=[]
amt=[]
dataset=pd.read_csv("Q_test_data.csv")
for key,value in dataset.iteritems():
    print(key)
    if key=="createdDate":
        print(1)
        for i in value:
            ye=i[:4]
            mo=i[5:7]
            if ye not in year:
                year[ye]={}
            try:
                year[ye][mo]+=1
            except Exception as e:
                year[ye][mo]=1

print(year)
for i in year:
    objects=[j for j in year[i]]
    performance=[year[i][j] for j in year[i]]
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, performance, align='center', alpha=1)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of transactions')
    plt.xlabel('Months')
    plt.title(i)
    sa=i+"_transactions.png"
    plt.savefig(sa)
    plt.show()

print(len(dataset.columns))
for cols in dataset.columns:
    print(cols)

for cols in dataset.columns:
    print(cols,len(dataset[cols].unique()))

obj = list(dataset.columns)
per = [len(dataset[cols].unique()) for cols in dataset.columns]
start=0
end=5
while end<=30:
    objects=obj[start:end]
    performance=per[start:end]
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, performance, align='center', alpha=1)
    plt.xticks(y_pos, objects)
    plt.ylabel('Unique values')
    plt.title('Column Names')
    start+=5
    end+=5
    plt.show()

dataset=dataset.drop('transactionCount',axis=1)
dataset['account_balance']=dataset['account_balance'].astype('category').cat.codes
dataset['check']=dataset['check'].astype('category').cat.codes
dataset['CS_FICO_str']=dataset['CS_FICO_str'].astype('category').cat.codes
dataset['CS_internal']=dataset['CS_internal'].astype('category').cat.codes
dataset['feeCode']=dataset['feeCode'].astype('category').cat.codes
dataset['feeDescription']=dataset['feeDescription'].astype('category').cat.codes
dataset['institutionName']=dataset['institutionName'].astype('category').cat.codes
dataset['isCredit']=dataset['isCredit'].astype('category').cat.codes
dataset['returnCode']=dataset['returnCode'].astype('category').cat.codes
dataset['status']=dataset['status'].astype('category').cat.codes
dataset['Student']=dataset['Student'].astype('category').cat.codes
dataset['subType']=dataset['subType'].astype('category').cat.codes
dataset['subTypeCode']=dataset['subTypeCode'].astype('category').cat.codes
dataset['type']=dataset['type'].astype('category').cat.codes
dataset['Age']=dataset['Age'].astype('float')
dataset['amount']=dataset['amount'].astype('float')
dataset['cardId']=dataset['cardId'].astype('float')
dataset['CS_FICO_num']=dataset['CS_FICO_num'].astype('float')
dataset['CustomerId']=dataset['customerId'].astype('float')
dataset['description']=dataset['description'].astype('float')
dataset['friendlyDescription']=dataset['friendlyDescription'].astype('float')
dataset['masterId']=dataset['masterId'].astype('float')
dataset['tag']=dataset['tag'].astype('float')
dataset['transactionId']=dataset['transactionId'].astype('float')
dataset['typeCode']=dataset['typeCode'].astype('float')
dataset['availableDate']=pd.to_datetime(dataset['availableDate'])
dataset['createdDate']=pd.to_datetime(dataset['createdDate'])
dataset['settledDate']=pd.to_datetime(dataset['settledDate'])
dataset['voidedDate']=pd.to_datetime(dataset['voidedDate'])
df=dataset
print(len(df.columns))
corr_matrix = df.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
f, ax = plt.subplots(figsize=(11, 15))

heatmap = sns.heatmap(corr_matrix,
                      mask = mask,
                      square = True,
                      linewidths = 3,
                      cmap = "coolwarm",
                      cbar_kws = {'shrink': .4,
                                "ticks" : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 5})

#add the column names as labels
ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('heatmap.png', bbox_inches='tight')
