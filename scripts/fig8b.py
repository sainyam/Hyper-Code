import numpy as np
import pandas as pd
from collections import namedtuple, Counter

import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import adult


df=adult.read_data()



X=df




from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_regression

debug=False
def get_combination(lst,tuplelst):
    i=0
    new_tuplelst=[]
    if len(tuplelst)==0:
        l=lst[0]
        for v in l:
            new_tuplelst.append([v])
        if len(lst)>1:
            return get_combination(lst[1:],new_tuplelst)
        else:
            return new_tuplelst
    

    currlst=lst[0]
    for l in tuplelst:
        
        for v in currlst:
            newl=copy.deepcopy(l)
            newl.append(v)
            new_tuplelst.append(newl)
        
    if len(lst)>1:
        return get_combination(lst[1:],new_tuplelst)
    else:
        return new_tuplelst
      
def get_C_set(df,C):
    lst=[]
    for Cvar in C:
        lst.append(list(set(list(df[Cvar]))))
        
    combination_lst= (get_combination(lst,[]))
    
    return combination_lst


def get_val(row,target,target_val):
    i=0
    while i<len(target):
        if not int(row[target[i]])==int(target_val[i]):
            return 0
        i+=1
    return 1
def train_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    count=0
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
        if new_lst[-1]==1:
            count+=1
    if len(conditional)==0:
        return count*1.0/df.shape[0]
    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return 1
        else:
            return 0
        
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    regr = RandomForestRegressor(random_state=0)
    #regr = LogisticRegression(random_state=0)
    regr.fit(X.values, new_lst)
    return regr
def get_prob_o_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    count=0
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
        if new_lst[-1]==1:
            count+=1
    if len(conditional)==0:
        return count*1.0/df.shape[0]
    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return 1
        else:
            return 0
        
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    regr = RandomForestRegressor(random_state=0)
    #regr = LogisticRegression(random_state=0)
    regr.fit(X.values, new_lst)
    #print (regr.coef_.tolist())
    #print (regr.predict_proba([conditional_values]),"ASDFDS")
    return (regr.predict([conditional_values])[0])
    #return(regr.predict_proba([conditional_values])[0][1])
  



# In[46]:


def check_g_Ac(row,g_Ac_lst):
    i=0
    for g_attr_lst in g_Ac_lst:
        if '*' in g_attr_lst:
            return True
        found=True
        for (attr,attrval) in g_attr_lst:
            if row[attr] == attrval:
                continue
            else:
                found=True
                break
        if found:
            return True
    return False


# In[47]:


def get_query_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c,g_Ac_lst,interference, blocks):

    print(df)
   
            
    #print (len(sub_df),len(sub_intervene))
    if q_type=='count':
        conditioning_set=prelst
        #        intervention=
        backdoorlst=[]
        for attr in Ac:
            backdoorlst.extend(backdoor[attr])
        backdoorlst=list(set(backdoorlst))
        if len(backdoorlst)>0:
            backdoorvals=get_C_set(df,backdoorlst)
            print(backdoorvals)
        else:
            backdoorvals=[]
        total_prob=0
        regr=''
        iter=0
        for backdoorvallst in backdoorvals:
            conditioning_set=[]
            conditioning_set.extend(prelst)
            conditioning_set.extend(Ac)
            conditioning_set.extend(backdoorlst)

            conditioning_val=[]
            conditioning_val.extend(prevallst)
            conditioning_val.extend(c)
            conditioning_val.extend(backdoorvallst)

            #print ("conditioning set",conditioning_set,conditioning_val)
            #print("post condition",postlst,postvallst)
            if iter==0:
                regr=train_regression(df,conditioning_set,conditioning_val,postlst,postvallst)
            pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
            
            
            print(prelst,prevallst,backdoorlst,backdoorvallst)
            pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
            print (pogivenck,pcgivenk)
            total_prob+=pogivenck * pcgivenk
            iter+=1
            
        print("final prob is ",total_prob)
        return total_prob



scores={}
raw_scores={}
for col in ['marital','occupation','edu','class']:
    values= list(set(df[col].values))
    scores[col]=[]
    for v in values:
        scores[col].append(get_query_output(df,'count','',[],[],['target'],[1],[col],[v],['*'],'',{}))#,{0:[1,2]}))
        raw_scores[(col,v)]=scores[col][-1]


# In[ ]:


hyper=[]
hypermax=[]
for col in ['marital','occupation','edu','class']:
    hyper.append(min(scores[col]))
    hypermax.append(max(scores[col]))
print(hyper,hypermax)




import seaborn as scs

hypername=['Marital', 'Occupation', 'Education','Class']

hypermaxname=['Marital', 'Occupation', 'Education','Class']


num_feat=4


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pylab as plot
fsize=20
params = {'legend.fontsize': fsize/1.2,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = hypername
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
plt.figure(figsize=(6, 10)) # in inches!

fig, ax = plt.subplots()
rects1 = ax.barh(x - 0.15, hyper, width, label='Minimum', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.barh(x, hypermax, width, label='Maximum', color='gainsboro', edgecolor='black', hatch="\\\\")
#rects3 = ax.barh(x + 0.15, baseline_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
#rects4 = ax.barh(x + 0.3, baseline, width, label='Indep', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Query Output', fontsize=fsize, labelpad=fsize/2)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=fsize)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xticks(np.arange(0, 2, .5))
ax.legend(loc=(0.52,0))
ax.invert_yaxis()
def autolabel(rects,rank):
    """Attach a text label above each bar in *rects*, displaying its height."""
    i=0
    for rect in rects:
        height = rect.get_width()
        val=len(shapley_rank)-int(rank[i])+1
        ax.annotate('{}'.format(val),
                    xy=(height - 0.02, rect.get_y() + 0.15),
                    xytext=(15, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fsize/1.5)
        i+=1

ax.margins(0.1,0.05)
plt.xlim([0, 1])
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('../freshRuns/8b.pdf')


