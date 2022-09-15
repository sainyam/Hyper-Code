
import numpy as np
import pandas as pd
from collections import namedtuple, Counter

import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_regression

import helper,germansyn

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

def train_regression_raw(df,conditional,conditional_values,AT):
    new_lst=[]
    count=0
    '''
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
    '''
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    #regr = RandomForestRegressor(random_state=0)
    regr = LinearRegression()#random_state=0)
    regr.fit(X, df[AT])
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
    start = time.process_time()

    regr = RandomForestRegressor(random_state=0)
    
    #regr = LogisticRegression(random_state=0)
    regr.fit(X, new_lst)
    #print("timesssssssss",time.process_time() - start)

    #print (regr.coef_.tolist())
    #print (regr.predict_proba([conditional_values]),"ASDFDS")
    print ("heeeeeere")
    return (regr.predict([conditional_values].values)[0])
    #return(regr.predict_proba([conditional_values])[0][1])
  



# In[78]:


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


# In[79]:


import time
def get_query_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c,g_Ac_lst,interference, blocks):
    #interference is set of attributes of other tuples in a block that affect current tuple's attribute
    #blocks are list of lists
    
    #Identify all attributes which are used for regression and add as columns 
    
            
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
            #print(backdoorvals)
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
                start = time.process_time()

                regr=train_regression(df,conditioning_set,conditioning_val,postlst,postvallst)
                #print("time",time.process_time() - start)
            #print (conditioning_val)
            pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
            #print("this",prelst,prevallst,backdoorlst,backdoorvallst)
            pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
            #print (pogivenck,pcgivenk)
            total_prob+=pogivenck * pcgivenk
            iter+=1
            
        print("final prob is ",total_prob)
        #print (iter)
        return total_prob
    if q_type=='avg':
        
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
            backdoorvals=[[]]
        total_prob=0
        regr=''
        iter=0
        print (backdoorvals)
        
        
        for backdoorvallst in backdoorvals:
            
            conditioning_set=[]
            conditioning_set.extend(prelst)
            conditioning_set.extend(Ac)
            conditioning_set.extend(backdoorlst)

            conditioning_val=[]
            conditioning_val.extend(prevallst)
            conditioning_val.extend(c)
            conditioning_val.extend(backdoorvallst)

            #print ("conditioning set",conditioning_set,conditioning_val, AT)
            if iter==0:
                regr=train_regression_raw(df,conditioning_set,conditioning_val,AT)
            pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
            #print("this",prelst,prevallst,backdoorlst,backdoorvallst)
            pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
            #print (pogivenck,pcgivenk)
            total_prob+=pogivenck * pcgivenk
            iter+=1
            
        #print("final prob is ",total_prob)
        return total_prob

        
        
        
        


# In[80]:


(df,df_U)=germansyn.get_data(10000,0)
feat=list(df.columns)


# In[81]:


backdoor={'S':[],'A':[],'St':['S','A'],'sav':['S','A'],'hous':['S','A'],'Cred':['S','A']}


# In[82]:


import time


(orig_df,df_U)=germansyn.get_data(1000000,0)
print ("running 1million")
orig_score=get_query_output(orig_df,'count','',[],[],['Y'],[1],['St'],[1],['*'],'',{})#,{0:[1,2]})
print ("completed 1million")
scores={}
times={}
for size in [1000,10000,25000,50000,100000,200000]:
    scores[size]=[]
    times[size]=[]
    for seed in [0,1,2,3,4,5,6,7,8,9,10]:
        print (seed,size)
        df=orig_df.sample(n=size,random_state=seed)
        feat=list(df.columns)
        start=time.time()
        score=get_query_output(df,'count','',[],[],['Y'],[1],['St'],[1],['*'],'',{})
        end=time.time()
        times[size].append (end-start)
        scores[size].append(score)
print (scores)
print (times)


# In[ ]:





# In[83]:


runtime=[]
for size in [10000,100000,200000,400000,600000,800000,1000000]:
    if size in times.keys():
        runtime.append(np.mean(times[size]))
        continue
    df=orig_df.sample(n=size,random_state=0)
    feat=list(df.columns)
    start=time.time()
    score=get_query_output(df,'count','',[],[],['Y'],[1],['St'],[1],['*'],'',{})
    end=time.time()
    runtime.append (end-start)

print (runtime)


# In[ ]:





# In[84]:


import statistics
import numpy as np
average=[]
dev=[]
gt=[]
for size in scores.keys():
    average.append (np.mean(scores[size]))
    dev.append (statistics.stdev(scores[size]))
    gt.append(orig_score)
print (average)
print(dev)


# In[85]:


from matplotlib import pyplot as plt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pylab as plot


fsize=20
params = {'legend.fontsize': fsize,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = [1000,10000,25000,50000,100000,200000]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

plt.figure(figsize=(6, 5)) # in inches!

#fig, ax = plt.subplots()


y = np.array([0.6130208937041055,0.6130208937041055,0.6130208937041055,0.6130208937041055,0.6130208937041055,0.6130208937041055])
x = np.array([1000,10000,25000,50000,100000,200000])
#error = np.array([0.02,0.003,0.0023,0.0001,0.001])#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)


y1 = np.array(average)
error1 = np.array(dev)#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)
print(error1)

plt.xticks(fontsize= fsize/1.2)
plt.plot(x, y, 'k-v',label='Ground Truth',color='salmon',markersize=18)
plt.plot(x, y1, 'k-x',label='Hyper-sampled',color='forestgreen',markersize=18)
plt.fill_between(x, y, y, alpha=0.3)
plt.fill_between(x, y1-error1, y1+error1, alpha=0.15,color='forestgreen')
#plt.show()
plt.xticks([0, 1000,50000, 100000,200000], ['', '1K','50K', '100K','200K'])
plot.ylim([0.55,0.85])
plt.legend()
#fig.tight_layout()
plt.xlabel('Sample Size',labelpad=5, fontsize=fsize/1.1)
plt.savefig('../freshRuns/6a.pdf')



# In[86]:


from matplotlib import pyplot as plt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=[0.010,0.1,0.2,0.4,0.6,0.8,1.0]
our=[390,390,390,390,390,390,390]
oursampled=[5.7,44.5,88.2,178.4,240,320,390]
indep=[1.1,8.9,19.2,36,55,74,95]
#ourNB=[22.4,222.5,80.2,164.4,232,310,390]
    
import pylab as plot
params = {'legend.fontsize': 65,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : 65}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = snname=[0.010,0.1,0.2,0.4,0.6,0.8,1.0]

x = np.arange(len(labels))  # the label locations

width = 0.25  # the width of the bars

plt.figure(figsize=(20, 20)) # in inches!

#fig, ax = plt.subplots()

#rects1 = ax.barh(x - width, trsn, width,xerr=trsnvar, label='Ground Truth', color='coral', edgecolor='black', hatch="/",error_kw=dict(elinewidth=5, ecolor='black'))
#rects2 = ax.barh(x, sn, width, xerr=snvar,label='RAVEN', color='forestgreen', edgecolor='black', hatch="||",error_kw=dict(elinewidth=5, ecolor='black'))
#rects3 = ax.barh(x + width, allScores['sn'], width, label='NeSuf', color='royalblue')


y = np.array([runtime[-1],runtime[-1],runtime[-1],runtime[-1],runtime[-1],runtime[-1],runtime[-1]])
x = np.array([0.010,0.1,0.2,0.4,0.6,0.8,1.0])

#error = np.array([0.02,0.003,0.0023,0.0001,0.001])#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)


#y1 = np.array(indep)
y2=np.array(runtime)
#error1 = np.array(snvar)#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)
#print(error1)

plt.plot(x, y, 'k-v',label='HypeR',linewidth=5,color='black',markersize=40)
plt.plot(x, y2, 'k-s',label='HypeR-sampled',linewidth=5,color='pink',markersize=40)
#plt.plot(x, y1, 'k-s',label='Indep',linewidth=5,color='forestgreen',markersize=40)

plt.fill_between(x, y, y, alpha=0.3)
#plt.fill_between(x, y1, y1, alpha=0.15,color='forestgreen')
#plt.show()
#plt.xticks([0, 1000,50000, 100000], ['', '1K','50K', '100K'])
#plot.ylim([0.4,0.7])
plt.legend()
plt.xlabel('Sample Size (in millions)',labelpad=30)
plt.ylabel('Time (in seconds)',labelpad=-11)
plt.savefig('../freshRuns/6b.pdf', bbox_inches='tight')



