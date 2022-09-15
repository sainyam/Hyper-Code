#!/usr/bin/env python
# coding: utf-8

# In[161]:

import time
import numpy as np
import pandas as pd
from collections import namedtuple, Counter
import copy


# In[162]:


def random_logit(x):
    z = 1./(1+np.exp(-x))
    #print(z) 
    s = np.random.binomial(n=1, p=z)

    return s
def roundlst(x):
    l=[]
    for v in x:
        if v>0.5:
            l.append(1)
        else:
            l.append(0)
    return l
def roundl(x,th):
    l=[]
    for v in x:
        if v>th:
            l.append(1)
        else:
            l.append(0)
    return l

def roundl4(x,th):
    l=[]
    for v in x:
        if v<=th:
            l.append(0)
        elif v<=2*th:
            l.append(1)
        elif v<=3*th:
            l.append(2)
        else:# v<=4*th:
            l.append(3)
    return l
from sklearn.preprocessing import KBinsDiscretizer

def bucketize(lst,num_buckets):
    maxval=max(lst)
    minval=min(lst)
    binlst=[minval-0.001]
    i=0
    labels=[]
    size=(maxval-minval)/num_buckets
    while i<num_buckets:
        binlst.append(binlst[-1]+size)#0.001)
        labels.append(i)
        i+=1
    binlst[-1]+=0.001
    return (pd.cut(x=lst, bins=binlst,labels=labels),binlst)



# In[168]:


def get_data(N,seed,num_bins):
    #N=10000
    np.random.seed(seed)
    S=np.random.binomial(n=1, p=0.5,size=N)
    A=np.random.binomial(n=2, p=0.5,size=N)
    noise1=np.random.binomial(n=2, p=0.01,size=N)
    noise2=np.random.binomial(n=2, p=0.01,size=N)
    noise3=np.random.binomial(n=2, p=0.01,size=N)
    noise4=np.random.binomial(n=2, p=0.01,size=N)
    St=((2*S+A+ noise1 )/2)
    Cred=np.random.binomial(n=2, p=0.21,size=N)#np.array(((0.5*S+1.5*A+ noise1 )/2))
    
    sav=np.array(((S+A+noise2)/3))
            
    hous=np.array(((S+noise3)/2))
    bins={}
    #X1 = (2*S+A+np.random.normal(loc=0.0, scale=0.2, size=N))#(np.random.normal(loc=100.0, scale=1.16, size=N)) # N(0,1)
    #X2 =(3*A+2*S+np.random.normal(loc=0.0, scale=0.2, size=N))#random_logit(100*X1)#+np.random.normal(loc=0.0, scale=0.16, size=N)
    #Y = random_logit((A+3*St+2*sav+hous)/50)
    Y = ((A+St/3+sav/3+hous/3+noise4))#+Cred*0.01))
    df=pd.DataFrame(S,columns=['S'])
    df_U=pd.DataFrame(S,columns=['S'])
    df_U['noise1']=noise1
    df_U['noise2']=noise2
    df_U['noise3']=noise3
    df_U['noise4']=noise4
    df['A']=A
    df['S']=S
    df['St_orig']=St
    df['Cred_orig']=Cred
    df['sav_orig']=sav
    df['hous_orig']=hous
    (df['St'],bins['St']) = bucketize(St,num_bins)
    (df['Cred'],bins['Cred']) = bucketize(Cred,num_bins)
    (df['sav'],bins['sav']) = bucketize(sav,num_bins)
    (df['hous'],bins['hous']) = bucketize(hous,num_bins)
    df['Y']=Y
    
    
    return (df,df_U,bins)
def causal_effect(dolst,df,df_U):

    N=df.shape[0]
    if 'S' in dolst.keys():
        S=np.array([dolst['S']]*df.shape[0])#Construct a list of same size as df
    else:
        S=df['S']
        
        
    if 'A' in dolst.keys():
        A=np.array([dolst['A']]*df.shape[0])#Construct a list of same size as df
    else:
        A=df['A']


    noise1=df_U['noise1']
    noise2=df_U['noise2']
    noise3=df_U['noise3']
    noise4=df_U['noise4']
    if 'St' in dolst.keys():
        St= np.array([dolst['St']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        St= np.array(roundl((2*S+A+ noise1 )/2,1))
    else:
        St=df['St']
        
    if 'sav' in dolst.keys():
        sav=np.array([dolst['sav']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        sav=np.array(roundl((S+A+noise2)/3,0.5))
    else:
        sav=df['sav']
        
    if 'hous' in dolst.keys():
        hous=np.array([dolst['hous']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        hous=np.array(roundl((S+noise3)/2,0.5))
    else:
        hous=df['hous']
        
    if 'Cred' in dolst.keys():
        Cred=np.array([dolst['Cred']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        Cred=np.array(roundl((0.5*S+1.5*A+ noise1 )/2,1))
    else:
        Cred=df['Cred']  

    #print (St)
    #X1 = (2*S+A+np.random.normal(loc=0.0, scale=0.2, size=N))#(np.random.normal(loc=100.0, scale=1.16, size=N)) # N(0,1)
    #X2 =(3*A+2*S+np.random.normal(loc=0.0, scale=0.2, size=N))#random_logit(100*X1)#+np.random.normal(loc=0.0, scale=0.16, size=N)
    #Y = random_logit((A+3*St+2*sav+hous)/50)
    Y = roundlst((St/3+sav/3+hous/3+noise4))
    df=pd.DataFrame(np.array(list(S)),columns=['S'])
    

    df['A']=np.array(list(A))
    #print(df)
    df['S']=np.array(list(S))
    df['St']=np.array(list(St))
    df['sav']=np.array(list(sav))
    df['hous']=np.array(list(hous))
    df['Cred']=np.array(list(Cred))
    df['Y']=np.array(list(Y))
    
    return df
    #For each variable in dolst, change their descendants
        
        


# In[197]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def get_val(row,target,target_val):
    i=0
    while i<len(target):
        #print (row[target[i]],target_val[i])
        if not int(row[target[i]])==int(target_val[i]):
            return 0
        i+=1
    return 1

def get_prob_o_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
    
    X=df[conditional]
    regr = LogisticRegression(random_state=0)#RandomForestRegressor(max_depth=5, random_state=0)
    regr.fit(X, new_lst)
    return (regr.predict_proba([conditional_values])[0][1])
    #return(regr.predict([conditional_values])[0])
    

def get_logistic_param(df,conditional,target,target_val):
    new_lst=[]
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
    X=df[conditional]
    regr = LinearRegression()
    regr.fit(X, df[target])
    print (regr.coef_)
    print (regr.intercept_)
    return (regr.coef_.tolist()[0],[regr.intercept_])
    #print(regr.get_params())
    
    


# In[198]:


import math
from mip import Model, xsum, maximize, minimize, BINARY
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

import copy

def optimization(df,A,aval,Adomain,klst,kval,alpha,betalst,beta0):
    
    backdoorvals=get_C_set(df,klst)
    print (backdoorvals,len(A),len(betalst))
    
    betalst_backdoor=betalst[len(A):]
    sum_backdoor=0
    for lst in backdoorvals:
        iter=0
        sampled_df=copy.deepcopy(df)
        tmpsum=0
        while iter<len(lst):
            sampled_df=sampled_df[sampled_df[klst[iter]]==lst[iter]]
            tmpsum+=betalst_backdoor[iter]*lst[iter]
            iter+=1
        sum_backdoor+= tmpsum*(sampled_df.shape[0]*1.0/df.shape[0])
        print("shapes",sampled_df.shape[0],df.shape[0],tmpsum)
    print (sum_backdoor)
     
    
    
    m = Model("Test")
    i=0
    var_lst=[]
    var_map={}
    while i<len(A):
        j=0
        while j<len(Adomain[i]):

            var_lst.append(m.add_var(var_type=BINARY))
            var_map[len(var_lst)-1]=(i,j)
            j+=1
        i+=1
    print ("beta list is ",betalst,beta0)
    cost_lst=[]
    constr_lst=[]
    constr_lst.append(beta0)
    iter=0
    i=0
    while i<len(A):
        j=0
        del_cons=[]
        while j<len(Adomain[i]):
            '''
            if Adomain[i][j]==aval[i]:
                constr_lst.append(betalst[i]*Adomain[i][j])
                j+=1
                continue
            '''
            constr_lst.append(betalst[i]*var_lst[iter]*(Adomain[i][j]))
            del_cons.append(var_lst[iter])
            iter+=1
            j+=1
        m += xsum(del_cons) <= 1
        i+=1
    constr_lst.append(sum_backdoor)
    '''i=len(A)
    while i<len(betalst):
        constr_lst.append(betalst[i]*kval[i-len(A)])
        i+=1
    '''

    print (constr_lst)
    #m+=xsum(constr_lst)>=math.log(alphak*1.0/(1-alphak))
    m.objective = maximize(xsum(constr_lst))
    m.optimize()
    if m.num_solutions:
        print('Objective value %g found:'
                  % (m.objective_value))
        i=0
        score=0
        gt_score=0
        while i<len(var_lst):
            print (i,var_lst[i].x,var_map[i], Adomain[var_map[i][0]][var_map[i][1]])
            if var_lst[i].x==1:
                vallst=[]
                if var_map[i][0]==0:
                    vallst=bins['St']
                elif var_map[i][0]==1:
                    vallst=bins['sav']
                elif var_map[i][0]==2:
                    vallst=bins['hous']
                else:
                    vallst=bins['Cred']
                print (vallst,(vallst[var_map[i][1]],vallst[var_map[i][1]+1]))
                print ("sc",betalst[var_map[i][0]]*((vallst[var_map[i][1]]+vallst[var_map[i][1]+1])/2))
                score+= betalst[var_map[i][0]]*((vallst[var_map[i][1]]+vallst[var_map[i][1]+1])/2)
                gt_score+=betalst[var_map[i][0]]*((vallst[-1]))
            i+=1
        print ("sum backdoor",sum_backdoor,score) 
        print ("score is ",score+sum_backdoor+beta0,gt_score+sum_backdoor+beta0)
        print ("score is ",score,sum_backdoor,beta0)#,gt_score+sum_backdoor+beta0)
        return (score+sum_backdoor+beta0)*1.0/(gt_score+sum_backdoor+beta0),(gt_score+sum_backdoor+beta0)
    


# In[ ]:





# In[199]:
scores={}
times={}
opt_times={}
opt_scores={}
for num_bins in [1,2,4,6,8,10]:
    print (num_bins)
    i=0
    lst=[]
    while i<num_bins:
        lst.append(i+1)
        i+=1
    
    (df,df_U,bins)=get_data(20000,0,num_bins)

    print (df)
    A=['St_orig','sav_orig','hous_orig','Cred_orig']#,'housing','employment']
    aval=[1,0,0]

    klst=['A','S']
    kval=[]

    target='Y'
    start=time.time()
    conditionals=copy.deepcopy(A)
    conditionals.extend(klst)
    #print (conditionals)
    (beta_lst,[[beta0]])=get_logistic_param(df,conditionals,[target],[1])
    print(beta_lst,beta0)


    Adomain=[lst,lst,lst,lst]#,[0,1],[0,1,2,3,4]]#List of list where each list is domain of each variable in A

    scores[num_bins],gt_score=optimization(df,A,aval,Adomain,klst,kval,0.8,beta_lst,beta0)
    end=time.time()
    times[num_bins]=end-start

    start=time.time()
    domain_lst=get_combination(Adomain,[])
    maxval=0
    for lst in domain_lst:
        print(lst)
        St_bins=bins['St']
        sav_bins=bins['sav']
        hous_bins=bins['hous']
        cred_bins=bins['Cred']
        (beta_lst,[[beta0]])=get_logistic_param(df,conditionals,[target],[1])
        print (conditionals,beta_lst)
        stval=beta_lst[0]*((St_bins[lst[0]-1]+St_bins[lst[0]])/2)
        savval=beta_lst[1]*((sav_bins[lst[0]-1]+sav_bins[lst[0]])/2)
        housval=beta_lst[2]*((hous_bins[lst[0]-1]+hous_bins[lst[0]])/2)
        credval=beta_lst[3]*((cred_bins[lst[0]-1]+cred_bins[lst[0]])/2)
    
        backdoorvals=get_C_set(df,klst)
    
        betalst_backdoor=beta_lst[len(A):]
        sum_backdoor=0
        for lst in backdoorvals:
            iter=0
            sampled_df=copy.deepcopy(df)
            tmpsum=0
            while iter<len(lst):
                sampled_df=sampled_df[sampled_df[klst[iter]]==lst[iter]]
                tmpsum+=betalst_backdoor[iter]*lst[iter]
                iter+=1
            sum_backdoor+= tmpsum*(sampled_df.shape[0]*1.0/df.shape[0])

            print(sampled_df.shape[0])
        if sum_backdoor+stval+savval+housval+credval> maxval:
            maxval=sum_backdoor+stval+savval+housval+credval
        print (sum_backdoor+stval+savval+housval)
    print ("sum backdoor",sum_backdoor,stval,savval,housval) 
    end=time.time()
    opt_times[num_bins]=end-start
    opt_scores[num_bins]=(maxval+beta0)*1.0/gt_score
    print (end-start)
#(A,aval,Adomain,klst,kval,alpha,betalst,beta0):

print (scores)
print (opt_scores)
print(times)
print(opt_times)
# In[ ]:


hyper=[]
baseline=[]
for num_bins in [1,2,4,6,8,10]:
    hyper.append(scores[num_bins])
    baseline.append(opt_scores[num_bins])

import pandas as pd
from functools import reduce

import seaborn as scs

hypername=[1,2,4,6,8,10]

gtname=[1,2,4,6,8,10]

#gt=[0.99,0.6298,0.507,0.5007,0.02]
#gtname=[ 'Age',  'Saving','Status','Sex', 'Housing']


#baseline=[0.640,0.76,0.91,0.92, 0.95, 0.98]
baselinename=[1,2,4,6,8,10]


#baseline_all=[0.64,0.44, 0.57,0.43]
baselinename_all=[1,2,4,6,8,10]


num_feat=4

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

labels = gtname
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
plt.figure(figsize=(6, 10)) # in inches!

fig, ax = plt.subplots()
#rects1 = ax.bar( x - 0.15,gt, width, label='Ground Truth', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.bar(x, hyper, width, label='Hyper-sampled', color='gainsboro', edgecolor='black', hatch="\\\\")
#rects3 = ax.bar(x + 0.15, baseline_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
rects4 = ax.bar(x + 0.15, baseline, width, label='Opt-discrete', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Solution Quality', fontsize=fsize, labelpad=fsize/2)
ax.set_xlabel('Number of buckets', fontsize=fsize, labelpad=fsize/2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fsize)
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.yticks(np.arange(0, 2, .5))
ax.legend(loc=(0.5,0.85))
#ax.invert_yaxis()
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

'''
autolabel(rects1,shapley_rank)
autolabel(rects2,feat_rank)
autolabel(rects3,sn_rank)
autolabel(rects4,tr_rank)
'''
ax.margins(0.1,0.05)
plt.ylim([0, 1.3])
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('9a.pdf')



import seaborn as scs
hyper=[]
baseline=[]

for num_bins in [1,2,4,6,8,10]:
    hyper.append(times[num_bins])
    baseline.append(opt_times[num_bins])


hypername=[1,2,4,6,8,10]


gtname=[1,2,4,6,8,10]

#gt=[0.99,0.6298,0.507,0.5007,0.02]
#gtname=[ 'Age',  'Saving','Status','Sex', 'Housing']


baselinename=[1,2,4,6,8,10]


baselinename_all=[1,2,4,6,8,10]
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

labels = gtname
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
plt.figure(figsize=(6, 10)) # in inches!

fig, ax = plt.subplots()
#rects1 = ax.bar( x - 0.15,gt, width, label='Ground Truth', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.bar(x, hyper, width, label='Hyper-sampled', color='gainsboro', edgecolor='black', hatch="\\\\")
#rects3 = ax.bar(x + 0.15, baseline_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
rects4 = ax.bar(x + 0.15, baseline, width, label='Opt-discrete', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Running Time (seconds)', fontsize=fsize, labelpad=fsize/2)
ax.set_xlabel('Number of buckets', fontsize=fsize, labelpad=fsize/2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fsize)
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.yticks(np.arange(0, 3.5, .5))
ax.legend(loc=(0.25,0.85))
#ax.invert_yaxis()
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

'''
autolabel(rects1,shapley_rank)
autolabel(rects2,feat_rank)
autolabel(rects3,sn_rank)
autolabel(rects4,tr_rank)
'''
ax.margins(0.1,0.05)
plt.yscale("log")

plt.ylim([0, 35000])
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('9b.pdf')
