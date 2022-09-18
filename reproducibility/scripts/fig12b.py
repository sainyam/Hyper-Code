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

def get_data_vary_var(N,seed,num_var):
    #N=10000
    np.random.seed(seed)
    
    S=np.random.binomial(n=1, p=0.5,size=N)
    A=np.random.binomial(n=2, p=0.5,size=N)
    Country=np.random.binomial(n=2, p=0.5,size=N)
    
    
    Attendance_U=np.random.binomial(n=4, p=0.5,size=N)
    Attendance=Attendance_U + S + A + Country
    
    hands_U=np.random.normal(10, 2, size=N)
    hands_raised=hands_U+ 2*S+ 3*A + Country
    
    discussion_U=np.random.normal(10, 2, size=N)
    discussion=discussion_U+ 2*Attendance + 2*S+ 3*A + Country
    
    assignment_U=np.random.normal(10, 2, size=N)
    assignment=assignment_U+ 2*S+ 3*A + Country -3*Attendance
    
    announcement_U=np.random.normal(10, 2, size=N)
    announcement=announcement_U+ 2*S+ 3*A + Country
    
    
    grade_U=np.random.normal(10, 2, size=N)
    grade=grade_U+ hands_raised + 2*discussion + assignment + announcement
    
    
      
    df=pd.DataFrame(S,columns=['S'])
    df_U=pd.DataFrame(S,columns=['S'])
    
    df['A']=A
    df['S']=S
    df['Country']=Country
    df['hands_raised']=hands_raised
    df['Attendance']=Attendance
    df['discussion']=discussion
    df['assignment']=assignment
    df['announcement']=announcement
    df['grade']=grade
    
    i=0
    while i<num_var:
        df["var"+str(i)] = np.random.binomial(n=5, p=0.5,size=N)
        i+=1
    
    
    df_U['hands']=hands_U
    df_U['Attendance']=Attendance_U
    df_U['discussion']=discussion_U
    df_U['assignment']=assignment_U
    df_U['announcement']=announcement_U
    df_U['grade']=grade_U
    
    
    return (df,df_U)


def get_data(N,seed):
    #N=10000
    np.random.seed(seed)
    
    S=np.random.binomial(n=1, p=0.5,size=N)
    A=np.random.binomial(n=2, p=0.5,size=N)
    Country=np.random.binomial(n=2, p=0.5,size=N)
    
    
    Attendance_U=np.random.binomial(n=4, p=0.5,size=N)
    Attendance=Attendance_U + S + A + Country
    
    hands_U=np.random.normal(10, 2, size=N)
    hands_raised=hands_U+ 2*S+ 3*A + Country
    
    discussion_U=np.random.normal(10, 2, size=N)
    discussion=discussion_U+ 2*Attendance + 2*S+ 3*A + Country
    
    assignment_U=np.random.normal(10, 2, size=N)
    assignment=assignment_U+ 2*S+ 3*A + Country -3*Attendance
    
    announcement_U=np.random.normal(10, 2, size=N)
    announcement=announcement_U+ 2*S+ 3*A + Country
    
    
    grade_U=np.random.normal(10, 2, size=N)
    grade=grade_U+ hands_raised + 2*discussion + assignment + announcement
    
    
      
    df=pd.DataFrame(S,columns=['S'])
    df_U=pd.DataFrame(S,columns=['S'])
    
    df['A']=A
    df['S']=S
    df['Country']=Country
    df['hands_raised']=hands_raised
    df['Attendance']=Attendance
    df['discussion']=discussion
    df['assignment']=assignment
    df['announcement']=announcement
    df['grade']=grade
    
    
    
    df_U['hands']=hands_U
    df_U['Attendance']=Attendance_U
    df_U['discussion']=discussion_U
    df_U['assignment']=assignment_U
    df_U['announcement']=announcement_U
    df_U['grade']=grade_U
    
    
    return (df,df_U)





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

    if 'Country' in dolst.keys():
        Country=np.array([dolst['Country']]*df.shape[0])#Construct a list of same size as df
    else:
        Country=df['Country']

    
    if 'hands_raised' in dolst.keys():
        hands_raised= np.array([dolst['hands_raised']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys() or 'Country' in dolst.keys() :
        hands_raised= df_U['hands_raised']+2*S+ 3*A + Country
    else:
        hands_raised=df['hands_raised']
    
    if 'Attendance' in dolst.keys():
        Attendance= np.array([dolst['Attendance']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys() or 'Country' in dolst.keys() :
        Attendance= df_U['Attendance']+S + A + Country
    else:
        Attendance=df['Attendance']

    #discussion_U+ hands_raised/2 + 2*S+ 3*A + Country
    if 'discussion' in dolst.keys():
        discussion= np.array([dolst['discussion']]*df.shape[0])
    elif 'Attendance' in dolst.keys() or 'A' in dolst.keys() or 'S' in dolst.keys() or 'Country' in dolst.keys() :
        discussion= df_U['discussion']+2*Attendance + 2*S+ 3*A + Country
    else:
        discussion=df['discussion']

   
    if 'assignment' in dolst.keys():
        assignment= np.array([dolst['assignment']]*df.shape[0])
    elif 'Attendance' in dolst.keys() or 'A' in dolst.keys() or 'S' in dolst.keys() or 'Country' in dolst.keys() :
        assignment= df_U['assignment']+2*S+ 3*A + Country-3*Attendance
    else:
        assignment=df['assignment']
 
    if 'announcement' in dolst.keys():
        announcement= np.array([dolst['announcement']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys() or 'Country' in dolst.keys() :
        announcement= df_U['announcement']+2*S+ 3*A + Country
    else:
        announcement=df['announcement']
 
    
    grade=df_U['grade']+ hands_raised + 2*discussion + assignment + announcement
    
    
    df=pd.DataFrame(np.array(list(S)),columns=['S'])
    

    df['A']=np.array(list(A))
    #print(df)
    df['S']=np.array(list(S))
    df['Country']=np.array(list(Country))
    df['hands_raised']=np.array(list(hands_raised))
    df['Attendance']=np.array(list(Attendance))
    df['assignment']=np.array(list(assignment))
    df['discussion']=np.array(list(discussion))
    df['announcement']=np.array(list(announcement))
    df['grade']=np.array(list(grade))
    
    return df
    #For each variable in dolst, change their descendants
        
#what fraction are married if age > X or edu=masters/bachelors?
#what fraction are edu> bachelors if age > X?
#


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
            i+=1 


# In[ ]:





# In[199]:
scores={}
times={}
opt_times={}
opt_scores={}
for num_var in [5,6,7,8,9,10]:#[2]:#[10000,100000,200000,400000,600000,800000,1000000]:
    i=0
    lst=[]
    while i<5:
        lst.append(i)
        i+=1
    
    (df,df_U)=get_data_vary_var(10000,0,num_var)

    print (df)
    A=[]#['St_orig','sav_orig','hous_orig','Cred_orig']#,'housing','employment']
    aval=[1,0,0]

    klst=['A','S']
    kval=[]

    target='grade'
    start=time.time()

    Adomain=[]
    t=0
    while t<num_var:
        A.append("var"+str(t))
        Adomain.append(lst)
        t+=1
    conditionals=copy.deepcopy(A)
    conditionals.extend(klst)
    #print (conditionals)
    (beta_lst,[[beta0]])=get_logistic_param(df,conditionals,[target],[1])
    print(beta_lst,beta0)

    scores[num_var]=optimization(df,A,aval,Adomain,klst,kval,0.8,beta_lst,beta0)
    end=time.time()
    times[num_var]=end-start

    start=time.time()
    domain_lst=get_combination(Adomain,[])
    maxval=0
    for lst in domain_lst:
        print(lst)
        #(beta_lst,[[beta0]])=get_logistic_param(df,conditionals,[target],[1])
        print (conditionals,beta_lst)
        t=0
        curr=0
        while t<num_var:
            print (t,beta_lst,Adomain,lst)
            curr+=beta_lst[t]*((Adomain[t][lst[0]-1]+Adomain[t][lst[0]])/2)
            t+=1
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
        if sum_backdoor+curr> maxval:
            maxval=sum_backdoor+curr
        print (sum_backdoor+curr+beta0)
     
    end=time.time()
    opt_times[num_var]=end-start
    opt_scores[num_var]=maxval
    print (end-start)
#(A,aval,Adomain,klst,kval,alpha,betalst,beta0):

print(times)
print(opt_times)
# In[ ]:

our=[]
opt=[]
for i in [5,6,7,8,9,10]:
    our.append(times[i])
    opt.append(opt_times[i])

from matplotlib import pyplot as plt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=[5,6,7,8,9,10]
    
import pylab as plot
params = {'legend.fontsize': 65,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : 65}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = snname=['Sex','Age','Status','Saving','Housing']

x = np.arange(len(labels))  # the label locations
width = 10  # the width of the bars

plt.figure(figsize=(20, 20)) # in inches!

#fig, ax = plt.subplots()

#rects1 = ax.barh(x - width, trsn, width,xerr=trsnvar, label='Ground Truth', color='coral', edgecolor='black', hatch="/",error_kw=dict(elinewidth=5, ecolor='black'))
#rects2 = ax.barh(x, sn, width, xerr=snvar,label='RAVEN', color='forestgreen', edgecolor='black', hatch="||",error_kw=dict(elinewidth=5, ecolor='black'))
#rects3 = ax.barh(x + width, allScores['sn'], width, label='NeSuf', color='royalblue')


y = np.array(our)
x = np.array([5,6,7,8,9,10])
#error = np.array([0.02,0.003,0.0023,0.0001,0.001])#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)


y1 = np.array(opt)
#error1 = np.array(snvar)#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)
#print(error1)

plt.plot(x, y, 'k-v',label='HypeR-sampled',linewidth=5,color='black',markersize=40)
plt.plot(x, y1, 'k-s',label='Opt-HowTo',linewidth=5,color='forestgreen',markersize=40)
plt.fill_between(x, y, y, alpha=0.3)
plt.fill_between(x, y1, y1, alpha=0.15,color='forestgreen')
#plt.show()
#plt.xticks([0, 1000,50000, 100000], ['', '1K','50K', '100K'])
#plot.ylim([0.4,0.7])
plt.legend()
plt.xlabel('Attributes',labelpad=30)
plt.ylabel('Time (in seconds)',labelpad=0)
plt.savefig('../freshRuns/12b.pdf')
