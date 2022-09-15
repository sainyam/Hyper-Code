#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import namedtuple, Counter

import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# In[2]:


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


# In[ ]:





# In[3]:


def get_data(N,seed):
    #N=10000
    #np.random.seed(seed)
    
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
    
    
    df_U['A']=A
    df_U['S']=S
    df_U['Country']=Country
    df_U['hands_raised']=hands_U
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



# In[ ]:





# In[ ]:





# In[34]:


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
        #print (lst,df[Cvar].value_counts())
        lst.append(list(set(list(df[Cvar]))))
        #print (C,lst,Cvar,df[Cvar].value_counts())
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
    regr.fit(X.values, df[AT])
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
  



# In[35]:


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


# In[40]:


def get_query_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c,g_Ac_lst,interference, blocks):
    #interference is set of attributes of other tuples in a block that affect current tuple's attribute
    #blocks are list of lists
    
    #Identify all attributes which are used for regression and add as columns 
    
    #New columns will be 
    new_df=[]
    newvars=[]
    for var in Ac:
        if var in interference:
            newvars.extend(backdoor[var])
        else:
            newvars.extend([var])
        
    
    for index,row in df.iterrows():
        if index in blocks.keys():
            sameblockelem=blocks[index]
        else:
            sameblockelem=[]
        
        #identify backdoor from other records and add those
        #In interference is not in Ac then interference is enough
        #else we need backdoor of intervened and interference    
        
        for var in newvars:
            iter=0
            #print(var,newvars)
            if len(sameblockelem)>0:
                row["new"+var]=0
            for sibling in sameblockelem:
                row["new"+var]+=df.iloc[sibling][var]
                #assumes A_c is a unique attribute. TODO: change to consider list
                if sibling in g_Ac_lst:
                    row[str(iter)+Ac] = c
                else:
                    row[str(iter)+Ac] = df.iloc[sibling][Ac]
                iter+=1
            if len(sameblockelem) >0:
                row["new"+var]/=len(sameblockelem)
                #We can add interference for all in blocks too
        new_df.append(row)
            
    
    df=pd.DataFrame(new_df)
    #print(df)
    '''
    
    sub_intervene=[]
    sub_df=[]
    for index,row in df.iterrows():
        iter=0
        for attr in prelst:
            if row[attr] == prevallst[iter]:
                continue
            else:
                break
            iter+=1
        if iter==len(prevallst):
            sub_df.append(row)
        if check_g_Ac(row,g_Ac_lst):
            sub_intervene.append(row)
    '''
            
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

            print ("conditioning set",conditioning_set,conditioning_val)
            print("post condition",postlst,postvallst)
            if iter==0:
                regr=train_regression(df,conditioning_set,conditioning_val,postlst,postvallst)
            pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
            print("this",prelst,prevallst,backdoorlst,backdoorvallst)
            pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
            print (pogivenck,pcgivenk)
            total_prob+=pogivenck * pcgivenk
            iter+=1
            
        print("final prob is ",total_prob)
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
            print(backdoorlst,backdoorvals)
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

            print ("conditioning set",conditioning_set,conditioning_val, AT)
            if iter==0:
                regr=train_regression_raw(df,conditioning_set,conditioning_val,AT)
            pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
            print("this",prelst,prevallst,backdoorlst,backdoorvallst)
            pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
            print (pogivenck,pcgivenk)
            total_prob+=pogivenck * pcgivenk
            iter+=1
            
        print("final prob is ",total_prob)
        return total_prob
        
        
        
        


# In[ ]:





# In[ ]:





# In[53]:


(df,df_U)=get_data(1000000,0)
for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    df[col+'_disc']=pd.cut(df[col],
           bins=[df[col].describe()['min']-1, df[col].describe()['50%'], df[col].describe()['max']], 
           labels=[0, 1])
feat=list(df.columns)
backdoor={'S':[],'A':[],'Country':[],'hands_raised':['S','A','Country'],'Attendance':['S','A','Country'],'discussion':['S','A','Country','Attendance'],'assignment':['S','A','Country'],
         'announcement':['S','A','Country']}

sampled_df=df.sample(n=10000,random_state=0)
sampled_scores=[]

for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    sampled_scores.append(get_query_output(sampled_df,'avg','grade',[],[],[],[],[col],[1],['*'],'',{}))


# In[54]:


df.isna()


# In[55]:


backdoor={'S':[],'A':[],'Country':[],'hands_raised':['S','A','Country','assignment_disc','Attendance_disc','announcement_disc','discussion_disc'],'Attendance':['S','A','Country','assignment_disc','announcement_disc','hands_raised_disc','discussion_disc'],'discussion':['S','A','Country','Attendance_disc','assignment_disc','Attendance_disc','announcement_disc','hands_raised_disc'],'assignment':['S','A','Country','Attendance_disc','announcement_disc','hands_raised_disc','discussion_disc'],
         'announcement':['S','A','Country','assignment_disc','Attendance_disc','hands_raised_disc','discussion_disc']}

sampled_scores_nb=[]

for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    print (col)
    sampled_scores_nb.append(get_query_output(sampled_df,'avg','grade',[],[],[],[],[col],[1],['*'],'',{}))


# In[ ]:





# In[35]:


df


# In[ ]:





# In[56]:


sampled_scores_nb


# In[57]:


sampled_scores


# In[59]:


gt=[]
for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    intervened=causal_effect({col:1},df,df_U)
    gt.append(intervened['grade'].describe()['mean'])


# In[ ]:





# In[58]:


backdoor={'S':[],'A':[],'Country':[],'hands_raised':[],'Attendance':[],'discussion':[],'assignment':[],
         'announcement':[]}

indep=[]

for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    indep.append(get_query_output(sampled_df,'avg','grade',[],[],[],[],[col],[1],['*'],'',{}))


# In[60]:


indep


# In[61]:


import pandas as pd
from functools import reduce
import seaborn as scs
hyper=sampled_scores

hypername=['Assignment', 'Attendance', 'Announcement','Hand Raised', 'Discussion']

gt=gt
gtname=['Assignment', 'Attendance', 'Announcement','Hand Raised', 'Discussion']

#gt=[0.99,0.6298,0.507,0.5007,0.02]
#gtname=[ 'Age',  'Saving','Status','Sex', 'Housing']


baseline=indep
baselinename=['Assignment', 'Attendance', 'Announcement','Hand Raised', 'Discussion']


hyper_all=sampled_scores_nb
'''
normalized_shapley=[]
normalized_feat=[]
normalized_sn=[]
normalized_tr=[]
i=0
for feat in snname:

    normalized_shapley.append(shapley[shapley_name.index(feat)]*1.0/max(shapley))
    normalized_feat.append(featval[feat_name.index(feat)]*1.0/max(featval))
    normalized_sn.append(sn[i]*1.0/max(sn))
    normalized_tr.append(tr[i]*1.0/max(tr))
    i+=1

import scipy.stats as ss
shapley_rank=ss.rankdata(normalized_shapley)
feat_rank=ss.rankdata(normalized_feat)
sn_rank=ss.rankdata(normalized_sn)
tr_rank=ss.rankdata(normalized_tr)
'''
num_feat=5
'''
normalized_shapley=normalized_shapley[:num_feat]
normalized_feat=normalized_feat[:num_feat]
normalized_sn=normalized_sn[:num_feat]
normalized_tr=normalized_tr[:num_feat]
snname=snname[:num_feat]

featnames = snname
featnames=featnames[:num_feat]
'''


# In[62]:


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
rects1 = ax.barh(x - 0.15, gt, width, label='Ground Truth', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.barh(x, hyper, width, label='Hyper-sampled', color='gainsboro', edgecolor='black', hatch="\\\\")
rects3 = ax.barh(x + 0.15, hyper_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
rects4 = ax.barh(x + 0.3, baseline, width, label='Indep', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Query Output', fontsize=fsize, labelpad=fsize/2)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=fsize)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xticks(np.arange(0, 150, 50))
ax.legend(loc=(0.384,0))
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

'''
autolabel(rects1,shapley_rank)
autolabel(rects2,feat_rank)
autolabel(rects3,sn_rank)
autolabel(rects4,tr_rank)
'''
plt.xlim([0, 120])
ax.margins(0.1,0.05)
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('../freshRuns/10b.pdf')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




