#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
from collections import namedtuple, Counter

import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# In[53]:


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





# In[54]:


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
        df["var"+str(i)] = roundlst(np.random.normal(0, 2, size=N) + np.random.rand()*Attendance)
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



# In[55]:


(df,df_U)=get_data(10000,0)
feat=list(df.columns)


# In[56]:


(new_df,newdf_U)=get_data_vary_var(10000,0,3)


# In[57]:


df


# In[58]:


new_df


# In[59]:


feat


# In[60]:


df['hands_raised'].describe()


# In[61]:


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
    regr.fit(X, new_lst)
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
    regr = RandomForestRegressor(random_state=0)
    #regr = LogisticRegression(random_state=0)
    regr.fit(X, new_lst)
    #print (regr.coef_.tolist())
    #print (regr.predict_proba([conditional_values]),"ASDFDS")
    return (regr.predict([conditional_values])[0])
    #return(regr.predict_proba([conditional_values])[0][1])
  



# In[62]:


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


# In[68]:


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
            return get_prob_o_regression(df,Ac,c,postlst,postvallst)
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
        
        
        
        


# In[69]:


import time
scores=[]
times=[]
backdoor={'S':[],'A':[],'Country':[],'hands_raised':['S','A','Country'],'Attendance':['S','A','Country'],'discussion':['S','A','Country','Attendance'],'assignment':['S','A','Country'],
         'announcement':['S','A','Country']}
    
for num_var in [0,3,5,8,10]:
    (new_df,newdf_U)=get_data_vary_var(10000,0,num_var)
    i=0
    prelst=[]
    prevallst=[]
    while i<num_var:
        #backdoor['hands_raised'].append("var"+str(i))
        prelst.append("var"+str(i))
        prevallst.append(0)
        i+=1
    start=time.time()
    scores.append (get_query_output(new_df,'avg','grade',prelst,prevallst,[],[],['hands_raised'],[1],['*'],'',{}))
    end=time.time()
    times.append(end-start)


# In[65]:


print(scores)


# In[66]:


times


# In[71]:


indep_times=[]
backdoor={'S':[],'A':[],'Country':[],'hands_raised':[],'Attendance':[],'discussion':[],'assignment':[],
         'announcement':[]}
    
for num_var in [0,3,5,8,10]:
    (new_df,newdf_U)=get_data_vary_var(10000,0,num_var)
    i=0
    prelst=[]
    prevallst=[]
    while i<num_var:
        #backdoor['hands_raised'].append("var"+str(i))
        prelst.append("var"+str(i))
        prevallst.append(0)
        i+=1
    start=time.time()
    scores.append (get_query_output(new_df,'avg','grade',prelst,prevallst,[],[],['hands_raised'],[1],['*'],'',{}))
    end=time.time()
    indep_times.append(end-start)


# In[72]:


indep_times


# In[73]:


times


# In[74]:


from matplotlib import pyplot as plt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
x=[0,3,5,8,10]
our=times
opt=indep_times
    
import pylab as plot
params = {'legend.fontsize': 65,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : 65}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = snname=['Sex','Age','Status','Saving','Housing']

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

plt.figure(figsize=(20, 20)) # in inches!



y = np.array(our)
x = np.array([0,3,5,8,10])

y1 = np.array(opt)

plt.plot(x, y, 'k-v',label='HypeR-sampled',linewidth=5,color='black',markersize=40)
plt.plot(x, y1, 'k-s',label='Indep',linewidth=5,color='forestgreen',markersize=40)
plt.fill_between(x, y, y, alpha=0.3)
plt.fill_between(x, y1, y1, alpha=0.15,color='forestgreen')
#plt.show()
#plt.xticks([0, 1000,50000, 100000], ['', '1K','50K', '100K'])
#plot.ylim([0.4,0.7])
plt.legend()
plt.xlabel('Attributes',labelpad=30)
plt.ylabel('Time (in seconds)',labelpad=0)
plt.savefig('../freshRuns/12a.pdf')


