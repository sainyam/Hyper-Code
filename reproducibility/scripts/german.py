import pandas as pd
import numpy as np

def read_data():
    df=pd.read_csv('./datasets/german.csv')
    l=[]
    for v in df['credit'].values:
        if v==2.0:
            l.append(0)
        else:
            l.append(1)
    df['credit']=l

    orig_df=df


    featlst=list(df.columns)
    featlst.remove('credit')


    for feat in featlst:
        highest=max(df[feat])
        lowest=min(df[feat])
        #if len(list(set(df[feat])))>4:
        #    print (feat,len(list(set(df[feat]))))
        l=df[feat]
        if feat=='month':
            processed=[]
            for v in l:
                if v<=10:
                    processed.append(0)
                elif v<15:
                    processed.append(1)
                elif v<25:
                    processed.append(2)
                else:
                    processed.append(3)
            df[feat]=processed
        if feat=='housing':
            print (feat)
            processed=[]
            for v in l:
                if v==1:
                    processed.append(1)
                else:
                    processed.append(0)
            df[feat]=processed
        
        if feat=='installment_plans' or feat=='number_of_credits':
            print (feat)
            processed=[]
            for v in l:
                if v<=1:
                    processed.append(0)
                else:
                    processed.append(1)
            df[feat]=processed
        if feat=='employment' or feat=='credit_history' or feat=='skill_level':
            print (feat)
            processed=[]
            for v in l:
                if v<=1:
                    processed.append(0)
                else:
                    processed.append(v)
            df[feat]=processed
            
        if feat=='credit_amount':
            processed=[]
            for v in l:
                if v<=1500:
                    processed.append(0)
                elif v<3000:
                    processed.append(1)
                elif v<5500:
                    processed.append(2)
                else:
                    processed.append(3)
            df[feat]=processed
        if feat=='purpose':
            l=[]
            for v in df['purpose']:
                if v>4:
                    l.append(5)
                elif v<=2:
                    l.append(2)
                else:
                    l.append(v)
            df[feat]=l
        if feat=='other_debtors':
            processed=[]
            for v in l:
                if v<=0:
                    processed.append(0)
                else:
                    processed.append(1)
            df[feat]=processed
        if feat=='savings':
            processed=[]
            for v in l:
                if v<=0:
                    processed.append(0)
                elif v<=3:
                    processed.append(1)
                else:
                    processed.append(2)
            df[feat]=processed
        #    #pd.cut(l)
        #    range_lst=pd.cut(l,4,labels=[0,1,2,3])
        #    df[feat]=(pd.cut(l,4,labels=[0,1,2,3]))



    return df



    # In[3]:
