import pandas as pd
import numpy as np

def read_data():
    df=pd.read_csv('./datasets/adult.txt',delimiter=' ')




    from sklearn.utils import resample


    backdoor={'Age':[],'sex':[],'country':[],'marital':['country','Age','sex'],'edu':['Age','sex','country'],
              'class':['Age','sex','country'],'occupation':['Age','sex','country'],
              'hours':['Age','sex','country']}

    samp=df['marital']



    y_train=df['target']

    col=list(df.columns)
    col.remove('target')
    col.remove('Unnamed: 15')

    col=list(backdoor.keys())
    df['sex']=1-df['sex']



    featlst=list(df.columns)
    featlst.remove('target')


    for feat in col:
        highest=max(df[feat])
        lowest=min(df[feat])
        if len(list(set(df[feat])))>4:
            print (feat,len(list(set(df[feat]))))
        l=df[feat]
        if feat=='country':
            proc=[]
            for v in l:
                if v==0:
                    proc.append(v)
                elif v<=4:
                    proc.append(2)
                elif v<=10:
                    proc.append(3)
                else:
                    proc.append(4)
            df[feat]=proc
        elif feat=='Age':
            proc=[]
            for v in l:
                if v<=30:
                    proc.append(1)
                elif v<=40:
                    proc.append(2)
                elif v<=50:
                    proc.append(3)
                else:
                    proc.append(4)
            df[feat]=proc
        elif feat=='marital':
            proc=[]
            for v in l:
                #if v<20:
                #    proc.append(v)
                if v==1:
                    proc.append(1)
                elif  v<=4:
                    proc.append(0)
                else:
                    proc.append(1)
            df[feat]=proc
        elif feat=='edu':
            proc=[]
            for v in l:
                if v<=1:
                    proc.append(v)
                elif v<=4:
                    proc.append(2)
                elif v==5:
                    proc.append(5)
                else:
                    proc.append(6)
            df[feat]=proc
        elif feat=='class':
            proc=[]
            for v in l:
                if v<=2:
                    proc.append(v)
                else:
                    proc.append(6)
            df[feat]=proc
        elif feat=='occupation':
            proc=[]
            for v in l:
                if v<=3:
                    proc.append(1)
                elif v<=5:
                    proc.append(v)
                elif v<=8:
                    proc.append(8)
                else:
                    proc.append(9)
            df[feat]=proc
        elif feat=='hours':
            proc=[]
            for v in l:
                if v<=25:
                    proc.append(1)
                elif v<=41:
                    proc.append(2)
                elif v<=55:
                    proc.append(8)
                else:
                    proc.append(9)
            df[feat]=proc
            
            
    return df