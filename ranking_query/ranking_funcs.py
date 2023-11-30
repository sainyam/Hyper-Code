import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import math as m
from collections import defaultdict

##packages need

def read_data(path):
    
    df=pd.read_csv(path)
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

## this function is from hyper

def get_new_G(G,df):
    """
    G: the causal graph
    df: the dataframe
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    col1s=[]
    col2s=[]
    for u, v in G.edges:
        col1s.append(u)
        col2s.append(v)
    new_G = nx.DiGraph()
    for i in range(len(col1s)):
        X = sm.add_constant(df[col1s[i]])
        model = sm.OLS(df[col2s[i]], X).fit()
        new_G.add_edge(col1s[i], col2s[i], weight=model.params[col1s[i]])
        
    return new_G

def adjust_node(node, delta, df_temp, G, condition_mask, updated_val, is_first_iteration=True):
    """
    node: the updtaed column
    delta: a list of delta between the updated value and the original value
    df_temp:  dataframe
    G: the causal graph
    condition mask: a list of boolean to filter the rows
    updated_val: the updated value which the columns will be setted to
    is_first_iteration: a boolean to make sure only the updated column is to be setted with updated val
    """
    for _, child, data in G.out_edges(node, data=True):
        d_w = data['weight']
        child_delta = [d_w*d for d in delta]
        df_temp.loc[condition_mask, child] += child_delta
        # recursion for children of children
        adjust_node(child, child_delta, df_temp, G, condition_mask, updated_val, is_first_iteration=False)
        
        if is_first_iteration:
            df_temp.loc[condition_mask, node] = updated_val
            is_first_iteration= False
            
    return df_temp

def ranking_query(G, df, k, update_vars, target_column, condition=None):
    """
    G: the causal graph
    df:  dataframe
    k: the top k
    update_vars: the variables need to be updated with the value
    target_column: the column we will rank by
    condition:the condtion to filter rows
    """
    df_temp = df.copy()
    #filter rows with the condtions
    if condition:
        condition_mask = df_temp[list(condition.keys())].eq(pd.Series(condition)).all(axis=1)
    else:
        condition_mask = pd.Series([True]*len(df_temp))

    for var, value in update_vars.items():
        temp_lst=df_temp.loc[condition_mask, var]
        delta = [value - x for x in temp_lst]
        adjust_node(var, delta, df_temp, G, condition_mask,value,True)

    top_k_values = [float('-inf')] * k
    top_k_indices = [-1] * k
    
    # greedy method
    for index, row in df_temp.iterrows():
        min_top_k = min(top_k_values)
        if row[target_column] > min_top_k:
            min_index = top_k_values.index(min_top_k)
            top_k_values[min_index] = row[target_column]
            top_k_indices[min_index] = index

    return df_temp.loc[top_k_indices]

def adjust_node_multi(node, delta, df_temp, G, condition_mask, updated_val, is_first_iteration=True):
    """
    node: the updtaed column
    delta: a list of delta between the updated value and the original value
    df_temp:  dataframe
    G: the causal graph
    condition mask: a list of boolean to filter the rows
    updated_val: the updated value which the columns will be setted to
    is_first_iteration: a boolean to make sure only the updated column is to be setted with updated val
    """
    for _, child, data in G.out_edges(node, data=True):
        d_w = data['weight']
        child_delta = [d_w*d for d in delta]
        df_temp.loc[condition_mask, child] += child_delta
        # recursion for children of children
        adjust_node(child, child_delta, df_temp, G, condition_mask, updated_val, is_first_iteration=False)
        
        if is_first_iteration:
            df_temp.loc[condition_mask, node]*= updated_val
            is_first_iteration= False
            
    return df_temp

def ranking_query_multi(G, df, k, update_vars, target_column, condition=None):
    """
    G: the causal graph
    df:  dataframe
    k: the top k
    update_vars: the variables need to be updated with the value
    target_column: the column we will rank by
    condition:the condtion to filter rows
    """
    df_temp = df.copy()
    #filter rows with the condtions
    if condition:
        condition_mask = df_temp[list(condition.keys())].eq(pd.Series(condition)).all(axis=1)
    else:
        condition_mask = pd.Series([True]*len(df_temp))

    for var, factor in update_vars.items():
        temp_lst=df_temp.loc[condition_mask, var]
        delta = [factor*x - x for x in temp_lst]
        adjust_node_multi(var, delta, df_temp, G, condition_mask,factor,True)

    top_k_values = [float('-inf')] * k
    top_k_indices = [-1] * k
    
    # greedy method
    for index, row in df_temp.iterrows():
        min_top_k = min(top_k_values)
        if row[target_column] > min_top_k:
            min_index = top_k_values.index(min_top_k)
            top_k_values[min_index] = row[target_column]
            top_k_indices[min_index] = index

    return df_temp.loc[top_k_indices]


def adjust_node_add(node, delta, df_temp, G, condition_mask, updated_val, is_first_iteration=True):
    """
    node: the updtaed column
    delta: a list of delta between the updated value and the original value
    df_temp:  dataframe
    G: the causal graph
    condition mask: a list of boolean to filter the rows
    updated_val: the updated value which the columns will be setted to
    is_first_iteration: a boolean to make sure only the updated column is to be setted with updated val
    """
    for _, child, data in G.out_edges(node, data=True):
        d_w = data['weight']
        child_delta = [d_w*d for d in delta]
        df_temp.loc[condition_mask, child] += child_delta
        # recursion for children of children
        adjust_node(child, child_delta, df_temp, G, condition_mask, updated_val, is_first_iteration=False)
        
        if is_first_iteration:
            df_temp.loc[condition_mask, node]+= updated_val
            is_first_iteration= False
            
    return df_temp

def ranking_query_add(G, df, k, update_vars, target_column, condition=None):
    """
    G: the causal graph
    df:  dataframe
    k: the top k
    update_vars: the variables need to be updated with the value
    target_column: the column we will rank by
    condition:the condtion to filter rows
    """
    df_temp = df.copy()
    #filter rows with the condtions
    if condition:
        condition_mask = df_temp[list(condition.keys())].eq(pd.Series(condition)).all(axis=1)
    else:
        condition_mask = pd.Series([True]*len(df_temp))

    for var, val in update_vars.items():
        temp_lst=df_temp.loc[condition_mask, var]
        delta = [val for _ in temp_lst]
        adjust_node_add(var, delta, df_temp, G, condition_mask,val,True)

    top_k_values = [float('-inf')] * k
    top_k_indices = [-1] * k
    
    # greedy method
    for index, row in df_temp.iterrows():
        min_top_k = min(top_k_values)
        if row[target_column] > min_top_k:
            min_index = top_k_values.index(min_top_k)
            top_k_values[min_index] = row[target_column]
            top_k_indices[min_index] = index

    return df_temp.loc[top_k_indices]


def get_ranking_query(G, df, k, update_vars, target_column, condition=None, opt="fix"):
    
    def divide_values(update_vars):
        return {key: 1/value for key, value in update_vars.items()}

    def negate_values(update_vars):
        return {key: -value for key, value in update_vars.items()}

    options = {
        'fix': lambda: ranking_query(G, df, k, update_vars, target_column, condition),
        'multiply_by': lambda: ranking_query_multi(G, df, k, update_vars, target_column, condition),
        'divided_by': lambda: ranking_query_multi(G, df, k, divide_values(update_vars), target_column, condition),
        'add': lambda: ranking_query_add(G, df, k, update_vars, target_column, condition),
        'subs': lambda: ranking_query_add(G, df, k, negate_values(update_vars), target_column, condition)
    }

    return options.get(opt, lambda: 'Invalid operator, operator must be one of "fix", "multiply_by", "divided_by", "add", and "subs"')()

def ranking_query_prob(G, df, k, update_vars, target_column, condition=None):
    """
    G: the causal graph
    df:  dataframe
    k: the top k
    update_vars: the variables need to be updated with the value
    target_column: the column we will rank by
    condition:the condtion to filter rows
    """
    df_temp = df.copy()
    #filter rows with the condtions
    if condition:
        condition_mask = df_temp[list(condition.keys())].eq(pd.Series(condition)).all(axis=1)
    else:
        condition_mask = pd.Series([True]*len(df_temp))

    for var, value in update_vars.items():
        temp_lst=df_temp.loc[condition_mask, var]
        delta = [value - x for x in temp_lst]
        adjust_node(var, delta, df_temp, G, condition_mask,value,True)
    
    prob_df=comp_rank_k(df_temp, target_column, k)
    
    return prob_df


def ranking_query_prob_multi(G, df, k, update_vars, target_column, condition=None):
    """
    G: the causal graph
    df:  dataframe
    k: the top k
    update_vars: the variables need to be updated with the value
    target_column: the column we will rank by
    condition:the condtion to filter rows
    """
    df_temp = df.copy()
    #filter rows with the condtions
    if condition:
        condition_mask = df_temp[list(condition.keys())].eq(pd.Series(condition)).all(axis=1)
    else:
        condition_mask = pd.Series([True]*len(df_temp))

    for var, factor in update_vars.items():
        temp_lst=df_temp.loc[condition_mask, var]
        delta = [factor*x - x for x in temp_lst]
        adjust_node_multi(var, delta, df_temp, G, condition_mask,factor,True)
    
    prob_df=comp_rank_k(df_temp, target_column, k)
    
    return prob_df

def ranking_query_prob_add(G, df, k, update_vars, target_column, condition=None):
    """
    G: the causal graph
    df:  dataframe
    k: the top k
    update_vars: the variables need to be updated with the value
    target_column: the column we will rank by
    condition:the condtion to filter rows
    """
    df_temp = df.copy()
    #filter rows with the condtions
    if condition:
        condition_mask = df_temp[list(condition.keys())].eq(pd.Series(condition)).all(axis=1)
    else:
        condition_mask = pd.Series([True]*len(df_temp))

    for var, val in update_vars.items():
        temp_lst=df_temp.loc[condition_mask, var]
        delta = [val for _ in temp_lst]
        adjust_node_add(var, delta, df_temp, G, condition_mask,val,True)
    
    prob_df=comp_rank_k(df_temp, target_column, k)
    
    return prob_df


def get_ranking_query_prob(G, df, k, update_vars, target_column, condition=None, opt="fix"):
        
    def divide_values(update_vars):
        return {key: 1/value for key, value in update_vars.items()}

    def negate_values(update_vars):
        return {key: -value for key, value in update_vars.items()}

    options = {
        'fix': lambda: ranking_query_prob(G, df, k, update_vars, target_column, condition),
        'multiply_by': lambda: ranking_query_prob_multi(G, df, k, update_vars, target_column, condition),
        'divided_by': lambda: ranking_query_prob_multi(G, df, k, divide_values(update_vars), target_column, condition),
        'add': lambda: ranking_query_prob_add(G, df, k, update_vars, target_column, condition),
        'subs': lambda: ranking_query_prob_add(G, df, k, negate_values(update_vars), target_column, condition)
    }

    return options.get(opt, lambda: 'Invalid operator, operator must be one of "fix", "multiply_by", "divided_by", "add", and "subs"')()

def stable_ranking_opt(G, df, k, update_vars, target_column, condition=None, max_iter=100):
    i = 0
    rank = ranking_query(G, df, k, update_vars, target_column, condition).index
    x, x_val = next(iter(update_vars.items()))
    x_sd = np.abs(df[x].std() * 0.01)
    x_val_lb = x_val
    upper_bound_changed = False
    lower_bound_changed = False
    x_upper = None
    x_lower = None
    x_lower_iter = None
    x_upper_iter = None

    while i < max_iter and not (upper_bound_changed and lower_bound_changed):
        i += 1
        if not upper_bound_changed:
            x_val += x_sd
            cur_rank_upper = ranking_query(G, df, k, {x: x_val}, target_column, condition).index
            if not np.array_equal(cur_rank_upper, rank):
                x_upper = x_val
                upper_bound_changed = True
                x_upper_iter = i
        
        if not lower_bound_changed:
            x_val_lb -= x_sd
            cur_rank_lower = ranking_query(G, df, k, {x: x_val_lb}, target_column, condition).index
            if not np.array_equal(cur_rank_lower, rank):
                x_lower = x_val_lb 
                lower_bound_changed = True
                x_lower_iter = i

    if upper_bound_changed and lower_bound_changed:
        print(f"Lower Bound Change: (Value: {x_lower}, Iteration: {x_lower_iter})")
        print(f"Upper Bound Change: (Value: {x_upper}, Iteration: {x_upper_iter})")
        return ([[x_lower, x_upper],[x_lower_iter,x_upper_iter]])
    return ([[x_lower, x_upper],[x_lower_iter,x_upper_iter]])

def stable_ranking_opt_multi(G, df, k, update_vars, target_column, condition=None, max_iter=100):
    i = 0
    rank = ranking_query_multi(G, df, k, update_vars, target_column, condition).index
    x, x_val = next(iter(update_vars.items()))
    x_sd = np.abs(df[x].std() * 0.01)+1
    x_val_lb = x_val
    upper_bound_changed = False
    lower_bound_changed = False
    x_upper = None
    x_lower = None
    x_lower_iter = None
    x_upper_iter = None

    while i < max_iter and not (upper_bound_changed and lower_bound_changed):
        i += 1
        if not upper_bound_changed:
            x_val *= x_sd
            cur_rank_upper = ranking_query_multi(G, df, k, {x: x_val}, target_column, condition).index
            if not np.array_equal(cur_rank_upper, rank):
                x_upper = x_val 
                upper_bound_changed = True
                x_upper_iter = i
        
        if not lower_bound_changed:
            x_val_lb /= x_sd
            cur_rank_lower = ranking_query_multi(G, df, k, {x: x_val_lb}, target_column, condition).index
            if not np.array_equal(cur_rank_lower, rank):
                x_lower = x_val_lb 
                lower_bound_changed = True
                x_lower_iter = i

    if upper_bound_changed and lower_bound_changed:
        print(f"Lower Bound Change: (Value: {x_lower}, Iteration: {x_lower_iter})")
        print(f"Upper Bound Change: (Value: {x_upper}, Iteration: {x_upper_iter})")
        return ([[x_lower, x_upper],[x_lower_iter,x_upper_iter]])
    return ([[x_lower, x_upper],[x_lower_iter,x_upper_iter]])


def stable_ranking_opt_add(G, df, k, update_vars, target_column, condition=None, max_iter=100):
    i = 0
    rank = ranking_query_add(G, df, k, update_vars, target_column, condition).index
    x, x_val = next(iter(update_vars.items()))
    x_sd = np.abs(df[x].std() * 0.01)
    x_val_lb = x_val
    upper_bound_changed = False
    lower_bound_changed = False
    x_upper = None
    x_lower = None
    x_lower_iter = None
    x_upper_iter = None

    while i < max_iter and not (upper_bound_changed and lower_bound_changed):
        i += 1
        if not upper_bound_changed:
            x_val += x_sd
            cur_rank_upper = ranking_query_add(G, df, k, {x: x_val}, target_column, condition).index
            if not np.array_equal(cur_rank_upper, rank):
                x_upper = x_val 
                upper_bound_changed = True
                x_upper_iter = i
        
        if not lower_bound_changed:
            x_val_lb -= x_sd
            cur_rank_lower = ranking_query_add(G, df, k, {x: x_val_lb}, target_column, condition).index
            if not np.array_equal(cur_rank_lower, rank):
                x_lower = x_val_lb
                lower_bound_changed = True
                x_lower_iter = i

    if upper_bound_changed and lower_bound_changed:
        print(f"Lower Bound Change: (Value: {x_lower}, Iteration: {x_lower_iter})")
        print(f"Upper Bound Change: (Value: {x_upper}, Iteration: {x_upper_iter})")
        return ([[x_lower, x_upper],[x_lower_iter,x_upper_iter]])
    return ([[x_lower, x_upper],[x_lower_iter,x_upper_iter]])


def get_stable_ranking_opt(G, df, k, update_vars, target_column, condition=None, max_iter=100, opt="fix"):
        
    def divide_values(update_vars):
        return {key: 1/value for key, value in update_vars.items()}

    def negate_values(update_vars):
        return {key: -value for key, value in update_vars.items()}

    options = {
        'fix': lambda: stable_ranking_opt(G, df, k, update_vars, target_column, condition,max_iter),
        'multiply_by': lambda: stable_ranking_opt_multi(G, df, k, update_vars, target_column, condition,max_iter),
        'divided_by': lambda: stable_ranking_opt_multi(G, df, k, divide_values(update_vars), target_column, condition,max_iter),
        'add': lambda: stable_ranking_opt_add(G, df, k, update_vars, target_column, condition,max_iter),
        'subs': lambda: stable_ranking_opt_add(G, df, k, negate_values(update_vars), target_column, condition,max_iter)
    }

    return options.get(opt, lambda: 'Invalid operator, operator must be one of "fix", "multiply_by", "divided_by", "add", and "subs"')()


def test_revert_ranking_rec(G, df, k, update_vars, target_column, condition=None, prev_results=None,
                            max_iter=1000,num_iter=0,max_num_iter=1,bond_check='uper'):
    if num_iter>max_num_iter:
        return("the iteration ends")
    i=0
    rank = ranking_query(G, df, k, update_vars, target_column, condition).index
    x, x_val = next(iter(update_vars.items()))
    x_sd = np.abs(df[x].std() * 0.01)
    x_val_upper=prev_results[0][1]
    x_upper_iter=prev_results[1][1]
    x_val_lower=prev_results[0][0]
    x_lower_iter=prev_results[1][0]
    upper_bound_changed = False
    lower_bound_changed = False
    x_pre=None
    
        
    while i<max_iter:
        i+=1
        if x_val_upper is not None and bond_check=='uper' and upper_bound_changed is False:
            x_val_upper += x_sd
            cur_rank_upper = ranking_query(G, df, k, {x: x_val_upper}, target_column, condition).index
            if np.array_equal(cur_rank_upper, rank):
                upper_bound_changed = True
                print([x_val_upper,x_upper_iter+i,"update upper"])
                x_pre=x_val_upper

            
        if x_val_lower is not None and bond_check=='lower' and lower_bound_changed is False:
            x_val_lower -= x_sd
            cur_rank_upper = ranking_query(G, df, k, {x: x_val_lower}, target_column, condition).index
            if np.array_equal(cur_rank_upper, rank):
                lower_bound_changed=True
                print([x_val_lower,x_lower_iter+i,"update lower"])
                x_pre=x_val_lower
    
    if x_pre is not None:
                print(x_pre)
                prev_results=stable_ranking_opt(G, df, k, {x: x_pre}, target_column, condition, max_iter)
    else:
        print("no updtaes,increase the max iteration by 1000") 
        max_iter+=1000
    num_iter+=1
    return(test_revert_ranking_rec(G, df, k, update_vars, target_column, condition, prev_results,max_iter,num_iter,max_num_iter,bond_check))


def test_revert_ranking_rec_multi(G, df, k, update_vars, target_column, condition=None, prev_results=None,
                            max_iter=1000,num_iter=0,max_num_iter=1,bond_check='uper'):
    if num_iter>max_num_iter:
        return("the iteration ends")
    i=0
    rank = ranking_query_multi(G, df, k, update_vars, target_column, condition).index
    x, x_val = next(iter(update_vars.items()))
    x_sd = np.abs(df[x].std() * 0.01)+1
    x_val_upper=prev_results[0][1]
    x_upper_iter=prev_results[1][1]
    x_val_lower=prev_results[0][0]
    x_lower_iter=prev_results[1][0]
    upper_bound_changed = False
    lower_bound_changed = False
    x_pre=None
    
        
    while i<max_iter:
        i+=1
        if x_val_upper is not None and bond_check=='uper' and upper_bound_changed is False:
            x_val_upper *= x_sd
            cur_rank_upper = ranking_query_multi(G, df, k, {x: x_val_upper}, target_column, condition).index
            if np.array_equal(cur_rank_upper, rank):
                upper_bound_changed = True
                print([x_val_upper,x_upper_iter+i,"update upper"])
                x_pre=x_val_upper

            
        if x_val_lower is not None and bond_check=='lower' and lower_bound_changed is False:
            x_val_lower /= x_sd
            cur_rank_upper = ranking_query_multi(G, df, k, {x: x_val_lower}, target_column, condition).index
            if np.array_equal(cur_rank_upper, rank):
                lower_bound_changed=True
                print([x_val_lower,x_lower_iter+i,"update lower"])
                x_pre=x_val_lower
    
    if x_pre is not None:
                print(x_pre)
                prev_results=stable_ranking_opt_multi(G, df, k, {x: x_pre}, target_column, condition, max_iter)
    else:
        print("no updtaes,increase the max iteration by 1000") 
        max_iter+=1000
    num_iter+=1
    return(test_revert_ranking_rec_multi(G, df, k, update_vars, target_column, condition, prev_results,max_iter,num_iter,max_num_iter,bond_check))


def test_revert_ranking_rec_add(G, df, k, update_vars, target_column, condition=None, prev_results=None,
                            max_iter=1000,num_iter=0,max_num_iter=1,bond_check='uper'):
    if num_iter>max_num_iter:
        return("the iteration ends")
    i=0
    rank = ranking_query(G, df, k, update_vars, target_column, condition).index
    x, x_val = next(iter(update_vars.items()))
    x_sd = np.abs(df[x].std() * 0.01)
    x_val_upper=prev_results[0][1]
    x_upper_iter=prev_results[1][1]
    x_val_lower=prev_results[0][0]
    x_lower_iter=prev_results[1][0]
    upper_bound_changed = False
    lower_bound_changed = False
    x_pre=None
    
        
    while i<max_iter:
        i+=1
        if x_val_upper is not None and bond_check=='uper' and upper_bound_changed is False:
            x_val_upper += x_sd
            cur_rank_upper = ranking_query_add(G, df, k, {x: x_val_upper}, target_column, condition).index
            if np.array_equal(cur_rank_upper, rank):
                upper_bound_changed = True
                print([x_val_upper,x_upper_iter+i,"update upper"])
                x_pre=x_val_upper

            
        if x_val_lower is not None and bond_check=='lower' and lower_bound_changed is False:
            x_val_lower -= x_sd
            cur_rank_upper = ranking_query_add(G, df, k, {x: x_val_lower}, target_column, condition).index
            if np.array_equal(cur_rank_upper, rank):
                lower_bound_changed=True
                print([x_val_lower,x_lower_iter+i,"update lower"])
                x_pre=x_val_lower
    
    if x_pre is not None:
                print(x_pre)
                prev_results=stable_ranking_opt_add(G, df, k, {x: x_pre}, target_column, condition, max_iter)
    else:
        print("no updtaes,increase the max iteration by 1000") 
        max_iter+=1000
    num_iter+=1
    return(test_revert_ranking_rec_add(G, df, k, update_vars, target_column, condition, prev_results,max_iter,num_iter,max_num_iter,bond_check))


def get_test_revert_ranking_rec(G, df, k, update_vars, target_column, condition=None, prev_results=None,
                            max_iter=1000,num_iter=0,max_num_iter=1,bond_check='uper',opt='fix'):
            
    def divide_values(update_vars):
        return {key: 1/value for key, value in update_vars.items()}

    def negate_values(update_vars):
        return {key: -value for key, value in update_vars.items()}

    options = {
        'fix': lambda: test_revert_ranking_rec(G, df, k, update_vars, target_column, condition, prev_results,
                            max_iter,num_iter,max_num_iter,bond_check),
        'multiply_by': lambda: test_revert_ranking_rec_multi(G, df, k, update_vars, target_column, condition, prev_results,
                            max_iter,num_iter,max_num_iter,bond_check),
        'divided_by': lambda: test_revert_ranking_rec_multi(G, df, k, update_vars, target_column, condition, prev_results,
                            max_iter,num_iter,max_num_iter,bond_check),
        'add': lambda: test_revert_ranking_rec_add(G, df, k, update_vars, target_column, condition, prev_results,
                            max_iter,num_iter,max_num_iter,bond_check),
        'subs': lambda: test_revert_ranking_rec_add(G, df, k, update_vars, target_column, condition, prev_results,
                            max_iter,num_iter,max_num_iter,bond_check)}

    return options.get(opt, lambda: 'Invalid operator, operator must be one of "fix", "multiply_by", "divided_by", "add", and "subs"')()

def backdoor_adjustment(df, Y, y, A, a, Z):
    """
    Compute p(Y = y | do(A) = a) using the backdoor criterion.
    
    Args:
    - df: DataFrame with data.
    - Y: Outcome variable name.
    - y: Specific value of Y.
    - A: Intervention variable name.
    - a: Specific value of A for the intervention.
    - Z: List of variables that satisfy the backdoor criterion.
    
    Returns:
    - Probability p(Y = y | do(A) = a).
    """
    
    prob = 0
    for z_values in df[Z].drop_duplicates().values:
        
        z_values_tuple = tuple(z_values)
        z_mask = (df[Z].apply(tuple, axis=1) == z_values_tuple)
        num = len(df[z_mask & (df[A] == a) & (df[Y] == y)])
        den = len(df[z_mask & (df[A] == a)])
        p_Y_given_A_Z = num / den if den != 0 else 0
        p_Z = len(df[z_mask]) / len(df)
        
        if den != 0:
            p_Y_given_A_Z = num / den 
            p_Z = len(df[z_mask]) / len(df)
            prob += p_Y_given_A_Z * p_Z

    return prob

def find_backdoor_sets(G, nodes):
    """
    G: causal graph
    nodes: the nodes in casaul graph
    """
    backdoor_sets = defaultdict(list)
    for node in nodes:
        for adj_node in G.predecessors(node):
            backdoor_sets[node].append(adj_node)
    return backdoor_sets


def get_prob_backdoor(df,G,y):
    """
    df: dataframe
    G: causal graph
    y: the target node
    """
    backdoor_sets = find_backdoor_sets(G, G.nodes())
    for node, bd_set in backdoor_sets.items():
        if node != y:
            dom_y = df[y].unique()
            dom_node = df[node].unique()
            for d_y in dom_y:
                for d_n in dom_node:
                    adjusted_prob = backdoor_adjustment(df, y, d_y, node, d_n, bd_set)
                    print(f"P({y} = {d_y}|{node} = {d_n} & {bd_set}: {adjusted_prob})")
                    


def comp_rank_k(df, y, k):
    """
    df: dataframe
    y: the target node
    k: numbers of element in a comparision,[e1,e2,...ek]
    """
    length = len(df.index)
    scores = df[y].tolist()
    prob_df = pd.DataFrame(0, index=df.index, columns=range(1, k+1))
    prob_df['row_index'] = prob_df.index
    cols = prob_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    prob_df = prob_df[cols]
    
    for rank in range(1, k+1):
        for i, score in enumerate(scores):
            higher_counts = sum(1 for s in scores if s > score)
            same_counts = sum(1 for s in scores if s == score)
            if higher_counts < rank <= higher_counts + same_counts:
                prob_df.loc[i, rank] = 1 / same_counts
    
    return prob_df


def filter_prob_df(df):
    """
    df: a probility dataframe generated by ranking_query_prob
    """
    df_edit = df.columns[1:]
    filtered_df = df.loc[df[df_edit].apply(lambda x: any(v != 0 for v in x), axis=1)]
    return filtered_df


def calc_prob(df,n):
    """
    df: a probility dataframe generated by ranking_query_prob or filter_prob_df
    n: top n comparision ranking like for n=3 is like [1,2,3,4,5],[2,5,6,3,1],[1,3,6,4,7]
    """
    results = []
    perms = itertools.permutations(df.index, len(df.columns)-1)
    
    for perm in perms:
        prob_vals = []
        for rank, index in enumerate(perm, start=1):
            prob = df.loc[index, rank]
            prob_vals.append(prob)
        
        result_prob = np.prod(prob_vals)
        results.append({"ranking_combos": perm, "probabilities": result_prob})
    results.sort(key=lambda x: x['probabilities'], reverse=True)
    total_prob = sum(result['probabilities'] for result in results)
    for result in results:
        result['probabilities'] /= total_prob
    top_results = results[:n]
    
    return top_results

def get_probs(df, row_indices):
    product = 1  
    for i, row_index in enumerate(row_indices):
        col_name = i+1
        value = df.loc[df['row_index'] == row_index, col_name].values[0]
        if value == 0:
            return 0
        product *= value
        
        # Adjusting the next rank's probability
        if value != 0 and value != 1:
            next_index = i+1
            if next_index < len(row_indices):
                next_row_index = row_indices[next_index]
                remaining_prob = 1 - value
                adjusted_value = value / remaining_prob
                df.loc[df['row_index'] == next_row_index, col_name+1] = adjusted_value
                
    return product

def comp_rank_k_grouped(df, y, k, group_col):
    """
    df: dataframe
    y: the target node
    k: numbers of element in a comparision,[e1,e2,...ek]
    group_col: the column will be used for groupby
    """
    group_means = df.groupby(group_col)[y].mean().reset_index(name='mean_score')
    prob_df = pd.DataFrame(0, index=group_means.index, columns=range(1, k+1))
    prob_df['group_col'] = group_means[group_col]

    cols = prob_df.columns.tolist()[:-1]
    cols.insert(0, 'group_col')
    prob_df = prob_df[cols]
    scores = group_means['mean_score'].tolist()
    
    for rank in range(1, k + 1):
        for i, score in enumerate(scores):
            higher_counts = sum(1 for s in scores if s > score)
            same_counts = sum(1 for s in scores if s == score)
            if higher_counts < rank <= higher_counts + same_counts:
                prob_df.loc[i, rank] = 1 / same_counts
    
    return prob_df

def get_ranking_query_prob_grouped(G, df, k, update_vars, target_column, group_col, condition=None,opt='fix'):
    """
    G: the causal graph
    df:  dataframe
    k: the top k
    update_vars: the variables need to be updated with the value
    target_column: the column we will rank by
    condition:the condtion to filter rows
    group_col: the group by column
    """
    df_temp = df.copy()
    if condition:
        condition_mask = df_temp[list(condition.keys())].eq(pd.Series(condition)).all(axis=1)
    else:
        condition_mask = pd.Series([True]*len(df_temp))
        
    if opt=='fix':
            for var, value in update_vars.items():
                temp_lst=df_temp.loc[condition_mask, var]
                delta = [value - x for x in temp_lst]
                adjust_node(var, delta, df_temp, G, condition_mask,value,True)
                
    elif opt=='multiply_by':
            for var, factor in update_vars.items():
                temp_lst=df_temp.loc[condition_mask, var]
                delta = [factor*x - x for x in temp_lst]
                adjust_node_multi(var, delta, df_temp, G, condition_mask,factor,True)
    
    elif opt=='divided_by':
            for var, factor in update_vars.items():
                temp_lst=df_temp.loc[condition_mask, var]
                delta = [x/factor - x for x in temp_lst]
                adjust_node_multi(var, delta, df_temp, G, condition_mask,factor,True)
        
    elif opt=='add':
            for var, val in update_vars.items():
                temp_lst=df_temp.loc[condition_mask, var]
                delta = [val for _ in temp_lst]
                adjust_node_add(var, delta, df_temp, G, condition_mask,val,True)
        
    elif opt=='subs':
            for var, val in update_vars.items():
                temp_lst=df_temp.loc[condition_mask, var]
                delta = [-val for _ in temp_lst]
                adjust_node_add(var, delta, df_temp, G, condition_mask,val,True)
                
    prob_df=comp_rank_k_grouped(df_temp,target_column, k,group_col)
    
    return prob_df


def filter_prob_df_grouped(df):
    """
    df: a probility dataframe generated by ranking_query_prob
    """
    df_edit = df.columns[1:-1]
    filtered_df = df.loc[df[df_edit].apply(lambda x: any(v != 0 for v in x), axis=1)]
    return filtered_df



def base_line(df,k):
    prob=1/(m.perm(len(df), k))
    return prob

def cal_top_k_tuples(rank, df):
    prob_presence = rank.iloc[:, 1:].sum(axis=1).tolist()
    df_rows = df.loc[rank.index].copy()
    df_rows['prob_presence'] = prob_presence
    df_rows['index'] = rank.index
    return df_rows
        
    
def get_top_k_tuples(rank, df):
    filtered_rank=filter_prob_df(rank)
    return cal_top_k_tuples(filtered_rank,df)
        
