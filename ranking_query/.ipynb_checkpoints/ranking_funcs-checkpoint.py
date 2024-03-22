import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import math as m
from collections import defaultdict
from collections import Counter
from pgmpy.models import BayesianNetwork
from pgmpy.inference.CausalInference import CausalInference
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
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
            
        # if feat=='credit_amount':
        #     processed=[]
        #     for v in l:
        #         if v<=1500:
        #             processed.append(0)
        #         elif v<3000:
        #             processed.append(1)
        #         elif v<5500:
        #             processed.append(2)
        #         else:
        #             processed.append(3)
        #     df[feat]=processed
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

def create_G(edge_lst):
    """
    input:
    edge_lst: the list of edges example [('a','b'),('b','c')]
    """
    G = nx.DiGraph()
    G.add_edges_from(edge_lst)
    return G

def draw_G(G):
    """
    draw the causal graph
    """
    layout = nx.circular_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=2500, edge_color='gray', arrowsize=20, pos=layout)

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

def get_new_G_rf(G, df):
    """
    G: the causal graph
    df: the dataframe
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    col1s = []
    col2s = []
    for u, v in G.edges:
        col1s.append(u)
        col2s.append(v)
    new_G = nx.DiGraph()
    for i in range(len(col1s)):
        X = df[[col1s[i]]]  
        y = df[col2s[i]]  #
        model = RandomForestRegressor()  
        model.fit(X, y)
        weight = model.feature_importances_[0]
        new_G.add_edge(col1s[i], col2s[i], weight=weight)
        
    return new_G


def get_new_G_combined(G, df):
    """
    G: the causal graph (networkx DiGraph)
    df: the dataframe
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    new_G = nx.DiGraph()

    for node in G.nodes():
        predecessors = list(G.predecessors(node))

        if predecessors:
            X = sm.add_constant(df[predecessors])
            y = df[node]

            # Fit the regression model
            model = sm.OLS(y, X).fit()

            for pred in predecessors:
                weight = model.params[pred]
                new_G.add_edge(pred, node, weight=weight)

    return new_G

def get_new_G_combined_rf(G, df):
    """
    G: the causal graph (networkx DiGraph)
    df: the dataframe
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    new_G = nx.DiGraph()

    for node in G.nodes():
        predecessors = list(G.predecessors(node))

        if predecessors:
            X = df[predecessors]
            y = df[node]

            model = RandomForestRegressor()
            model.fit(X, y)

            feature_importances = model.feature_importances_
            for idx, pred in enumerate(predecessors):
                weight = feature_importances[idx]
                new_G.add_edge(pred, node, weight=weight)

    return new_G

def draw_G_step_by_step(G):
    """
    Draw the causal graph with edges appearing one by one and create a GIF.
    
    Parameters:
    - G: A networkx graph
    
    Returns:
    - gif_path: The path to the created GIF file showing the graph's construction.
    """
    # Create a directory to save our frames
    frames_dir = "graph_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get a list of edges so we can add them one by one
    edges = list(G.edges())
    layout = nx.circular_layout(G)
    
    # Draw the graph step by step
    for i in range(len(edges) + 1):
        plt.figure(figsize=(6, 6))
        # Draw nodes and edges up to the current iteration
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=2500, edge_color='gray', arrowsize=20, pos=layout, edgelist=edges[:i])
        
        # Draw edge labels for the current subset of edges
        edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True) if (u, v) in edges[:i]}
        nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edge_labels)
        
        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        plt.close()
    
    # Create GIF
    images = []
    for filename in sorted(os.listdir(frames_dir)):
        if filename.endswith('.png'):
            frame_path = os.path.join(frames_dir, filename)
            images.append(imageio.imread(frame_path))
    gif_path = "graph_animation.gif"
    imageio.mimsave(gif_path, images, fps=1)

    # Cleanup: remove the frames and the directory
    for filename in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, filename))
    os.rmdir(frames_dir)
    
    return gif_path

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

#     top_k_values = [float('-inf')] * k
#     top_k_indices = [-1] * k
    
#     # greedy method
#     for index, row in df_temp.iterrows():
#         min_top_k = min(top_k_values)
#         if row[target_column] > min_top_k:
#             min_index = top_k_values.index(min_top_k)
#             top_k_values[min_index] = row[target_column]
#             top_k_indices[min_index] = index

    return df_temp.sort_values(by=target_column,ascending=False).head(k)

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

#     top_k_values = [float('-inf')] * k
#     top_k_indices = [-1] * k
    
#     # greedy method
#     for index, row in df_temp.iterrows():
#         min_top_k = min(top_k_values)
#         if row[target_column] > min_top_k:
#             min_index = top_k_values.index(min_top_k)
#             top_k_values[min_index] = row[target_column]
#             top_k_indices[min_index] = index

    return df_temp.sort_values(by=target_column,ascending=False).head(k)


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
    
    # # greedy method
    # for index, row in df_temp.iterrows():
    #     min_top_k = min(top_k_values)
    #     if row[target_column] > min_top_k:
    #         min_index = top_k_values.index(min_top_k)
    #         top_k_values[min_index] = row[target_column]
    #         top_k_indices[min_index] = index

    return df_temp.sort_values(by=target_column,ascending=False).head(k)


def get_ranking_query(G, df, k, update_vars, target_column, condition=None, opt="fix"):
    
    if update_vars==None:
        return df.sort_values(by=target_column,ascending=False).head(k)
    
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


def get_top_k_ranking_query_prob(G, df, k, update_vars, target_column,row_indexes, condition=None, opt="fix"):
    tok_k_prob_df=ranking_funcs.get_ranking_query_prob(G, df, k, update_vars, target_column, condition=None, opt="fix")
    topk_ranking_probs=ranking_funcs.filter_prob_df(tok_k_prob_df)
    for idx in row_indexes:
        if idx in topk_ranking_probs.index:
            print(topk_ranking_probs.loc[idx])

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
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
                x_upper = x_val
                upper_bound_changed = True
                x_upper_iter = i
        
        if not lower_bound_changed:
            x_val_lb -= x_sd
            cur_rank_lower = ranking_query(G, df, k, {x: x_val_lb}, target_column, condition).index
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
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
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
                x_upper = x_val 
                upper_bound_changed = True
                x_upper_iter = i
        
        if not lower_bound_changed:
            x_val_lb /= x_sd
            cur_rank_lower = ranking_query_multi(G, df, k, {x: x_val_lb}, target_column, condition).index
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
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
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
                x_upper = x_val 
                upper_bound_changed = True
                x_upper_iter = i
        
        if not lower_bound_changed:
            x_val_lb -= x_sd
            cur_rank_lower = ranking_query_add(G, df, k, {x: x_val_lb}, target_column, condition).index
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
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
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
                upper_bound_changed = True
                print([x_val_upper,x_upper_iter+i,"update upper"])
                x_pre=x_val_upper

            
        if x_val_lower is not None and bond_check=='lower' and lower_bound_changed is False:
            x_val_lower -= x_sd
            cur_rank_upper = ranking_query(G, df, k, {x: x_val_lower}, target_column, condition).index
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
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
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
                upper_bound_changed = True
                print([x_val_upper,x_upper_iter+i,"update upper"])
                x_pre=x_val_upper

            
        if x_val_lower is not None and bond_check=='lower' and lower_bound_changed is False:
            x_val_lower /= x_sd
            cur_rank_upper = ranking_query_multi(G, df, k, {x: x_val_lower}, target_column, condition).index
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
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
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
                upper_bound_changed = True
                print([x_val_upper,x_upper_iter+i,"update upper"])
                x_pre=x_val_upper

            
        if x_val_lower is not None and bond_check=='lower' and lower_bound_changed is False:
            x_val_lower -= x_sd
            cur_rank_upper = ranking_query_add(G, df, k, {x: x_val_lower}, target_column, condition).index
            if not np.array_equal(sorted(cur_rank_upper), sorted(rank)):
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
    k: numbers of elements in a comparison, [e1,e2,...ek]
    """
    original_indices = df.index.tolist()
    scores = df[y].tolist()
    prob_df = pd.DataFrame(0, index=original_indices, columns=range(1, k+1)) 
    prob_df['row_index'] = original_indices
    cols = ['row_index'] + [col for col in prob_df.columns if col != 'row_index']
    prob_df = prob_df[cols]
    
    for rank in range(1, k+1):
        for idx, score in zip(original_indices, scores):  
            higher_counts = sum(1 for s in scores if s > score)
            same_counts = sum(1 for s in scores if s == score)
            if higher_counts < rank <= higher_counts + same_counts:
                prob_df.at[idx, rank] = 1 / same_counts 
    
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
    prob=1/(m.comb(len(df), k))
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


def find_backdoor_sets_opt(cgm, Y, X):
    return cgm.get_all_backdoor_adjustment_sets(Y, X)

def get_cgm(G):
    return CausalGraphicalModel(
    nodes=G.nodes,
    edges=G.edges)


def backdoor_adjustment_opt(df, Y, y, A, a, Z):
    prob = 0
    total_len = len(df)
    total_relevant_Z = 0  
    unique_Z_combinations = df[Z].drop_duplicates()
    for z_values in unique_Z_combinations.itertuples(index=False):
        mask_Z = np.ones(len(df), dtype=bool)
        for column, value in zip(Z, z_values):
            mask_Z = mask_Z & (df[column] == value)
        
        df_Z = df[mask_Z]
        df_A_a_Z = df_Z[df_Z[A] == a]

        if not df_A_a_Z.empty:
            p_Y_given_A_Z = (df_A_a_Z[Y] == y).sum() / len(df_A_a_Z)
            p_Z = len(df_Z) / total_len
            total_relevant_Z += len(df_Z)
            prob += p_Y_given_A_Z * p_Z
    if total_relevant_Z > 0:
        prob = prob * total_len / total_relevant_Z

    return prob

def get_lst_prob(lsts):
    flat_lsts = [list(lst) for lst in lsts]
    lst_counts = Counter(map(tuple, flat_lsts)) 
    total = sum(lst_counts.values())
    prob = {lst: count / total for lst, count in lst_counts.items()}
    data = {'rank': [list(lst) for lst in prob.keys()], 'prob': list(prob.values())}
    df = pd.DataFrame(data)
    return df

def Greedy_Algo(G, df, k, target_column, vars_test,thresh_hold=0,condition=None,max_iter=100, opt="add",force=0.01):
    rank_result=[]
    if opt=='add'or 'subs':
        if opt=='add':
            pos=1
        else:
            pos=-1
        for var in vars_test:
            x_up=0
            x_sd = np.abs(df[var].std() * force)*pos
            for i in range(max_iter):
                x_up+=x_sd
                new_rank=get_ranking_query(G, df, k, {var:x_up},                                     target_column,condition,opt).index
                rank_result.append(new_rank)
                
    elif opt=='multiply_by'or 'divided_by':
        if opt=='divided_by':
            def op_chang(x_sd):
                return 1/x_sd
        else:
            def op_chang(x_sd):
                return x_sd    
        for var in vars_test:
            x_up=0
            x_sd = op_chang(1+np.abs(df[var].std() * force))
            for i in range(max_iter):
                x_up*=x_sd
                new_rank=get_ranking_query(G, df, k, {var:x_up}, target_column,condition,opt).index
                rank_result.append(new_rank)
    else:
        print('invalid operator, operator must be add,subs,multiply_by and divided_by')
    res=get_lst_prob(rank_result)
    filter_res=res[res['prob'] >= thresh_hold]
    filter_res['total_iters']=len(rank_result)*filter_res['prob']
    return filter_res

def get_most_probable_elements(df):
    element_probs = defaultdict(float)
    element_iters = defaultdict(float)
    for _, row in df.iterrows():
        unique_elements = set(row['rank'])  
        for element in unique_elements:
            element_probs[element] += row['prob']  
            element_iters[element] += row['total_iters'] 
    combined_data = {
        'element': list(element_probs.keys()),
        'prob': list(element_probs.values()),
        'total_iters': [element_iters[el] for el in element_probs.keys()]
    }
    stats_df = pd.DataFrame(combined_data)
    return stats_df



##currently not use and needs testing

def check_constr(df, conditions):
    if df[list(conditions.keys())].eq(pd.Series(conditions)).all(axis=1).any():
        return 'invalid'
    else:
        return 'valid'
    

def check_constraint_violation(df, G, X, X_values, Y, Y_value):
    cgm = get_cgm(G)
    if len(X) != len(X_values):
        raise ValueError("Length of X and X_values must be the same")


    violation_prob = 0
    for x, x_val in zip(X, X_values):
        bd_sets = find_backdoor_sets_opt(cgm, Y, x)

        for bd_set in bd_sets:
            adjusted_prob = backdoor_adjustment_opt(df, Y, Y_value, x, x_val, list(bd_set))
            violation_prob += (1 - adjusted_prob)  

    return violation_prob / len(X)


def create_pwd_dataframe(pwds, validity):
    pwd_stats = defaultdict(lambda: {'valid_count': 0, 'invalid_count': 0})

    for pwd, status in zip(pwds, validity):
        pwd_tuple = tuple(pwd) 
        if status == 'valid':
            pwd_stats[pwd_tuple]['valid_count'] += 1
        else:
            pwd_stats[pwd_tuple]['invalid_count'] += 1
    total_count = len(validity)
    data = []
    for pwd, stats in pwd_stats.items():
        valid_prob = stats['valid_count'] / total_count
        invalid_prob = stats['invalid_count'] / total_count
        data.append({
            'valid_pwd': list(pwd),
            'valid_pwd_prob': valid_prob,
            'invalid_pwd_prob': invalid_prob,
            'valid_total_iters': stats['valid_count'],
            'invalid_total_iters': stats['invalid_count']
        })

    return pd.DataFrame(data)


def noisy_causal(G, df, k, target_column,constr, vars_test,thresh_hold=0,condition=None,max_iter=100, opt="add",force=0.01):
    rank_result=[]
    validity_check=[]
    if opt=='add'or 'subs':
        if opt=='add':
            pos=1
        else:
            pos=-1
        for var in vars_test:
            x_up=0
            x_sd = np.abs(df[var].std() * force)*pos
            for i in range(max_iter):
                x_up+=x_sd
                new_pwd=get_ranking_query(G, df, len(df), {var:x_up}, target_column,condition,opt)
                new_rank=new_pwd.head(k).index
                rank_result.append(new_rank)
                validity_check.append(check_constr(new_pwd,constr))
                
    elif opt=='multiply_by'or 'divided_by':
        if opt=='divided_by':
            def op_chang(x_sd):
                return 1/x_sd
        else:
            def op_chang(x_sd):
                return x_sd    
        for var in vars_test:
            x_up=0
            x_sd = op_chang(1+np.abs(df[var].std() * force))
            for i in range(max_iter):
                x_up*=x_sd
                new_pwd=get_ranking_query(G, df, len(df), {var:x_up}, target_column,condition,opt)
                new_rank=new_pwd.head(k).index
                rank_result.append(new_rank)
                validity_check.append(check_constr(new_pwd,constr))
    else:
        print('invalid operator, operator must be add,subs,multiply_by and divided_by')
    
#########



def read_imdb_data(path):
    df = pd.read_csv(path)
    df['averageRating'] = round(df['averageRating'], 1)

    featlst = list(df.columns)

    for feat in featlst:
        l = df[feat]  

        if feat == 'isAdult':
            processed = [1 if v == 1 else 0 for v in l]
            df[feat] = processed

        if feat == 'runtimeMinutes':
            processed = [0 if v <= 23 else 1 if v <= 37 else 2 if v <= 47 else 3 for v in l]
            df[feat] = processed

        if feat == 'seasonNumber':
            processed = [0 if v <= 1 else 1 if v <= 2 else 2 if v <= 5 else 3 for v in l]
            df[feat] = processed

        if feat == 'episodeNumber':
            processed = [0 if v <= 4 else 1 if v <= 8 else 2 if v <= 15 else 3 for v in l]
            df[feat] = processed

        if feat == 'numVotes':
            processed = [0 if v <= 14 else 1 if v <= 35 else 2 if v <= 139 else 3 for v in l]
            df[feat] = processed

    return df

def find_backdoor_sets_opt(G, Y, X):
    new_G = BayesianNetwork()
    new_G.add_nodes_from(G.nodes)
    new_G.add_edges_from(G.edges)
    inference = CausalInference(new_G)
    backdoor_sets = inference.get_all_backdoor_adjustment_sets(X, Y)
    if backdoor_sets:
        min_length = min(len(s) for s in backdoor_sets)
        shortest_backdoor_sets = [s for s in backdoor_sets if len(s) == min_length]
        return shortest_backdoor_sets
    else:
        return None



def backdoor_adjustment_opt(df, Y, y, A, a, Z):
    prob = 0
    total_len = len(df)
    total_relevant_Z = 0  

    unique_Z_combinations = df[Z].drop_duplicates()
    for z_values in unique_Z_combinations.itertuples(index=False):
        mask_Z = np.ones(len(df), dtype=bool)
        for column, value in zip(Z, z_values):
            mask_Z = mask_Z & (df[column] == value)
        
        df_Z = df[mask_Z]
        df_A_a_Z = df_Z[df_Z[A] == a]

        if not df_A_a_Z.empty:
            p_Y_given_A_Z = (df_A_a_Z[Y] == y).sum() / len(df_A_a_Z)
            p_Z = len(df_Z) / total_len
            total_relevant_Z += len(df_Z)
            prob += p_Y_given_A_Z * p_Z
    if total_relevant_Z > 0:
        prob = prob * total_len / total_relevant_Z

    return prob



def get_prob_backdoor_opt(G, df, k, update_vars, target_column, condition, opt, row_indexes,theta):
    ### get the updated dataframe
    updated_df = ranking_funcs.get_ranking_query(G, df, len(df), update_vars, target_column, condition, opt)
    ### the updated variable
    node = list(update_vars.keys())[0]
    results = []
    ### find the one of the backdoor set of updated variable
    bd_set = ranking_funcs.find_backdoor_sets_opt(G, target_column, node)[0]
    dom_y = updated_df[target_column].unique()
    dom_node = updated_df[node].unique()
    for d_y in dom_y:
        for d_n in dom_node:
            adjusted_prob = ranking_funcs.backdoor_adjustment_opt(updated_df, target_column, d_y, node, d_n, list(bd_set))
            results.append({
                'Y': target_column, 
                'Y_value': d_y, 
                'X': node, 
                'X_value': d_n, 
                'Z': ', '.join(bd_set), 
                'prob': adjusted_prob
            })
    ## get the probability dataframe
    prob_df = pd.DataFrame(results)
    
    z_relevant_probs = prob_df[(prob_df['Y_value'] >= theta)]
    prob_groups = []
    for row_index in row_indexes:
        row = updated_df.loc[row_index]                    
        x_value = row[node]
        prob_sum = z_relevant_probs[(z_relevant_probs['X_value'] == x_value)]['prob'].sum()
        prob_groups.append(prob_sum)
        
    return m.prod(prob_groups)

def Comp_Greedy_Algo_backdoor(row_indexes,G, df, k, target_column, vars_test,thresh_hold=0,condition=None,max_iter=100, opt="add",force=0.01):
    prob_result=[]
    
    if opt=='add'or 'subs':
        if opt=='add':
            pos=1
        else:
            pos=-1
        for var in vars_test:
            x_up=0
            x_sd = np.abs(df[var].std() * force)*pos
            for i in range(max_iter):
                x_up+=x_sd
                updated_df=get_ranking_query(G, df, len(df), {var:x_up}, target_column, condition, opt)
                theta=updated_df[target_column].iloc[k-1]
                prob_backdoor=get_prob_backdoor_opt(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_result.append(prob_backdoor)
                
    elif opt=='multiply_by'or 'divided_by':
        if opt=='divided_by':
            def op_chang(x_sd):
                return 1/x_sd
        else:
            def op_chang(x_sd):
                return x_sd    
        for var in vars_test:
            x_up=0
            x_sd = op_chang(1+np.abs(df[var].std() * force))
            for i in range(max_iter):
                x_up*=x_sd
                updated_df=get_ranking_query(G, df, len(df), {var:x_up}, target_column, condition, opt)
                theta=updated_df[target_column].iloc[k-1] 
                prob_backdoor=get_prob_backdoor_opt(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_result.append(prob_backdoor)
    else:
        print('invalid operator, operator must be add,subs,multiply_by and divided_by')
    return prob_result


def read_imdb_actor_data(path):
    df = pd.read_csv(path)
    df['averageRating'] = round(df['averageRating'], 0)
    df['isAdult'] = df['isAdult'].apply(lambda x: 1 if x == 1 else 0)

    def runtime_category(runtime):
        if runtime <= 5: return 0
        elif runtime <= 15: return 1
        elif runtime <= 30: return 2
        else: return 3
    df['runtimeMinutes'] = df['runtimeMinutes'].apply(runtime_category)

    def votes_category(votes):
        if votes <= 15: return 0
        elif votes <= 35: return 1
        elif votes <= 100: return 2
        else: return 3
    df['numVotes'] = df['numVotes'].apply(votes_category)

    name_map = {'Michael Smith': 0, 'David Smith': 1, 'Michael Johnson': 2,
                'Chris Smith': 3, 'David Brown': 4, 'David Jones': 5}
    df['primaryName'] = df['primaryName'].apply(lambda x: name_map.get(x, -1))

    title_type_map = {'tvSeries': 0, 'tvMiniSeries': 0, 'tvSpecial': 1,
                      'tvMovie': 1, 'tvShort': 1, 'tvEpisode': 1,
                      'movie': 2, 'short': 3, 'video': 4, 'videoGame': 4}
    df['titleType'] = df['titleType'].apply(lambda x: title_type_map.get(x, -1))
    return df


def backdoor_adjustment_opt2(df, Y, y, A, a, Z):
    df_A_a = df[df[A] == a]
    grouped = df_A_a.groupby(Z).apply(lambda g: (g[Y] == y).sum() / len(g) if not g.empty else 0)
    grouped = grouped[grouped > 0].reset_index()
    grouped.rename({0: 'probs'}, axis=1, inplace=True)
    grouped['Y_value'] = y
    grouped[A] = a 
    grouped['expected_value'] = grouped['Y_value']*grouped['probs']

    return grouped



def predict_backdoor_opt2(G, df, k, update_vars, target_column, condition, opt):
    """
    Use P(Y|do(X),Z) to estimate
    """
    updated_df = get_ranking_query(G, df, len(df), update_vars, target_column, condition, opt)
    node = list(update_vars.keys())[0]
    results = []
    bd_set = ranking_funcs.find_backdoor_sets_opt(G, target_column, node)[0]
    dom_y = updated_df[target_column].unique()
    dom_node = updated_df[node].unique()
    for d_y in dom_y:
        for d_n in dom_node:
            result_df = backdoor_adjustment_opt2(updated_df, target_column, d_y, node, d_n, list(bd_set))
            if not result_df.empty:
                results.append(result_df)

    merged_df = pd.concat(results, ignore_index=True)
    flat_bd_sets = [col for subset in bd_sets for col in subset]+[node]
    grouped_df = merged_df.groupby(flat_bd_sets).agg({'expected_value': 'sum'}).reset_index()

    expected_values = []
    for row_index, row in updated_df.iterrows():
        match_conditions = {col: row[col] for col in flat_bd_sets}
        matched_row = grouped_df[(grouped_df[list(match_conditions)] == pd.Series(match_conditions)).all(axis=1)]
        if not matched_row.empty:
            expected_value = matched_row['expected_value'].values[0]
        expected_values.append(expected_value)

    result_df = pd.DataFrame({'row_index': updated_df.index, 'expected_value': expected_values})
    return result_df.sort_values(by='expected_value', ascending=False).head(k)


def get_prob_backdoor_opt2(G, df, k, update_vars, target_column, condition, opt, row_indexes, theta):
    """
    Use P(Y|do(X),Z) to estimate
    """
    updated_df = get_ranking_query(G, df, len(df), update_vars, target_column, condition, opt)
    node = list(update_vars.keys())[0]
    results = []
    bd_set = ranking_funcs.find_backdoor_sets_opt(G, target_column, node)[0]
    dom_y = updated_df[target_column].unique()
    dom_node = updated_df[node].unique()
    for d_y in dom_y:
        for d_n in dom_node:
            result_df = backdoor_adjustment_opt2(updated_df, target_column, d_y, node, d_n, list(bd_set))
            if not result_df.empty:
                results.append(result_df)
    merged_df = pd.concat(results, ignore_index=True)
    flat_bd_sets = [col for subset in bd_sets for col in subset]+[node]
    filtered_merged_df=merged_df[merged_df['Y_value']>=theta]
    prob_df=filtered_merged_df.groupby(flat_bd_sets).agg({'probs': 'sum'}).reset_index()
    
    total_probs = []
    for row_index in row_indexes:
        row = updated_df.loc[row_index]                    
        match_conditions = {col: row[col] for col in flat_bd_sets}
        matched_row = prob_df[(prob_df[list(match_conditions)] == pd.Series(match_conditions)).all(axis=1)]
        if not matched_row.empty:
            total_prob = matched_row['probs'].values[0]
        else:
            total_prob=0
        total_probs.append(total_prob)
        
    result_df = pd.DataFrame({'row_index': row_indexes, 'total_probs': total_probs})
    return result_df['total_probs'].prod()


def predict_backdoor_opt(G, df, k, update_vars, target_column, condition, opt):
    updated_df = get_ranking_query(G, df, len(df), update_vars, target_column, condition, opt)
    nodes = update_vars.keys()
    results = []
    ### only one node update each time
    for node in nodes:
        bd_sets = find_backdoor_sets_opt(G, target_column, node)
        for bd_set in bd_sets:
            dom_y = updated_df[target_column].unique()
            dom_node = updated_df[node].unique()
            for d_y in dom_y:
                for d_n in dom_node:
                    adjusted_prob = backdoor_adjustment_opt(updated_df, target_column, d_y, node, d_n, list(bd_set))
                    results.append({
                        target_column: d_y, 
                        node: d_n, 
                        'Z': ', '.join(bd_set), 
                        'prob': adjusted_prob
                    })
    group_df = pd.DataFrame(results)
    group_df['expected_value']=group_df['prob']*group_df[target_column]
    prob_df = group_df.groupby([node]).agg({'expected_value': 'sum'}).reset_index()
    
    expected_values = []
    for row_index, row in updated_df.iterrows():                  
        match_conditions = row[node]
        matched_row = prob_df[prob_df[node] == match_conditions]
        if not matched_row.empty:
            expected_value = matched_row['expected_value'].values[0]
        else:
            expected_value=0
        expected_values.append(expected_value)
        
    result_df = pd.DataFrame({'row_index': updated_df.index, 'expected_value': expected_values})
    return result_df.sort_values(by='expected_value', ascending=False).head(k)

def get_backdoor_opt1(G, df, k, update_vars, target_column, condition, opt,row_indexes,theta):
    updated_df = get_ranking_query(G, df, len(df), update_vars, target_column, condition, opt)
    nodes = update_vars.keys()
    results = []
    ### only one node update each time
    for node in nodes:
        bd_sets = find_backdoor_sets_opt(G, target_column, node)
        for bd_set in bd_sets:
            dom_y = updated_df[target_column].unique()
            dom_node = updated_df[node].unique()
            for d_y in dom_y:
                for d_n in dom_node:
                    adjusted_prob = backdoor_adjustment_opt(updated_df, target_column, d_y, node, d_n, list(bd_set))
                    results.append({
                        target_column: d_y, 
                        node: d_n, 
                        'Z': ', '.join(bd_set), 
                        'prob': adjusted_prob
                    })
    group_df = pd.DataFrame(results)
    filtered_group_df = group_df[group_df[target_column] >= theta]
    prob_df = filtered_group_df.groupby([node]).agg({'prob': 'sum'}).reset_index()
    
    total_probs = []
    for row_index in row_indexes:
        row = updated_df.loc[row_index]                    
        match_conditions = row[node]
        matched_row = prob_df[prob_df[node] == match_conditions]
        if not matched_row.empty:
            total_prob = matched_row['prob'].values[0]
        else:
            total_prob=0
        total_probs.append(total_prob)
    return np.prod(total_probs)

def accuracy_topk_rank(pred_rank, true_rank):
    if len(pred_rank) != len(true_rank):
        raise ValueError("Both lists must be of the same length.")

    correct_count = sum(p == t for p, t in zip(pred_rank, true_rank))
    return correct_count / len(pred_rank)



def accuracy_in_topk(pred_rank, true_rank):
    if len(pred_rank) != len(true_rank):
        raise ValueError("Both lists must be of the same length.")

    correct_count = sum(p in true_rank for p in pred_rank)
    return correct_count / len(pred_rank)

def read_imdb_movie_actor_data(path):
    df = pd.read_csv(path)
    df['averageRating'] = round(df['averageRating'], 0)

    def runtime_category(runtime):
        if runtime <= 120: return 0
        elif runtime <= 150: return 1
        else: return 2
    df['runtimeMinutes'] = df['runtimeMinutes'].apply(runtime_category)

    def votes_category(votes):
        if votes <= 10000: return 0
        elif votes <= 50000: return 1
        elif votes <= 100000: return 2
        else: return 3
    df['numVotes'] = df['numVotes'].astype(int).apply(votes_category)
    
    def startYear_category(year):
        if year <= 2000: return 0
        elif year <= 2010: return 1
        elif year <= 2020: return 2
        else: return 3
    df['startYear'] = df['startYear'].astype(int).apply(startYear_category)

    name_map = {'Scarlett Johansson': 0,
            'Emma Mackey': 1,
            'Margot Robbie': 2,
            'Johnny Depp': 3,
            'Jason Momoa': 4,
            'Rinko Kikuchi': 5,
            'Ben Kingsley': 6,
            'Om Puri': 7}
    
    df['primaryName'] = df['primaryName'].apply(lambda x: name_map.get(x, -1))

    return df

def data_size_backdoor(G, G_method, df, k, update_vars, target_column, condition, opt, row_indexes, n_splits, random_state):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df_dropped = df.drop(row_indexes)
    
    folds = []
    for _, test_index in kf.split(df_dropped):
        fold_data = df_dropped.iloc[test_index]
        folds.append(fold_data)
    
    selected_rows = df.loc[row_indexes]

    data = pd.concat([selected_rows, folds[0]], axis=0)
    
    back_result = []
    back2_result = []
    back_result_len = []
    for i in range(1, len(folds)):
        data = pd.concat([data, folds[i]], axis=0)
        
        if condition and not set(condition.keys()).issubset(data.columns):
            update_vars = None
        
        updated_df=get_ranking_query(G, data, len(data), update_vars, target_column, condition, opt).sort_values(by=target_column,ascending=False)
        theta=updated_df[target_column].iloc[k-1]
        back_result_len.append(len(updated_df))
        back_result.append(get_prob_backdoor_opt(G, data, k, update_vars,
                                                 target_column, condition, opt, row_indexes, theta))
        
        back2_result.append(get_prob_backdoor_opt2(G, data, k, update_vars,
                                                 target_column, condition, opt, row_indexes, theta))
        
    return back_result_len,back_result,back2_result


def k_range_backdoor(G, G_method, df, k, update_vars, target_column, condition, opt, row_indexes,end_k):
    back_result=[]
    back2_result = []
    for z in range(k,end_k+1):
        updated_df=get_ranking_query(G, df, len(df), update_vars, target_column, condition, opt)
        theta=updated_df[target_column].iloc[z-1]
        back_result.append(get_prob_backdoor_opt(G, df, z, update_vars,
                                                 target_column, condition, opt, row_indexes, theta))
        back2_result.append(get_prob_backdoor_opt2(G, df, k, update_vars,
                                                 target_column, condition, opt, row_indexes, theta))
    return back_result,back2_result


def Comp_Greedy_Algo_backdoor_edited(row_indexes,G, df, k, target_column, vars_test,thresh_hold=0,condition=None,max_iter=100, opt="add",force=0.01):
    prob_result=[]
    prob_result2=[]
    if opt=='add'or 'subs':
        if opt=='add':
            pos=1
        else:
            pos=-1
        for var in vars_test:
            x_up=0
            x_sd = np.abs(df[var].std() * force)*pos
            for i in range(max_iter):
                x_up+=x_sd
                updated_df=get_ranking_query(G, df, len(df), {var:x_up}, target_column, condition, opt)
                theta=updated_df[target_column].iloc[k-1]
                prob_backdoor=get_prob_backdoor_opt(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_backdoor2=get_prob_backdoor_opt2(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_result.append(prob_backdoor)
                prob_result2.append(prob_backdoor2)
                
    elif opt=='multiply_by'or 'divided_by':
        if opt=='divided_by':
            def op_chang(x_sd):
                return 1/x_sd
        else:
            def op_chang(x_sd):
                return x_sd    
        for var in vars_test:
            x_up=0
            x_sd = op_chang(1+np.abs(df[var].std() * force))
            for i in range(max_iter):
                x_up*=x_sd
                updated_df=get_ranking_query(G, df, len(df), {var:x_up}, target_column, condition, opt)
                theta=updated_df[target_column].iloc[k-1] 
                prob_backdoor=get_prob_backdoor_opt(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_backdoor2=get_prob_backdoor_opt2(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_result.append(prob_backdoor)
                prob_result2.append(prob_backdoor2)
    else:
        print('invalid operator, operator must be add,subs,multiply_by and divided_by')
    return prob_result,prob_result2

def Comp_Greedy_Algo_backdoor2(row_indexes,G, df, k, target_column, vars_test,thresh_hold=0,condition=None,max_iter=100, opt="add",force=0.01):
    prob_result=[]
    prob_result2=[]
    if opt=='add'or 'subs':
        if opt=='add':
            pos=1
        else:
            pos=-1
        for var in vars_test:
            x_up=0
            x_sd = np.abs(df[var].std() * force)*pos
            for i in range(max_iter):
                x_up+=x_sd
                updated_df=get_ranking_query(G, df, len(df), {var:x_up}, target_column, condition, opt)
                theta=updated_df[target_column].iloc[k-1]
                prob_backdoor2=get_prob_backdoor_opt2(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_result2.append(prob_backdoor2)
                
    elif opt=='multiply_by'or 'divided_by':
        if opt=='divided_by':
            def op_chang(x_sd):
                return 1/x_sd
        else:
            def op_chang(x_sd):
                return x_sd    
        for var in vars_test:
            x_up=0
            x_sd = op_chang(1+np.abs(df[var].std() * force))
            for i in range(max_iter):
                x_up*=x_sd
                updated_df=get_ranking_query(G, df, len(df), {var:x_up}, target_column, condition, opt)
                theta=updated_df[target_column].iloc[k-1] 
                prob_backdoor2=get_prob_backdoor_opt2(G, df, k, {var:x_up}, target_column, condition, opt, row_indexes, theta)
                prob_result2.append(prob_backdoor2)
    else:
        print('invalid operator, operator must be add,subs,multiply_by and divided_by')
    return prob_result2

def factor_imdb(df):

    def runtime_category(runtime):
        if runtime <= 120: return 0
        elif runtime <= 150: return 1
        else: return 2
    df['runtimeMinutes'] = df['runtimeMinutes'].apply(runtime_category)

    def votes_category(votes):
        if votes <= 10000: return 0
        elif votes <= 50000: return 1
        elif votes <= 100000: return 2
        else: return 3
    df['numVotes'] = df['numVotes'].astype(int).apply(votes_category)
    
    def startYear_category(year):
        if year <= 2000: return 0
        elif year <= 2010: return 1
        elif year <= 2020: return 2
        else: return 3
    df['startYear'] = df['startYear'].astype(int).apply(startYear_category)

    name_map = {'Scarlett Johansson': 0,
            'Emma Mackey': 1,
            'Margot Robbie': 2,
            'Johnny Depp': 3,
            'Jason Momoa': 4,
            'Rinko Kikuchi': 5,
            'Ben Kingsley': 6,
            'Om Puri': 7,
            'Jennifer Aniston':8,
            'Angelina Jolie': 9,
            'Taylor Kitsch':10,
            'Chris Hemsworth':11,}
    
    df['primaryName'] = df['primaryName'].apply(lambda x: name_map.get(x, -1))
    
    genre_map = {'Comedy': 0, 
                 'Action & Adventure': 1, 
                 'Drama': 2, 
                 'Documentary & Biography': 3,
                 'Horror & Thriller': 4, 
                 'Family & Animation': 5, 
                 'Other': 6, 
                 'Crime & Mystery': 7}
    
    df['genres'] = df['genres'].apply(lambda x: genre_map.get(x, -1)) 

    return df