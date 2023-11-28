# README for Ranking Functions

## Examples


### Example 1: Creating a New Causal Graph
```python
import networkx as nx
import pandas as pd
from ranking_funcs import get_probs,get_prob_backdoor,filter_prob_df_grouped,
get_ranking_query_prob_grouped,get_stable_ranking_opt,get_ranking_query,get_ranking_query_prob,
get_test_revert_ranking_rec,get_new_G,filter_prob_df,calc_prob,base_line

# Example causal graph representing influence among variables
G = nx.DiGraph()
G.add_edges_from([('Education', 'Income'), ('Experience', 'Income'), ('Income', 'Satisfaction')])

df = pd.DataFrame({
    'Education': [12, 16, 14, 18],  # Years of education
    'Experience': [5, 8, 6, 10],    # Years of work experience
    'Income': [50000, 80000, 65000, 90000],  # Annual income
    'Satisfaction': [3, 4, 3.5, 4.5],  # Job satisfaction score
    'marrital':[0,0,0,1] # marrital status (0 means status other than married,1 means married)
})

# Update causal graph based on data
new_G = get_new_G(G, df)
```

### Example 2: Performing a Ranking Query by fix
```python
# Prepare update variables for ranking
update_vars = {'Income': 100000} 

# Prepare the target column
target_column = 'Satisfaction'

# Prepare the condtion with Education equals to 12
condition={'Education':12}

# Perform a ranking query on 'Satisfaction' column with top 3 results
ranked_output =get_ranking_query(new_G, df, 3,update_vars,target_column,condition, opt="fix")
```

### Example 2: Performing a Ranking Query by mutiply by
```python
# Prepare update variables for ranking
update_vars = {'Income': 3} 

# Prepare the target column
target_column = 'Satisfaction'

# Prepare the condtion with Education equals to 12
condition={'Education':12}

# Perform a ranking query on 'Satisfaction' column with top 3 results 
ranked_output =get_ranking_query(new_G, df, 3,update_vars,target_column,condition, opt="multiply_by")
```

### Example 3: Using Backdoor Adjustment for Causal Inference
```python
# get the post intervention dataframe
post_inter_df=get_ranking_query(new_G, df, len(df),update_vars,target_column,condition, opt="multiply_by")

# Compute the probability of of post intervention by using backdoor criterion
backdoor_prob=get_prob_backdoor(post_inter_df,new_G,'Satisfaction')
```

### Example 4: Finding Stable Rankings
```python
# Finding stable ranking options
stable_result = stable_ranking_opt(new_G, df, 3,update_vars,target_column,condition, opt="multiply_by")
```

### Example 5 Getting the multiple stable Ranking
```python
#Prepare the previous ranking
prev_results=get_ranking_query(new_G, df, 3,update_vars,target_column,condition, opt="multiply_by")

#Prepare the max iteration that the ranking query will used to find the stable ranking
max_iter=1000

# Prepare the intial iteration for numbers of total iteration
num_iter=0

# prepare the max iteration for numbers of total iteration
max_num_iter=1

#Get the multiple stable ranking results for upper bond
multiple_stable_result = get_test_revert_ranking_rec(new_G, df, 3, update_vars, target_column,condition,prev_results,max_iter,num_iter,max_num_iter,'uper','multiply_by')
```
### Example 6: Get Ranking Probability
```python
import math as m
## get the the post intervention ranking probility 
ranking_prob=get_ranking_query(new_G, df, 3,update_vars,target_column,condition, opt="multiply_by")

#prepare the numbers of pwd will be generated 
num_pwd=m.perm(len(df), k)

# Filter the pwd with at least one item have top k ranking more than 0
ranking_prob_filtered=filter_prob_df(ranking_prob)

# calculate all the possible world probility after filtering
probs_multi=calc_prob(ranking_prob_filtered,num_pwd)
```

### Example 7: Getting the Grouped Ranking Probability
```python
import math as m
# Prepare the column will be used for group by
group_col='marrital'

# get the post intervention ranking probility with group by marrital
rank_rating_by_ma=get_ranking_query_prob_grouped(new_G, df, 3, update_vars, target_column,group_col,condition,'multiply_by')

#prepare the numbers of pwd will be generated 
num_pwd=m.perm(len(df), k)

# Filter the pwd with at least one item have top k ranking more than 0
rank_rating_by_ma_filtered=filter_prob_df_grouped(rank_rating_by_ma)

# calculate all the possible world probility after filtering
pwd_prob=calc_prob(rank_rating_by_ma_filtered,num_pwd)
```
