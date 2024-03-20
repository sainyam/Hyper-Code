# Hyper-Code

Suggested Python: 3.9.13

## Reproducibility Steps

### Step 1: Download this repo and cd to it

```
git clone https://github.com/sainyam/Hyper-Code
cd Hyper-Code
```

### Step 2: Install dependencies

```
sudo apt-get update
sudo apt-get install virtualenv
virtualenv venv
source ./venv/bin/activate 

pip install -r requirements.txt
```

### Examples


#### Example 1: Creating a New Causal Graph
```python
import pandas as pd
import ranking_funcs
# Example causal graph representing influence among variables
edges = [('Education', 'Income'), ('Experience', 'Income'), ('Income', 'Satisfaction')]
G = create_G(edges)
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

#### Example 2: Performing a Ranking Query by fix
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

#### Example 3: Performing a Ranking Query by mutiply by
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

#### Example 4: Using Backdoor Adjustment for Causal Inference
```python
# get the post intervention dataframe
post_inter_df=get_ranking_query(new_G, df, len(df),update_vars,target_column,condition, opt="multiply_by")

# get the causal graph
cgm=get_cgm(new_G)

# Compute the probability of of post intervention by using backdoor criterion
backdoor_prob=get_prob_backdoor_opt(post_inter_df,cgm,'Satisfaction')
```

#### Example 5: Finding Stable Rankings
```python
# Finding stable ranking options
stable_result = stable_ranking_opt(new_G, df, 3,update_vars,target_column,condition, opt="multiply_by")
```

#### Example 6: Getting the multiple stable Ranking
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
#### Example 7: Get Ranking Probability
```python
import math as m
## get the the post intervention ranking probility 
ranking_prob=get_ranking_query_prob(new_G, df, 3,update_vars,target_column,condition, opt="multiply_by")

#prepare the numbers of pwd will be generated 
num_pwd=m.perm(len(df), k)

# Filter the pwd with at least one item have top k ranking more than 0
ranking_prob_filtered=filter_prob_df(ranking_prob)

# calculate all the possible world probility after filtering
probs_multi=calc_prob(ranking_prob_filtered,num_pwd)
```

#### Example 8: Getting the Grouped Ranking Probability
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

#### Example 9: Get the tuple for most probable top k
```python
# Prepare the rank
rank_rating_by_ma=get_ranking_query_prob_grouped(new_G, df, 3, update_vars, target_column,group_col,condition,'multiply_by')

# Get the top k tuples
top_k_tuples=get_top_k_tuples(rank_rating_by_ma,df)
```

#### Example 10: Get the tuple for most probable top k
```python
# prepare the variables will be used to change the rank that it should exclude the target column 
var=['Education', 'Experience', 'Income']

# Prepare the occurance probility datafrmae for rank
df_gr=Greedy_Algo(new_G, df, 3, target_column,var,condition,100,'multiply_by',0.5)

# Get the most probable elements in top k pwd 
most_probable_elements=get_most_probable_elements(df_gr)
```