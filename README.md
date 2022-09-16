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
sudo apt-get virtualenv
virtualenv venv
source ./venv/bin/activate 

pip install -r requirements.txt
```

### Step 3: Generate all plots

```
bash reproduce.sh
```

The plots generated after running the code are present in reproducibility/freshRuns/ folder. Please go over reproducibility/readme.pdf to interpret the results.


### (Optional) Step 4: Reproduce individual plots

```
cd reproducibility/scripts/
bash fig6.sh
bash fig8.sh
bash fig9.sh
bash fig10.sh
bash fig11.sh
bash fig12.sh
```

