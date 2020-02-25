# Data Aggregation CEDR Ranker for robust 04

## Setup
1. setup environment
> conda install -r requirements.txt (use conda)
2. place robust04 dataset under data directory
> https://archive.org/download/deep_relevance_ranking_data/robust04_data.tar.gz  
> https://drive.google.com/open?id=1PfRriJGHq3fm85AhKXNt72z9GqrbL9IU  
3. make sure downloaded data is placed correctly
> assert os.listdir('data/robust04/split_1/') == ['prepro.train.pkl', 'prepro.test.pkl', 'rob04.train.s1.json', ...]
> assert os.listdir('data') == ['robust04', 'run_eval.py', 'trec_eval']
4. unzip data/gpt2-large-docs.zip
> assert 'gpt2-large-docs.pkl' in os.listdir('data')


## Run
#### Data Generation: run gpt-2-Pytorch/gen_data.py (run under gpt-2-Pytorch directory)

##### Argument

- '--nsamples': Number of documents for each query
- '--unconditional': If true, unconditional generation
- '--batch_size'
- '--length': Length of document
- '--temperature': Temperature of softmax
- '--top_k': Number of candidate of predicted next tokens
- '--checkout_k': Generate top k query from given tsv file
- '--split_num': Save data each k query
- '--rel_mean': Gaussian mean of the #(relevant doc)
- '--rel_std': Gaussian std of the #(relevant doc)
- '--rel_min': Minimum number of the #(relevant doc)
- '--rel_max': Maximum number of the #(relevant doc)
- '--tsv_path(default='../data/clueweb09_queries.tsv')
- '--save_path(default='../data/')
- '--verbose'

#### Train(or Test) Ranker: run main.py
##### Argument

- '--model'(choices=['bert', 'cedr_knrm', 'cedr_pacrr', 'cedr_drmm']): Choose model
- '--split': Choose split (ex. --split 1 3 5 or --split 1)
- '--epoch': Train epoch
- '--batch_size': Mini-batch size for train
- '--batch_size_test': Mini-batch size for test
- '--min_map': Save threshold 
- '--load_path': Load weight in path if load_path is not None
- '--name': save directory name
- '--verbose'
- '--test'
