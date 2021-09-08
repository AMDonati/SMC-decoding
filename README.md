# SMC-decoding

### requirements
`pip install -r requirements.txt`  
Install the datasets library from source:  
`cd ..`    
`git clone https://github.com/huggingface/datasets.git`  
`cd datasets`  
`pip install -e .`
#### if not working:
* install cython and scm_tools: 
  *`pip install cython`
  * `pip install setuptools_scm`
* upgrade pip

Before running scripts: 
`export PYTHONPATH=src:${PYTHONPATH}`

### Loading and saving cached dataset and models
* python src/save_datasets_models.py

### Training the attribute model
#### with GPT2 Tokenizer
* `python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -label 1 -emb_size 32 -hidden_size 64 -p_drop 0.0 -ep 50 -bs 32 -num_workers 4`
* `python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "gpt2" -label 1  -p_drop 0.0 -ep 50 -bs 32 -num_workers 4`
#### with SST Tokenizer
*  `python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -tokenizer "sst" -label 1 -emb_size 32 -hidden_size 64 -p_drop 0.0 -ep 5 -bs 32 -num_workers 4`
