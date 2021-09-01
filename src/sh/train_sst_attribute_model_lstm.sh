#!/usr/bin/env bash
python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -label 1 -emb_size 32 -hidden_size 64 -p_drop 0.0 -ep 50 -bs 32 -num_workers 4
python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -label 1 -emb_size 32 -hidden_size 64 -p_drop 0.0 -ep 50 -bs 16 -num_workers 4
python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -label 1 -emb_size 32 -hidden_size 64 -p_drop 0.0 -ep 50 -bs 64 -num_workers 4
python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -label 1 -emb_size 32 -hidden_size 64 -p_drop 0.1 -ep 50 -bs 64 -num_workers 4
python src/attribute_models/train_attribute_model.py -out_path "output/sst_attribute_model" -model "lstm" -label 1 -emb_size 64 -hidden_size 128 -p_drop 0.0 -ep 50 -bs 64 -num_workers 4