# HousE: Knowledge Graph Embedding with Householder Parameterization

This is the code of the paper *HousE: Knowledge Graph Embedding with Householder Parameterization* for ICML 2022.

A more powerful and general framework for knowledge graph embedding.

## Requirements
- pytorch == 1.8.0
- numpy == 1.19.2
- scikit-learn == 0.23.2

## Data
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

## Models
- [x] HousE-r: relational Householder rotations
- [x] HousE: relational Householder projections + relational Householder rotations
- [x] HousE-r<sup>+</sup>: relational Householder rotations + translations
- [x] HousE<sup>+</sup>: relational Householder projections + relational Householder rotations + translations

## Usage
All training commands are listed in [best_config.sh](./best_config.sh). 
For example, you can run the following commands to train HousE on WN18RR and FB15k-237 datasets.
```
# WN18RR
bash run.sh HousE wn18rr 0 0 0 1000 200 800 8 1 0.5 6.0 1.14940435933987 0.000575323908649059 60000 20000 8 0.0960737047401994

# FB15k-237
bash run.sh HousE FB15k-237 0 0 0 500 500 600 20 6 0.6 5.0 2.00378388680359 0.000794267891285676 100000 10000 16 0.00336727231946076
```

## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.
