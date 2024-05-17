# GNOME
Graph Node Matching for Edit Distance\
Repository of code of the article "Graph Node Matching for Edit Distance", Pattern Recognition Elsevier.
This repo contains a supervised metric learning approach that combines Graph Neural Networks(GNN) and optimal transport to learn an approximation of the Graph Edit Distance (GED) in an end-to-end fashion. The model consists of two siamese GNNs and a comparison block. Each graph pair’s nodes are augmented by positional encoding and embedded by multiple Graph Isomorphism Network(GIN) layers. The obtained embeddings are
then compared through a Multi-Layer Perceptron and Linear Sum Assignement Problem(LSAP) solver applied on a node-wise Euclidean metric defined in the embedding space.

![GNOME architecture](GNOME_img.png)

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [License](#license)
- [Acknoledgements](#acknoledgements)
- [Cite](#cite)

## Installation

python version: Python 3.9.7 \
torch version: Pytorch 1.11.0 \
CUDA Version: CUDA 11.6

## Data
Link for data repository: [data_folder_link](https://drive.google.com/drive/folders/1wiebwTGNJ3oNL1phEoL5TKk8b72gBJ0f?usp=sharing)

## Usage
First Download the Data repository above. \
In the following, /my_directory design the working directory. \
Following is the command line to launch a training of GNOME on Linux dataset : 
\
``` python python/main_GNOME.py --basedir /my_directory --dataset Linux_csv/Linux.csv```
\
Following is the command line to launch a training of GNOME on AIDS dataset : 
\
``` python python/main_GNOME.py --basedir /my_directory --dataset AIDS_csv/AIDS.csv```
\
Following is the command line to launch a training of GNOME on IMDB dataset : 
\
``` python python/main_GNOME.py --basedir /my_directory --dataset Linux_csv/Linux.csv```

\
Following is the command line to launch a training of GNOME on MAO dataset with cost1, consider --edge_features True for taking into account Edges features and using GINE instead of GIN : 
\
``` python python/main_GNOME.py --basedir /my_directory --dataset datasets_ged/MAO/mao_cost1.csv```
\
Other Hyperparameters can be used in the command line see the parser help for more indications.


## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

## Acknoledgements
The authors acknowledge the support of the French Agence Nationale de la Recherche
(ANR) under grant ANR-21-CE23-0025 (CoDeGNN project). The authors acknowledge the
support of the ANR and the Région Normandie under grant ANR-20-THIA-0021 (HAISCoDe
project).

## Cite
To cite our work, use the following bibtex

@Journal{ Moscatelli2024GNOME, title={Graph Node Matching for Edit Distance}, author={Moscatelli, Aldo and Piquenot, Jason and Berar,Maxime and Héroux, Pierre and Adam, Sébastien}, booktitle={Pattern Recognition Letters Elsevier}, year={2024}}
