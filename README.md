# GNOME
Graph Node Matching for Edit Distance\
Repository of code of the article "Graph Node Matching for Edit Distance", Pattern Recognition Elsevier.
This repo contains a supervised metric learning approach that combines Graph Neural Networks(GNN) and optimal transport to learn an approximation of the Graph Edit Distance (GED) in an end-to-end fashion. The model consists of two siamese GNNs and a comparison block. Each graph pair’s nodes are augmented by positional encoding and embedded by multiple Graph Isomorphism Network(GIN) layers. The obtained embeddings are
then compared through a Multi-Layer Perceptron and Linear Sum Assignement Problem(LSAP) solver applied on a node-wise Euclidean metric defined in the embedding space.\

![GNOME architecture](GNOME/archi_GED_OT.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [License](#license)

## Installation

Describe the steps to install your project here. Make sure to include any prerequisites needed and the commands to run to install the project.

## Usage
TO DO

## Data
Link for data repository: [data_folder_link](https://drive.google.com/drive/folders/1wiebwTGNJ3oNL1phEoL5TKk8b72gBJ0f?usp=sharing)

## License
