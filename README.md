# SFDIA using Graph Convolution Networks and Autoencoders

Welcome to the repository for sensor fault detection, isolation, and accommodation using Graph Convolution Networks (GCNs) and Autoencoders. This repository contains two main Python projects and two datasets to support the analysis and development.


![alt text]([SFDIA.png](https://github.com/h777d/SFDIA-using-Graph-Convolution-Networks-and-Autoencoders/blob/09c6cfc4a3418acf5c8f153a81fa193b879d603b/SFDIA.png))

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
  - [PeMSD8 Dataset](#pemsd8-dataset)
  - [Water Tank Network Dataset](#water-tank-network-dataset)
- [Graph Convolution Network](#graph-convolution-network)
- [Autoencoder](#autoencoder)
- [Installation](#installation)


## Introduction

This repository provides implementations of two machine learning models for sensor fault detection, isolation, and accommodation in digital twins. The models used are:

1. **Graph Convolution Network (GCN)**
2. **Autoencoder**

The primary aim is to identify faults in sensor networks, isolate the faulty sensors, and accommodate these faults to ensure the robustness of the digital twin systems.

## Datasets

### PeMSD8 Dataset

The PeMSD8 dataset is used for traffic speed prediction. It contains data from various sensors placed on highways. This dataset helps in understanding and developing models for fault detection in a large-scale sensor network.

### Water Tank Network Dataset

This dataset includes measurements from a network of 50 pressure sensors placed in a water tank system. It provides a basis for developing and testing fault detection algorithms in fluid dynamics scenarios.

## Graph Convolution Network

The GCN-based model leverages the structure of the sensor network to detect and isolate faults. It captures the spatial relationships between sensors, making it suitable for applications where sensor placement and network topology play a critical role.


## Autoencoder

The autoencoder model is used for anomaly detection in sensor readings. It learns the normal behavior of the system and identifies deviations from this behavior, which indicate potential faults.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/h777d/SFDIA-using-Graph-Convolution-Networks-and-Autoencoders.git
   cd sensor-fault-detection
   ```
Install the required packages:
```
bash
pip install -r requirements.txt
```
