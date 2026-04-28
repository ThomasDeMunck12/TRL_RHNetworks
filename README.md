# Reinforcement Learning with Pretraining for Pricing and Repositioning in Ride-Hailing Networks

This repository contains the code and data used to reproduce the numerical experiments from the paper: **"Reinforcement Learning with Pretraining for Pricing and Driver Repositioning in Ride-Hailing Networks"**

## Overview
This project implements a reinforcement learning framework for jointly optimizing pricing and vehicle repositioning decisions in ride-hailing networks. The approach combines:
- A Markov decision process (MDP) formulation.
- Two-timescale actor-critic methods based on Proximal Policy Optimization (PPO).
- Pretraining using expert policies derived from lookahead strategies.

## Data
The raw data used in Section 5 is publicly available from the NYC Taxi & Limousine Commission:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
We use the following datasets:
- February 2019 – High Volume For-Hire Vehicle Trip Records  
- March 2019 – High Volume For-Hire Vehicle Trip Records  
The scripts in the `data` folder preprocess these datasets to estimate model parameters such as arrival rates, travel times, and distances.

## Dependencies 
The project mainly relies on the following packages:
- Python 3.10+
- NumPy
- PyTorch
- Gymnasium
- Stable-Baselines3

## Core modules
These folders contain the main components of the methodology:
- `data`: Scripts to preprocess raw data and generate model parameters (arrival rates, distances, travel times), stored as NumPy files.
- `envs_1timescale`: Implementation of the singe-timescale MDP formulation using Gymnasium.
- 'envs_2timescale': Implementation of the two-timescale MDP formulation using Gymnasium
- `2timescale_ppo`: Implementation of the two-timescale actor-critic framework along with a customized PPO algorithm, building on Stable-Baselines3.
- `pretraining`: Implementation of actor and critic pretraining procedures tailored to the proposed two-timescale actor-critic framework.

## Additional modules
These folders contain scripts used to generate the results presented in the paper:
- `PPO_no_pretraining`: Training and evaluation of PPO agents without pretraining.
- `PPO_pretraining`: Pretraining of actor-critic networks followed by PPO training and evaluation.
- `LH_strategies`: Simulation of lookahead strategies used to generate expert state-decision datasets.
- `Experiments`: Scripts and output data used to generate figures and tables in Section 5. Subfolders correspond to the specific subsections of the paper.

## Installation
Clone the repository:
```bash
git clone https://github.com/ThomasDeMunck12/RL_wPT_RHNetworks/tree/main
cd RL_wPT_RHNetworks
