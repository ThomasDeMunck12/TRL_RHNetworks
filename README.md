# Reinforcement Learning with Pretraining for Ride-Hailing Networks
This repository contains all the code supporting numerical experiments in "Reinforcement Learning with Pretraining for Pricing and Repositioning in Ride-Hailing Networks". 

The raw data used in Numerical Experiments can be accessed from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. See 'February, March 2019 - High Volume For-Hire Vehicle Trip Records" and "February, March 2019 - High Volume For-Hire Vehicle Trip Records". 

# Repository structure 
The folders with lower capital letters contain the main code to compute the results of the paper:
- data: code to derive parameters used in the paper, and final parameters (i.e., arrival rates, distances, and travel times parameters), saved as numpy files.
- envs: code to implement the Markov decision process described in the paper, using the package Gymnasium.
- ppo: code to implement our actor-critic framework, and our specific adaptation of the PPO algorithm with two decision levels. This code builds upon the implementation of the PPO algorithm presented in the package Stable-Baselines3
- pretraining: code to implement actor pretraining, and critic pretraining such as described in the paper. This code is specifically tailored to our adaptation of our PPO algorithm, but can be easily adapted to any DRL algorithm proposed by the package Stable-Baselines3.

The folders with upper capital letters contain additional code to derive the results of the paper:
- PPO_no_pretraining: Code to simulate the lookahead strategies in the MDP, and generate the state-decision datasets approximating the expert policy provided by one of the lookahead strategies.
- PPO_pretraining: Code to pretrain our actor-critic networks, and then continue the training process by actively learning using the PPO algorithm.
- LH_strategies: Code to simulate the lookahead strategies in the MDP, and generate the state-decision datasets approximating the expert policy provided by one of the lookahead strategies.
-Experiments: Code and output data to generate figures presented in Section 5. This folder is divided into subfolders, each of them representing a specific subsection of the article.

