# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:43:28 2026

@author: thoma
"""

print('Start task number: ', parameter)

import numpy as np 

from stable_baselines3.common.utils import obs_as_tensor

from gymnasium import spaces

import torch as T
from torch.utils.data.dataset import Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def pretrain_agent(
    student,
    batch_size=128,
    epochs=100,
    scheduler_gamma=0.8,
    learning_rate=1.0):
    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    criterion = nn.MSELoss()

    # Extract initial policy
    policy_student = student.policy.to(device)

    def train(policy_student, device, train_loader, optimizer):
        policy_student.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            action, _, _ = policy_student(data)  #PPO policy outputs actions, values, log_prob
            action_prediction = action.double()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 54 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    train_loader = T.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    # Define an optimizer and a learning rate schedule.
     #optimizer = optim.Adadelta(policy_student.parameters(), lr=learning_rate)
    optimizer = optim.Adam(policy_student.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Train and test the policy model.
    for epoch in range(1, epochs + 1):
        train(policy_student, device, train_loader, optimizer)
        scheduler.step()

    # Implement the trained policy network back into the RL student agent
    pppo.policy = policy_student