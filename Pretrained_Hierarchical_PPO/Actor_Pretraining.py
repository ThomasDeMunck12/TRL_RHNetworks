# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:13:46 2025

@author: thoma
"""

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#Pretrain first actor network

def pretrain_actor1(
    hpppo,
    train_expert_dataset1,
    test_expert_dataset1,
    batch_size=360,
    epochs=100,
    scheduler_gamma=0.995,
    log_interval=54,

    learning_rate=1.0e-4):
    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")
    
    def train(policy1_pt, device, train_loader, optimizer):
        policy1_pt.train()

        for batch_idx, (obs_dict, target) in enumerate(train_loader):
            data = {
                "p": obs_dict["p"].to(device, dtype=T.float32),
                "xy": obs_dict["xy"].to(device, dtype=T.float32),
                "t": obs_dict["t"].to(device, dtype=T.float32),

                }
            target = target.to(device)
            optimizer.zero_grad()

            action, _, _ = policy1_pt(data)  #PPO policy outputs actions, values, log_prob
            action_prediction = action.double()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data["p"]),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(policy1_pt, device, test_loader):
        policy1_pt.eval()
        test_loss = 0
        with T.no_grad():
            for (obs_dict, target) in test_loader:
                data = {
                    "p": obs_dict["p"].to(device, dtype=T.float32),
                    "xy": obs_dict["xyt"].to(device, dtype=T.float32),
                    "t": obs_dict["t"].to(device, dtype=T.float32),
                    }
                target = target.to(device)

                action, _, _ = policy1_pt(data)
                action_prediction = action.double()

                test_loss = criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    criterion = nn.MSELoss()

    # Extract initial policy
    policy1_pt = hpppo.policy1.to(device)
    
    # Load dataset
    train_loader = T.utils.data.DataLoader(
        dataset=train_expert_dataset1, batch_size=batch_size, shuffle=True, **kwargs
    )
    
    test_loader = T.utils.data.DataLoader(
        dataset=test_expert_dataset1,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Define an optimizer and a learning rate schedule.
    #optimizer = optim.Adadelta(policy_student.parameters(), lr=learning_rate)
    optimizer = optim.Adam(policy1_pt.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Train and test the policy model.
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(policy1_pt, device, train_loader, optimizer)
        #test(policy1_pt, device, test_loader)

        scheduler.step()

    # Implement the trained policy network back into the RL student agent
    hpppo.policy1 = policy1_pt
    
#Pretrain second actor network.

def pretrain_actor2(
    hpppo,
    train_expert_dataset2,
    test_expert_dataset2,
    batch_size=720,
    epochs=100,
    scheduler_gamma=0.995,
    log_interval=54,

    learning_rate=1.0e-4):
    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")
    
    def train(policy2_pt, device, train_loader, optimizer):
        policy2_pt.train()

        for batch_idx, (obs_dict, target) in enumerate(train_loader):
            data = {
                "p": obs_dict["p"].to(device, dtype=T.float32),
                "xy": obs_dict["xy"].to(device, dtype=T.float32),
                "t": obs_dict["t"].to(device, dtype=T.float32),
                }
            
            target = target.to(device)
            
            optimizer.zero_grad()

            action, _, _ = policy2_pt(data)  #PPO policy outputs actions, values, log_prob
            action_prediction = action.double()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data["p"]),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(policy2_pt, device, test_loader):
        policy2_pt.eval()
        test_loss = 0
        with T.no_grad():
            for (obs_dict, target) in test_loader:
                data = {
                    "p": obs_dict["p"].to(device, dtype=T.float32),
                    "xy": obs_dict["xy"].to(device, dtype=T.float32),
                    "t": obs_dict["t"].to(device, dtype=T.float32),
                    }
                
                target = target.to(device)

                action, _, _ = policy2_pt(data)
                action_prediction = action.double()

                test_loss = criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    criterion = nn.MSELoss()

    # Extract initial policy
    policy2_pt = hpppo.policy2.to(device)
    
    # Load dataset
    train_loader = T.utils.data.DataLoader(
        dataset=train_expert_dataset2, batch_size=batch_size, shuffle=True, **kwargs
    )
    
    test_loader = T.utils.data.DataLoader(
        dataset=test_expert_dataset2,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Define an optimizer and a learning rate schedule.
    #optimizer = optim.Adadelta(policy_student.parameters(), lr=learning_rate)
    optimizer = optim.Adam(policy2_pt.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Train and test the policy model.
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(policy2_pt, device, train_loader, optimizer)
        #test(policy2_pt, device, test_loader)

        scheduler.step()

    # Implement the trained policy network back into the RL student agent
    hpppo.policy2 = policy2_pt
    