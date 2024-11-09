# 2AMC15 Group 13

This is our final code for the course 2AMC15.

Please first install the requirements using:

```bash
pip install -r requirements.txt
```

## Agents

Within the `agents` directory you can find all the agents. These agents can be called within train.py or evaluate.py with multiple parameters (alpha, epsilon, gamma, ddqn). All parameters are self-explanatory except for ddqn, this parameters decides whether to use Double DQN to compute the target q-values.

Within the files of the agents themselves one can also choose to change parameters, such as hidden dimensions and kernel sizes for the CNNs; batch size and memory size for the experience replay buffer.

## Training

Within `train.py` there is a list called `agents` where you are able to specify which agents you want to run. There are already some agents present for you to comment in/out.
Next to the default arguments when calling train.py, two new arguments were added:

```bash
--save_agent_model  Whether to save the DQN network once done training  
                        
--cat               Whether to deploy a randomly moving obstacle on the grid
```

The training can be run using one of the following commands (these ones were used for the results in the report):

```bash
python train.py grid_configs/small.grd --out results/ --iter x --no_gui --save_agent_model
python train.py grid_configs/medium.grd --out results/ --iter x --no_gui --save_agent_model
python train.py grid_configs/house.grd --out results/ --iter x --no_gui --save_agent_model
```

Here `x` can be chosen (25000 and 40000) were used for the report for the DQN agents, (500000) can be used for the normal Q-Learning agent. You can also choose to leave out `--save_agent_model` and/or add `--cat`.

There are also 2 `.bat` files included to run experiments. These files can be ran using the `bash` command in the terminal: `bash normal_q_learning.bat`. For the `normal_q_learning.bat` all agents in the `agents` list within `train.py` should be commented out except for `QLearningAgent` like this:

```python
agents = [
            QLearningAgent(agent_number=0),
            # DQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=False
            # ),
            # DQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
            # DuelQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
            # PERDQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
            # PERDuelDQLAgent(
            #     agent_number=0,
            #     input_dim=(channels_used, len(obs), len(obs[0])),
            #     ddqn=True
            # ),
        ]
```

For the other file, `dqn_agents_training.bat` it should be the other way around:

```python
agents = [
            # QLearningAgent(agent_number=0),
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=False
            ),
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
            DuelQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
            PERDQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
            PERDuelDQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
        ]
```

Keep in mind that the training speed of the DQN models is HEAVILY dependent on the GPU. So the times that were managed in the report may not be case for you.

## Reproducing the score plot seen in the report

To reproduce the results from figure 4 in the report, uncomment all agents in the train.py file, and simply run:

```bash
python results.py
```

Please note that in the report we did not include the regular Q-learning agent in this plot. We advise to comment this agent out for getting the most accurate results:

```python
agents = [
            # QLearningAgent(agent_number=0),
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=False
            ),
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
            DuelQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
            PERDQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
            PERDuelDQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            ),
        ]
```


This will automatically create the plot of the 14x14 grid. It does not automatically show both plots due to time issues.  
If you wish to see the other grid (16x16), please change the number in `run_experiments(14)` to `run_experiments(16)` on line 221 and 227 in `results.py`.

## Evaluation of saved models

If models are saved, their path can be added to `model_paths` in `evaluate.py`. This file can then be run to see the evaluation of that model in the GUI. Here are some example commands:

```bash
python evaluate.py grid_configs/small.grd --cat
python evaluate.py grid_configs/medium.grd --cat
python evaluate.py grid_configs/house.grd --cat
```

If evaluating more models at once, be sure that the index of the path in the `model_paths` list corresponds to the index of the agent in the `agents` list. The model names reflect to which kind of agents they belong, example:

```python
model_paths = ['models/BaseModel-small-2023-06-23__15-09-08.pth', 'models/BaseModelWithDouble-small-2023-06-23__15-15-48.pth']
```

belong to

```python
agents = [
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=False
            ),
            DQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            )
        ]
```

respectively. As can be seen in the path names of the model, these models are trained on the `small.grd` grid, so they should also be evaluated on these grids. So a model that is trained on a different grid should be evaluated seperately:

```python
model_paths = ['models/DuelingPERModelWithDouble-medium-2023-06-23__16-01-35.pth']
```

belong to

```python
agents = [
            PERDuelDQLAgent(
                agent_number=0,
                input_dim=(channels_used, len(obs), len(obs[0])),
                ddqn=True
            )
        ]
```