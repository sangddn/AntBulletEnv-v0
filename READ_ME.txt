-----------------
FILE ORGANIZATION
-----------------

This directory has 3 main files and 1 main folder:

1. report.pdf		Graph & comments.

2. hw1_A.py		Main file. 

3. misc.py		A miscellaneous file containing functions, classes and hyper-parameters. Any change to the parameters in this file will have an impact on ALL algorithms.

4. Folder algos:
	1. agent.py	Network classes.
	2. pgb.py	Policy Gradient with a Baseline.
	3. `ppo.py	Proxmial Policy Optimization.
	4. vpg.py	Vanilla Policy Gradient.

------
OTHERS
------

For informational purposes, I included three other folders. YOU CAN IGNORE THESE.
	
1. Folder `utils`		Containing utility functions. These are actual Python packages. I included the source code in here for clarity.

2. Folder `learning_data`		Epoch-level data recorded during my training of the algorithms. Used to produce the graph in this report.

3. Folder `trained_models`	Three trained PyTorch models. I trained all three for 2,000 epochs, at most 1,000 steps per each epoch.

------------
HOW TO TRAIN
------------

1. Load ALL folders and files to the directory you're running them on.

2. Adjust hyper-parameters as you'd like in `misc.py`. 
	WARNING: These algorithms might be highly sensitive to these parameters, especially VPG.

3. From that directory, run `python 'hw1_A.py' --algo "ALGORITHM"` in which `ALGORITHM` can be either VPG, PGB or PPO.

4. Enjoy your time waiting. Here's how long it took Colab to train my models (2,000 epochs):
	1. VPG: 49 minutes 34 seconds for a total of 1,692,714 steps.
	2. PGB: 1 hour 19 minutes 19 seconds for a total of 1,976,363 steps.
	3. PPO: 1 hour 13 minutes 4 seconds for a total of 1,992,747 steps.

4. After finishing training, the trained model will be saved directly to your directory. For example, `trained_VPG.pt`.






