# Split Screen Communication

This is a PyTorch demonstration of split-screen communication learning.

## Usage

To view trained communication model running:
```
OMP_NUM_THREADS=1 python3 main.py --env_name "BreakoutDeterministic-v4" --num_processes 0 --max_vocab_size 40 --dirichlet_vocab False --load True --load_file "model_comm" --save False --demonstrate True --max_episode_length 3000
```
^Note: you will see the whole screen rendered; but agent with controls only sees portion of screen from paddle to bottom of screen, and other agent with no controls (other than sending a single discrete integer msg per timestep) only sees portion of screen above the paddle.

to view tensorboard of trained communication model:
tensorboard --logdir tmp/tb_log_comm --port 6006


To view baseline with disabled communication running:
```
OMP_NUM_THREADS=1 python3 main.py --env_name "BreakoutDeterministic-v4" --num_processes 0 --max_vocab_size 1 --dirichlet_vocab False --load True --load_file "model_no_comm" --save False --demonstrate True --max_episode_length 3000
```

to view tensorboard of baseline with disabled communication:
tensorboard --logdir tmp/tb_log_no_comm_baseline --port 6007


To train your own communication model:
```
sudo OMP_NUM_THREADS=1 nohup python3 main.py --env_name "BreakoutDeterministic-v4" --num_processes 63 --max_vocab_size 40 --dirichlet_vocab False --save True > /dev/null 2>&1&
```


Note:
Install version '0.1.12+49ec984' of PyTorch via this command: 
`
pip3 install git+https://github.com/pytorch/pytorch@49ec984c406e67107aae2891d24c8839b7dc7c33
` 
I can't guarantee the same performance in other versions of pytorch.

If you want to use aws, the ami of the image used is ami-c77752d1 with name "new-brodcast-pt-new-gym-sk-image"

## Results

With 63 processes with communication enabled it obtains average reward per episode of 320 (3.5 times that of baseline with disabled communication) for BreakoutDeterministic-v4 after 108 hours of training on aws m4.16xlarge cpu instance.

Comm Enabled
![Commmunication_enabled BreakoutDeterministic-v4](images/comm_enabled.png)

Comm Disabled (Baseline)
![Commmunication_disabled BreakoutDeterministic-v4](images/comm_disabled_baseline.png)
