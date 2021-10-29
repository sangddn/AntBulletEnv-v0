import torch

from algos.pgb import PGB
from algos.vpg import VPG
from algos.ppo import PPO

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo', type=str, default='VPG',
        help='Choose one: VPG, PGB or PPO.'
    )
    args = parser.parse_args()

    bullet_env = "AntBulletEnv-v0"

    if args.algo == "VPG":
        trained = VPG(bullet_env)
    elif args.algo == "PGB":
        trained = PGB(bullet_env)
    elif args.algo == "PPO":
        trained = PPO(bullet_env)

    # Save trained model to current directory
    path = 'trained_' + args.algo + '.pt'
    torch.save(trained.state_dict(), path)
