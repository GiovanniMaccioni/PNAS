import argparse
import torch
import evaluate_correlation as ec
import utils as U

config ={}

if __name__=="__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Correlation')
    parser.add_argument("--dir_path", type=str, default="random/run0", help='')

    #Arguments cnn
    parser.add_argument("--seed", type=int, default=10, help='Set the seed for reproducibility')
    parser.add_argument("--num_blocks", type=int, default=2, help='Number of Blocks')
    parser.add_argument("--k", type=int, default=256, help='Number of models for the next step')
    parser.add_argument("--t", type=int, default=20, help='Number of repeated trys')
    parser.add_argument("--ensemble", default=True, action='store_true', help='True use Ensemble')

    args = parser.parse_args()

    config['dir_path']=args.dir_path

    #Populate config
    config['seed'] = args.seed
    config['num_blocks'] = args.num_blocks
    config['k'] = args.k
    config['t'] = args.t
    config['ensemble'] = args.ensemble

    EPOCHS_PREDICTOR = 5

    U.set_reproducibility(config['seed'])
    ec.correlation_(2+config['num_blocks']-1, 8, config['num_blocks'], config['k'], config['t'], EPOCHS_PREDICTOR, config['dir_path'], config['ensemble'], device)