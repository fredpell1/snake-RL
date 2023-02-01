from controller import user_mode, agent_mode
from util import *
from agents import heuristic_agent, mc_agent, td_agent
from argparse import ArgumentParser

def main(new, filename, n_episodes):
    if new:
        agent = initialize_new_agent(
            td_agent.TDLambdaNN,
            epsilon=0.1,
            gamma = 0.5,
            learning_rate = 0.00001,
            lambda_ = 0.1,
            input_size = 6,
            hidden_size = 50,
            value_function=None,
            optimizer=None,
            loss_function = None
            )
    else:
        agent = load_agent(
            td_agent.TDLambdaNN,
            filename=filename,
            epsilon=0.1,
            gamma = 0.5,
            learning_rate = 0.00001,
            lambda_ = 0.1,
            input_size = 6,
            hidden_size = 50
        )
    train_and_save(agent, n_episodes, 500, filename)
    test(agent, 1, 100)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--new', action='store_true')
    parser.add_argument('-f', '--file')
    parser.add_argument('-e', '--episodes', type=int)
    args = parser.parse_args()
    main(args.new, args.file, args.episodes)