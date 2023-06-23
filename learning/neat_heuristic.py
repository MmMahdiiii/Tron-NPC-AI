import neat
import os
import pickle
import subprocess
import random
import sys
import json
import os
import threading
import time
import json

directory = os.getcwd()
server_directory = directory + '\\PythonServer'
client_directory = directory + '\\PythonClient'

# TODO : peek a random map
# TODO : run num_servers servers (pop_size % num_servers == 0)
# TODO : each neural network would dump to a file
# TODO : generate python client for each neural network
# TODO : run num_servers clients (pop_size % num_servers == 0)
# TODO : each client would load the neural network from the file
# TODO : we have to assign each client to a server (port)
# TODO : they have to play against two zigzag bot and GeneticMinimax bot

maps = []


def init_maps():
    maps_path = server_directory + '\\maps\\'
    for filename in os.listdir(maps_path):
        maps.append('maps/' + filename)


def generate_servers_config(num_servers):
    #     read the config gamecfg.json
    #     change the ports of the configs to be different
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gamecfg.json"
    )
    chosen_map = random.choice(maps)
    with open(config_path, 'r') as f:
        config = json.load(f)
    for i in range(num_servers):
        config['net']['port'] = 8000 + i
        config['gui']['port'] = 9000 + i
        #        write the config to a new file
        file_name = 'gamecfg' + str(i) + '.json'
        # dump the configs to server_configs folder
        config['game_handler']['map'] = chosen_map

        file_path = server_directory + '\\server_configs\\' + file_name
        with open(file_path, 'w') as outfile:
            json.dump(config, outfile)


def run_server(num_servers):
    #   give each server a config file_path as an argument
    #   change the directory to the server directory
    generate_servers_config(num_servers)
    os.chdir(server_directory)
    server_processes = []
    results = [open('server_results\\server' + str(i) + '.txt', 'w') for i in range(num_servers)]
    for i in range(num_servers):
        file_name = 'gamecfg' + str(i) + '.json'
        file_path = 'server_configs\\' + file_name
        server_processes.append(subprocess.Popen(['python', 'main.py', file_path], stdout=results[i]))
    return server_processes

def run_clients(genomes, num_servers, config):
    #     dump each nn to a file with name nn_i in nn folder
    #     generate a config file for each client in configs folder
    #     relocate to the client directory
    #     run the clients

    os.chdir(client_directory)

    with open('gamecfg.json', 'w') as sample_config:
        client_config = json.load(sample_config)

    counter = 0
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        file_name = 'nn_' + str(counter) + '.pkl'
        file_path = 'nn\\' + file_name
        with open(file_path, 'wb') as output:
            pickle.dump(net, output, 1)

        # generate a config file for each client in configs folder
        config_path = 'configs\\' + 'gamecfg' + str(counter) + '.json'
        client_config['ai']['nn_path'] = file_path
        client_config['net']['port'] = 8000 + counter % num_servers

        with open(config_path, 'w') as outfile:
            json.dump(client_config, outfile)

        subprocess.Popen(['python', 'main.py', config_path])
        counter += 1






#
# def eval_genomes(genomes, config):
#     global WIN, gen
#     win = WIN
#     gen += 1
#
#     nets = []
#     ge = []
#     for genome_id, genome in genomes:
#         genome.fitness = 0  # start with fitness level of 0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         nets.append(net)
#         birds.append(Bird(230, 350))
#         ge.append(genome)
#
#     base = Base(FLOOR)
#     pipes = [Pipe(700)]
#     score = 0
#
#     clock = pygame.time.Clock()
#
#     run = True
#     while run and len(birds) > 0:
#         clock.tick(30)
#
#         pipe_ind = 0
#         if len(birds) > 0:
#             if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[
#                 0].PIPE_TOP.get_width():  # determine whether to use the first or second
#                 pipe_ind = 1  # pipe on the screen for neural network input
#
#         for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
#             ge[x].fitness += 0.1
#             bird.move()
#
#             # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
#             output = nets[birds.index(bird)].activate(
#                 (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
#
#             if output[
#                 0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
#                 bird.jump()
#
#         base.move()
#
#         rem = []
#         add_pipe = False
#         for pipe in pipes:
#             pipe.move()
#             # check for collision
#             for bird in birds:
#                 if pipe.collide(bird, win):
#                     ge[birds.index(bird)].fitness -= 1
#                     nets.pop(birds.index(bird))
#                     ge.pop(birds.index(bird))
#                     birds.pop(birds.index(bird))
#
#             if pipe.x + pipe.PIPE_TOP.get_width() < 0:
#                 rem.append(pipe)
#
#             if not pipe.passed and pipe.x < bird.x:
#                 pipe.passed = True
#                 add_pipe = True
#
#         if add_pipe:
#             score += 1
#             # can add this line to give more reward for passing through a pipe (not required)
#             for genome in ge:
#                 genome.fitness += 5
#             pipes.append(Pipe(WIN_WIDTH))
#
#         for r in rem:
#             pipes.remove(r)
#
#         for bird in birds:
#             if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
#                 nets.pop(birds.index(bird))
#                 ge.pop(birds.index(bird))
#                 birds.pop(birds.index(bird))
#
#         draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)
#
#         # break if score gets large enough
#         '''if score > 20:
#             pickle.dump(nets[0],open("best.pickle", "wb"))
#             break'''
#
#
# def run(config_file):
#     """
#     runs the NEAT algorithm to train a neural network to play flappy bird.
#     :param config_file: location of config file
#     :return: None
#     """
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                                 config_file)
#
#     # Create the population, which is the top-level object for a NEAT run.
#     p = neat.Population(config)
#
#     # Add a stdout reporter to show progress in the terminal.
#     p.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     p.add_reporter(stats)
#     # p.add_reporter(neat.Checkpointer(5))
#
#     # Run for up to 50 generations.
#     winner = p.run(eval_genomes, 50)
#
#     # show final stats
#     print('\nBest genome:\n{!s}'.format(winner))
#
#
# if __name__ == '__main__':
#     # Determine path to configuration file. This path manipulation is
#     # here so that the script will run successfully regardless of the
#     # current working directory.
#     local_dir = os.path.dirname(__file__)
#     config_path = os.path.join(local_dir, 'config-feedforward.txt')
#     run(config_path)

