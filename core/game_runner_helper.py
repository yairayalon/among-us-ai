import copy

import numpy as np
import pandas as pd

from config.game_parser import GameParser
from config.constants import *


class GameRunnerHelper:
    FILE_JSON_HANDLER = None

    @staticmethod
    def get_file_data():
        GameRunnerHelper.FILE_JSON_HANDLER = \
            GameParser.parse_game_settings()
        return GameRunnerHelper.FILE_JSON_HANDLER

    @staticmethod
    def get_obs_names():
        grh = GameRunnerHelper
        colors = list(grh.FILE_JSON_HANDLER['colors'])
        max_rounds = len(colors) - 2
        max_bodies = len(colors) - grh.FILE_JSON_HANDLER['num_impostors'] - 2
        cm_obs = ['loc', 'time', 'time in round', 'round', 'num of agents in vision', 'table cooldown',
                   'table calls left', 'num tasks left', 'color', 'tasks left']
        imp_obs = ['loc', 'time', 'color', 'time in round', 'round', 'table cooldown', 'table calls left',
                    'kill cooldown', 'tasks faked', 'num of agents in vision',
                    'time faking tasks']
        cm_des = ['win', 'finished tasks']
        imp_des = ['win', 'killed']
        for i in range(1, max_rounds + 1):
            cm_des.append(f'survived round {i}')
            imp_des.append(f'survived round {i}')
            cm_des.append(f'voted impostor in round {i}')
            cm_des.append(f'votes received in round {i}')
            imp_des.append(f'votes received in round {i}')
            cm_des.append(f'surviving crewmates in round {i}')
            imp_des.append(f'surviving impostors in round {i}')
            cm_des.append(f'impostor ejected in round {i}')
            imp_des.append(f'crewmate ejected in round {i}')
            imp_obs.append(f'killed in round {i}')
            cm_obs.append(f'who died in round {i}')
            imp_obs.append(f'who died in round {i}')
            cm_obs.append(f'body reported by in round {i}')
            imp_obs.append(f'body reported by in round {i}')
            cm_obs.append(f'body reported in round {i}')
            imp_obs.append(f'body reported in round {i}')
            cm_obs.append(f'table used by in round {i}')
            imp_obs.append(f'table used by in round {i}')
            cm_obs.append(f'body loc according to reporter in round {i}')
            imp_obs.append(f'body loc according to reporter in round {i}')
            cm_obs.append(f'killer according to reporter in round {i}')
            imp_obs.append(f'killer according to reporter in round {i}')
            cm_obs.append(f'ejected in round {i}')
            imp_obs.append(f'ejected in round {i}')
            for j in range(1, max_bodies + 1):
                cm_obs.append(f'body {j} seen in round {i}')
                imp_obs.append(f'body {j} seen in round {i}')
                cm_obs.append(f'body {j} loc in round {i}')
                imp_obs.append(f'body {j} loc in round {i}')
                cm_obs.append(
                    f'body {j} time observed in round {i}')
                imp_obs.append(
                    f'body {j} time observed in round {i}')
        for c in colors:
            imp_obs.append(f'is impostor {c}')
            cm_obs.append(f'last loc seen {c}')
            imp_obs.append(f'last loc seen {c}')
            cm_obs.append(f'last time seen {c}')
            imp_obs.append(f'last time seen {c}')
            cm_obs.append(f'tasks num seen on {c}')
            imp_obs.append(f'tasks num seen on {c}')
            cm_obs.append(f'tasks seen on {c}')
            imp_obs.append(f'tasks seen on {c}')
            cm_obs.append(f'is dead {c}')
            imp_obs.append(f'is dead {c}')
            cm_obs.append(f'seen kill {c}')
            imp_obs.append(f'seen kill {c}')
            imp_obs.append(f'was seen killing by {c}')
            for i in range(1, max_rounds + 1):
                cm_obs.append(
                    f'percentage of round {i} {c} was seen')
                imp_obs.append(
                    f'percentage of round {i} {c} was seen')
                cm_obs.append(f'seen {c} kill in round {i}')
                imp_obs.append(f'seen {c} kill in round {i}')
                for c2 in colors:
                    cm_obs.append(f'{c} voted {c2} in round {i}')
                    imp_obs.append(f'{c} voted {c2} in round {i}')
        cm_obs.append('current action')
        imp_obs.append('current action')
        return cm_obs, cm_des, imp_obs, imp_des

    @staticmethod
    def read_dataframe(*args):
        dataframes = []
        for file in args:
            dataframes.append(pd.read_csv(file, low_memory=False))
        return pd.concat(dataframes)

    @staticmethod
    def get_coord_after_move(start_coord, move):
        if move == "move up":
            return start_coord[0], start_coord[1] - 1
        if move == "move down":
            return start_coord[0], start_coord[1] + 1
        if move == "move right":
            return start_coord[0] + 1, start_coord[1]
        if move == "move left":
            return start_coord[0] - 1, start_coord[1]
        if move == "move none":
            return start_coord

    @staticmethod
    def is_valid_coord(coord):
        return coord in valid_start_coords

    @staticmethod
    def get_first_dir_to_task(agent_coord, task_coord):
        return directs_map[(agent_coord, task_coord)]

    @staticmethod
    def get_valid_dirs(agent_coord):
        return [direct for direct in directs if
                GameRunnerHelper.is_valid_coord(
                    GameRunnerHelper.get_coord_after_move(agent_coord, direct))]

    @staticmethod
    # this function is also being used to get a direction to checkpoints coords
    def get_dir_to_task(agent_coord, task_coord):
        first_dir_to_task = GameRunnerHelper.get_first_dir_to_task(
            agent_coord, task_coord)
        random_dirs = directs.difference({first_dir_to_task})
        valid_random_dirs = [random_dir for random_dir in random_dirs if
                             GameRunnerHelper.is_valid_coord(
                                 GameRunnerHelper.get_coord_after_move(agent_coord, random_dir))]
        if not valid_random_dirs:
            return first_dir_to_task
        random_dir = np.random.choice(valid_random_dirs)
        return np.random.choice([first_dir_to_task, random_dir], p=[0.9, 0.1])

    @staticmethod
    def get_manhattan_dist(start_coord, end_coord):
        return abs(end_coord[0] - start_coord[0]) + abs(end_coord[1] -
                                                        start_coord[1])

    @staticmethod
    def get_closest_task_coord(agent_coord, tasks_coords=None):
        # relevant for crewmates
        if tasks_coords:
            tasks_coords_dists = {
                task_coord: GameRunnerHelper.get_manhattan_dist(
                    agent_coord, task_coord) for task_coord in tasks_coords
                if agent_coord != task_coord}
        # relevant for impostors
        else:
            tasks_coords_dists = {
                task_coord: GameRunnerHelper.get_manhattan_dist(
                    agent_coord, task_coord) for task_coord in
                all_tasks_coords}
        return min(tasks_coords_dists, key=tasks_coords_dists.get)

    @staticmethod
    def get_random_checkpoint_coord(agent_coord):
        if agent_coord in checkpoints_coords:
            checkpoints_coords_copy = copy.deepcopy(checkpoints_coords)
            checkpoints_coords_copy.remove(agent_coord)
            random_checkpoint_idx = np.random.choice(len(
                checkpoints_coords_copy))
            random_checkpoint_coord = checkpoints_coords_copy[
                random_checkpoint_idx]
            return random_checkpoint_coord
        random_checkpoint_idx = np.random.choice(len(checkpoints_coords))
        random_checkpoint_coord = checkpoints_coords[random_checkpoint_idx]
        return random_checkpoint_coord

    @staticmethod
    def is_move_valid(move, legal_moves):
        return move in legal_moves
