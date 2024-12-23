import copy
import os

from core.crewmate_game_flow import CrewmateGameFlow
from core.impostor_game_flow import ImpostorGameFlow
from config.constants import *
from agents.crewmate import Crewmate
from core.game_runner_helper import GameRunnerHelper as Grh
from agents.impostor import Impostor
from map.board import Board
import pandas as pd
from core.vote_flow import *
from agents.crewmate_neural_network import CrewmateNeuralNetwork
from agents.impostor_neural_network import ImpostorNeuralNetwork


class GameRunner:
    living_crewmates_counter = 0
    living_impostors_counter = 0

    cm_dict = None
    imp_dict = None
    file_data = None
    board = None
    crewmate_game_flow = None
    impostor_game_flow = None
    crewmate_nn_dm = None
    impostor_nn_dm = None
    cm_desired = None
    imp_desired = None
    pair_votes = set()
    tasks_per_crewmate = -1
    cur_round_cycles = 1
    cur_round = 1
    total_cycles = 1
    file_num = 0
    cm_rows = 0
    imp_rows = 0
    agents = dict()
    all_agents = dict()
    crewmates = []
    impostors = []
    directs_map = directs_map
    agents_votes = dict()
    ejected_now = None
    is_gui = False

    @staticmethod
    def init_game(is_gui):
        GameRunner.is_gui = is_gui
        GameRunner.file_data = Grh.get_file_data()
        if GameRunner.file_data["num_crewmates_tree"]:
            GameRunner.crewmate_game_flow = CrewmateGameFlow(
                GameRunner.file_data["tasks_per_agent"])
        if GameRunner.file_data["num_impostors_tree"]:
            GameRunner.impostor_game_flow = ImpostorGameFlow()
        GameRunner.tasks_per_crewmate = GameRunner.file_data['tasks_per_agent']

        # When running games with neural network agents, the parameter 'train'
        # determines the network's way of training:
        # - train = 0 => fits on the csv files in the dir
        #                crewmate/impostor_training and exports the fitted
        #                model to the binary file crewmate/impostor
        # - train = 1 => loads into the network the fitted network from
        #                the binary file crewmate/impostor
        # - train = 2 => loads into the network the fitted network from
        #                the binary file crewmate/impostor, then performs
        #                fit_update on the csv files in the dir
        #                crewmate/impostor_training and then exports the fitted
        #                model to the binary file crewmate/impostor
        if GameRunner.file_data["num_crewmates_nn"]:
            train = 1
            GameRunner.crewmate_nn_dm = CrewmateNeuralNetwork(train=train)
        if GameRunner.file_data["num_impostors_nn"]:
            train = 1
            GameRunner.impostor_nn_dm = ImpostorNeuralNetwork(train=train)

        cm_obs, GameRunner.cm_desired, imp_obs, GameRunner.imp_desired = (
            Grh.get_obs_names())
        GameRunner.cm_dict = {ver: [] for ver in cm_obs}
        GameRunner.imp_dict = {ver: [] for ver in imp_obs}
        GameRunner.cm_dict = {ver: [] for ver in cm_obs}
        GameRunner.imp_dict = {ver: [] for ver in imp_obs}
        GameRunner.board = Board(GameRunner.file_data["length"],
                                 GameRunner.file_data["width"],
                                 GameRunner.generate_agents(),
                                 GameRunner.file_data[
                                     "wall_locs"], GameRunner.file_data[
                                     "tasks"],
                                 GameRunner.file_data[
                                     "table"], cm_obs, imp_obs)
        for agent in GameRunner.crewmates:
            agent.board = GameRunner.board

    @staticmethod
    def run_game():
        GameRunner.cur_round_cycles += 1
        GameRunner.total_cycles += 1
        board_after = set()
        dict_plays = dict()
        index_plays = 0
        GameRunner.pair_votes.clear()
        if GameRunner.cur_round_cycles < MAX_ROUNDS_TILL_VOTE:
            living_agents = copy.copy(list(GameRunner.agents.values()))
            for agent in living_agents:
                if agent.is_dead:
                    continue
                agent_observation = GameRunner.board.get_observation(agent)
                if agent.agent_type:
                    GameRunner.board.board_update_impostor_observation(
                        agent, GameRunner.total_cycles, GameRunner.cur_round,
                        GameRunner.cur_round_cycles)
                else:
                    GameRunner.board.board_update_crewmate_observation(
                        agent, GameRunner.total_cycles, GameRunner.cur_round,
                        GameRunner.cur_round_cycles)
                agent_moves = GameRunner.board.get_pos_plays(agent)
                if agent.agent_type == IMPOSTOR:
                    if agent.decider_type == "dec_tree_dm":
                        try:
                            chosen_action = \
                                GameRunner.impostor_game_flow.get_chosen_act(
                                    GameRunner.board.get_observation(agent),
                                    agent_moves,
                                    GameRunner.agents, GameRunner.crewmates,
                                    agent)
                        except:
                            chosen_action = np.random.choice(list(agent_moves))
                    elif agent.decider_type == "nn_dm":
                        try:
                            chosen_action = GameRunner.impostor_nn_dm. \
                                decide_action(agent_observation, agent_moves)
                        except:
                            chosen_action = np.random.choice(list(agent_moves))
                    else:
                        chosen_action = np.random.choice(
                            np.array(list(agent_moves)))
                else:
                    if agent.decider_type == "dec_tree_dm":
                        try:
                            chosen_action = \
                                GameRunner.crewmate_game_flow.get_chosen_act(
                                    GameRunner.board.get_observation(agent),
                                    agent_moves,
                                    GameRunner.agents, GameRunner.crewmates,
                                    agent)
                        except:
                            chosen_action = np.random.choice(list(agent_moves))
                    elif agent.decider_type == "nn_dm":
                        try:
                            chosen_action = GameRunner.crewmate_nn_dm. \
                                decide_action(agent_observation, agent_moves)
                        except:
                            chosen_action = np.random.choice(list(agent_moves))
                    else:
                        chosen_action = np.random.choice(
                            np.array(list(agent_moves)))
                GameRunner.export_observation(agent, chosen_action)
                lst_sp = chosen_action.split()
                if lst_sp[0] == "move":
                    move_cors = \
                        GameRunner.board.move_agent(agent, chosen_action)
                    dict_plays[index_plays] = move_cors
                elif lst_sp[0] == "report":
                    GameRunner.update_agents_observations()
                    reporter = agent.color
                    body_reported = lst_sp[1]
                    reporter_obs = GameRunner.board.get_observation(agent)
                    if agent.decider_type == "nn_dm":
                        if agent.agent_type:
                            obj_dm = GameRunner.impostor_nn_dm
                        else:
                            obj_dm = GameRunner.crewmate_nn_dm
                        setter = GameRunner.board.get_possible_killers(agent)
                        setter = setter.union({"none declared"})
                        try:
                            killer = obj_dm.decide_action(reporter_obs, setter)
                            body_loc = obj_dm.decide_action(
                                reporter_obs,
                                GameRunner.board.get_possible_body_locs())
                        except:
                            killer, body_loc = -1, -1
                    else:
                        setter = GameRunner.board.get_possible_killers(agent)
                        setter = setter.union({"none declared"})
                        setter = list(setter)
                        locs = tuple(GameRunner.board.get_possible_body_locs())
                        body_loc = locs[np.random.choice(len(locs))]
                        killer = np.random.choice(setter)
                    GameRunner.export_observation(agent, body_loc)
                    GameRunner.export_observation(agent, killer)
                    GameRunner.board.meeting_update_observations(
                        GameRunner.cur_round, reporter, body_reported,
                        caller_loc=body_loc, caller_killer=killer)
                    for i in GameRunner.agents_voting(agent, lst_sp[1]):
                        board_after.add(i)
                    board_after.update(
                        GameRunner.board.round_end_update_observations(
                            GameRunner.cur_round, GameRunner.pair_votes,
                            GameRunner.ejected_now
                        ))
                    GameRunner.cur_round += 1
                elif lst_sp[0] == "kill":
                    GameRunner.agents[agent.color].kill_cd = \
                        IMPOSTOR_KILL_COOLDOWN
                    kill_cors = GameRunner.eject_agent(lst_sp[1], agent)
                    board_after.update(kill_cors)
                    dict_plays[index_plays] = kill_cors
                elif lst_sp[0] == "call_meeting":
                    GameRunner.update_agents_observations()
                    agent.table_calls_left -= 1
                    reporter = agent.color
                    reporter_obs = GameRunner.board.get_observation(agent)
                    if agent.decider_type == "nn_dm":
                        if agent.agent_type == IMPOSTOR:
                            obj_dm = GameRunner.impostor_nn_dm
                        else:
                            obj_dm = GameRunner.crewmate_nn_dm
                        setter = GameRunner.board.get_possible_killers(agent)
                        setter = setter.union({"none declared"})
                        try:
                            killer = obj_dm.decide_action(reporter_obs, setter)
                        except:
                            killer, body_loc = -1, -1
                    else:
                        setter = GameRunner.board.get_possible_killers(agent)
                        setter = setter.union({"none declared"})
                        setter = list(setter)
                        killer = np.random.choice(setter)
                    GameRunner.board.meeting_update_observations(
                        GameRunner.cur_round, table_user=reporter,
                        caller_killer=killer)
                    rs = GameRunner.agents_voting(agent, None)
                    for r in rs:
                        board_after.add(r)
                    board_after.update(
                        GameRunner.board.round_end_update_observations(
                            GameRunner.cur_round, GameRunner.pair_votes,
                            GameRunner.ejected_now
                        ))
                    GameRunner.cur_round += 1
                elif lst_sp[0] == "do_task":
                    agent.do_task()
                index_plays += 1
        else:
            GameRunner.update_agents_observations()
            GameRunner.board.meeting_update_observations(
                GameRunner.cur_round)
            for v in GameRunner.agents_voting():
                board_after.add(v)
            board_after.update(GameRunner.board.round_end_update_observations(
                GameRunner.cur_round, GameRunner.pair_votes,
                GameRunner.ejected_now
            ))
            GameRunner.cur_round += 1
        winning_team = GameRunner.is_game_ended()
        if winning_team:
            GameRunner.update_agents_observations()
            cm_df, imp_df = GameRunner.insert_final_values(winning_team)
            cm_df.to_csv(
                f'training/crewmate_training{os.path.sep}crewmates{GameRunner.file_num}.csv')
            imp_df.to_csv(
                f'training/impostor_training{os.path.sep}impostors{GameRunner.file_num}.csv')
            GameRunner.reset_runner()
            return winning_team, dict_plays, board_after
        for impostor in GameRunner.impostors:
            impostor.kill_cd = max(impostor.kill_cd - 1, 0)
            impostor.table_cd = max(impostor.table_cd - 1, 0)
        for crewmate in GameRunner.crewmates:
            crewmate.table_cd = max(crewmate.table_cd - 1, 0)
        GameRunner.board.remove_kill_marks()

        if GameRunner.is_gui:
            return dict_plays, board_after
        else:
            return 0, 0, 0

    @staticmethod
    def reset_runner():
        GameRunner.living_crewmates_counter = 0
        GameRunner.living_impostors_counter = 0
        GameRunner.cur_round_cycles = 1
        GameRunner.cur_round = 1
        GameRunner.total_cycles = 1
        GameRunner.file_num += 1
        GameRunner.cm_rows = 0
        GameRunner.imp_rows = 0
        GameRunner.agents = dict()
        GameRunner.all_agents = dict()
        GameRunner.crewmates = []
        GameRunner.impostors = []
        GameRunner.agents_votes = dict()
        GameRunner.ejected_now = None
        cm_obs, GameRunner.cm_desired, imp_obs, GameRunner.imp_desired = (
            Grh.get_obs_names())
        GameRunner.cm_dict = {ver: [] for ver in cm_obs}
        GameRunner.imp_dict = {ver: [] for ver in imp_obs}
        GameRunner.board = Board(GameRunner.file_data["length"],
                                 GameRunner.file_data["width"],
                                 GameRunner.generate_agents(),
                                 GameRunner.file_data[
                                     "wall_locs"], GameRunner.file_data[
                                     "tasks"],
                                 GameRunner.file_data[
                                     "table"], cm_obs, imp_obs)
        for agent in GameRunner.crewmates:
            agent.board = GameRunner.board

    @staticmethod
    def agents_voting(reporter=None, dead_agent=None):
        pos_votes = GameRunner.board.get_possible_votes()
        agents_votes = {agent: 0 for agent in GameRunner.agents}
        for c, agent in GameRunner.agents.items():
            if not agent.is_dead:
                if agent.agent_type:
                    if agent.decider_type == 'nn_dm':
                        agent_to_vote = \
                            GameRunner.impostor_nn_dm.decide_action(
                                GameRunner.board.get_observation(agent),
                                pos_votes).split()[1]
                    elif agent.decider_type == 'dec_tree_dm':
                        living_agents = copy.copy(
                            list(GameRunner.agents.values()))
                        try:
                            agent_to_vote = impostor_vote(
                                GameRunner.board.get_observation(agent),
                                living_agents,
                                GameRunner.cur_round,
                                len(GameRunner.crewmates) - 2,
                                agent, reporter,
                                dead_agent,
                                GameRunner.impostors,
                                GameRunner.board)
                        except:
                            agent_to_vote = GameRunner.crewmates[
                                np.random.choice(
                                    len(GameRunner.crewmates))].color
                    else:
                        agent_to_vote = np.random.choice(
                            np.array(list(pos_votes))).split()[1]
                else:
                    if agent.decider_type == 'nn_dm':
                        agent_to_vote = \
                            GameRunner.crewmate_nn_dm.decide_action(
                                GameRunner.board.get_observation(agent),
                                pos_votes).split()[1]
                    elif agent.decider_type == 'dec_tree_dm':
                        living_agents = copy.copy(
                            list(GameRunner.agents.values()))
                        # if not  agent.agent_type:
                        try:
                            agent_to_vote = get_crew_vote(
                                GameRunner.board.get_observation(agent),
                                living_agents,
                                GameRunner.cur_round,
                                len(GameRunner.crewmates) - 2,
                                agent, reporter,
                                dead_agent,
                                GameRunner.board)
                        except:
                            excluded_myself = [ag for ag in living_agents
                                               if ag != agent]
                            agent_to_vote = excluded_myself[
                                np.random.choice(len(excluded_myself))].color
                    else:
                        agent_to_vote = np.random.choice(
                            np.array(list(pos_votes))).split()[1]
                    GameRunner.export_observation(agent,
                                                  f'vote {agent_to_vote}')
                    GameRunner.pair_votes.add((c, agent_to_vote))
                try:
                    agents_votes[agent_to_vote] += 1
                except:
                    agents_v = [ag for ag in GameRunner.agents if ag !=
                                agent.color]
                    agents_votes[np.random.choice(agents_v)] += 1

        agent_color_to_eject = max(agents_votes, key=agents_votes.get)
        GameRunner.ejected_now = agent_color_to_eject
        r = GameRunner.eject_agent(agent_color_to_eject)
        for imp in GameRunner.impostors:
            imp.table_cd = TABLE_CD
            imp.kill_cd = IMPOSTOR_KILL_COOLDOWN
        for crewmate in GameRunner.crewmates:
            crewmate.table_cd = TABLE_CD
            crewmate.is_performing_task = False
            crewmate.task_time = 0
        board_after = set()
        for i in r:
            board_after.add(i)
        GameRunner.cur_round_cycles = 1
        return board_after

    @staticmethod
    def update_agents_observations():
        for agent in GameRunner.crewmates:
            GameRunner.board.board_update_crewmate_observation(agent,
                                                               GameRunner.total_cycles,
                                                               GameRunner.cur_round,
                                                               GameRunner.cur_round_cycles)
        for impostor in GameRunner.impostors:
            GameRunner.board.board_update_impostor_observation(impostor,
                                                               GameRunner.total_cycles,
                                                               GameRunner.cur_round,
                                                               GameRunner.cur_round_cycles)

    @staticmethod
    def eject_agent(agent_color_to_eject, killer=None):
        if GameRunner.agents[agent_color_to_eject].agent_type == \
                CREWMATE:
            GameRunner.crewmates.remove(GameRunner.agents[
                                            agent_color_to_eject])
            GameRunner.living_crewmates_counter -= 1
        else:
            GameRunner.impostors.remove(GameRunner.agents[
                                            agent_color_to_eject])
            GameRunner.living_impostors_counter -= 1
        GameRunner.agents[agent_color_to_eject].is_dead = True
        after = GameRunner.board.kill_agent(agent_color_to_eject, killer)
        del GameRunner.agents[agent_color_to_eject]
        return after

    @staticmethod
    def is_game_ended():
        if GameRunner.living_impostors_counter >= \
                GameRunner.living_crewmates_counter:
            return IMPOSTORS_WIN
        if GameRunner.living_impostors_counter == 0:
            return CREWMATES_WIN
        for crewmate in GameRunner.crewmates:
            if crewmate.tasks:
                return GAME_CONTINUE
        return CREWMATES_WIN

    @staticmethod
    def generate_agents():
        agents_colors = GameRunner.file_data["colors"]
        impostors_colors = np.random.choice(agents_colors,
                                            GameRunner.file_data[
                                                "num_impostors"],
                                            replace=False)
        imps_colors = [color for color in impostors_colors]
        crewmates_colors = list(set(agents_colors).difference(set(
            impostors_colors)))
        cms_colors = [color for color in crewmates_colors]
        crewmates_trees = np.random.choice(cms_colors,
                                           GameRunner.file_data[
                                               "num_crewmates_tree"],
                                           replace=False)
        cms_colors = list(set(cms_colors).difference(set(
            crewmates_trees)))
        impostors_trees = np.random.choice(imps_colors,
                                           GameRunner.file_data[
                                               "num_impostors_tree"],
                                           replace=False)
        imps_colors = list(set(imps_colors).difference(set(
            impostors_trees)))
        crewmates_nns = np.random.choice(cms_colors,
                                         GameRunner.file_data[
                                             "num_crewmates_nn"],
                                         replace=False)
        cms_colors = list(set(cms_colors).difference(set(
            crewmates_nns)))
        impostors_nns = np.random.choice(imps_colors,
                                         GameRunner.file_data[
                                             "num_impostors_nn"],
                                         replace=False)
        imps_colors = list(set(imps_colors).difference(set(
            impostors_nns)))
        crewmates_randoms = np.random.choice(cms_colors,
                                             GameRunner.file_data[
                                                 "num_crewmates_random"],
                                             replace=False)
        impostors_randoms = np.random.choice(imps_colors,
                                             GameRunner.file_data[
                                                 "num_impostors_random"],
                                             replace=False)
        tasks_items = list(GameRunner.file_data["tasks"].items())
        for agent_color in agents_colors:
            if agent_color in list(impostors_nns):
                decision_maker = "nn_dm"
                GameRunner.agents[agent_color] = Impostor(
                    agent_color, IMPOSTOR,
                    IMPOSTOR_VISION_RANGE,
                    IMPOSTOR_KILL_COOLDOWN, decision_maker)

                GameRunner.impostors.append(GameRunner.agents[agent_color])
                GameRunner.living_impostors_counter += 1
            elif agent_color in list(crewmates_nns):
                decision_maker = "nn_dm"
                tasks_indices = np.random.choice(len(GameRunner.file_data[
                                                         "tasks"]),
                                                 GameRunner.tasks_per_crewmate,
                                                 replace=False)
                agent_tasks = {tasks_items[task_index][0]: tasks_items[
                    task_index][1] for task_index in tasks_indices}
                GameRunner.agents[agent_color] = Crewmate(
                    agent_color, CREWMATE, CREWMATE_VISION_RANGE,
                    agent_tasks, decision_maker)
                GameRunner.crewmates.append(GameRunner.agents[agent_color])
                GameRunner.living_crewmates_counter += 1
            elif agent_color in list(impostors_trees):
                decision_maker = "dec_tree_dm"
                GameRunner.agents[agent_color] = Impostor(
                    agent_color, IMPOSTOR,
                    IMPOSTOR_VISION_RANGE,
                    IMPOSTOR_KILL_COOLDOWN, decision_maker)
                GameRunner.impostors.append(GameRunner.agents[agent_color])
                GameRunner.living_impostors_counter += 1
            elif agent_color in list(crewmates_trees):
                decision_maker = "dec_tree_dm"
                tasks_indices = np.random.choice(len(GameRunner.file_data[
                                                         "tasks"]),
                                                 GameRunner.tasks_per_crewmate,
                                                 replace=False)
                agent_tasks = {tasks_items[task_index][0]: tasks_items[
                    task_index][1] for task_index in tasks_indices}
                GameRunner.agents[agent_color] = Crewmate(
                    agent_color, CREWMATE, CREWMATE_VISION_RANGE,
                    agent_tasks, decision_maker)
                GameRunner.crewmates.append(GameRunner.agents[agent_color])
                GameRunner.living_crewmates_counter += 1
            elif agent_color in list(impostors_randoms):
                decision_maker = "random"
                GameRunner.agents[agent_color] = Impostor(
                    agent_color, IMPOSTOR,
                    IMPOSTOR_VISION_RANGE,
                    IMPOSTOR_KILL_COOLDOWN, decision_maker)
                GameRunner.impostors.append(GameRunner.agents[agent_color])
                GameRunner.living_impostors_counter += 1
            elif agent_color in list(crewmates_randoms):
                decision_maker = "random"
                tasks_indices = np.random.choice(len(GameRunner.file_data[
                                                         "tasks"]),
                                                 GameRunner.tasks_per_crewmate,
                                                 replace=False)
                agent_tasks = {tasks_items[task_index][0]: tasks_items[
                    task_index][1] for task_index in tasks_indices}
                GameRunner.agents[agent_color] = Crewmate(
                    agent_color, CREWMATE, CREWMATE_VISION_RANGE,
                    agent_tasks, decision_maker)
                GameRunner.crewmates.append(GameRunner.agents[agent_color])
                GameRunner.living_crewmates_counter += 1
        GameRunner.all_agents = {key: val for key, val in
                                 GameRunner.agents.items()}
        return list(GameRunner.agents.values())

    @staticmethod
    def export_observation(agent, cur_action):
        agent_observation = GameRunner.board.get_observation(agent)
        agent_observation['current action'] = cur_action
        if agent.agent_type:
            for key, val in agent_observation.as_dict().items():
                GameRunner.imp_dict[key].append(val)
            GameRunner.imp_rows += 1
        else:
            for key, val in agent_observation.as_dict().items():
                GameRunner.cm_dict[key].append(val)
            GameRunner.cm_rows += 1

    @staticmethod
    def insert_final_values(winning):
        max_rounds = len(GameRunner.all_agents) - 2
        colors = {agent.color for agent in GameRunner.all_agents.values()}
        cm_colors = {agent.color for agent in GameRunner.all_agents.values() if
                     not agent.agent_type}
        imp_colors = {agent.color for agent in GameRunner.all_agents.values()
                      if agent.agent_type}

        # win
        cm_win, imp_win = (winning + 1) / 2, (-winning + 1) / 2
        GameRunner.cm_dict['win'] = [cm_win for _ in range(GameRunner.cm_rows)]
        GameRunner.imp_dict['win'] = [imp_win for _ in
                                      range(GameRunner.imp_rows)]

        for key, val in GameRunner.cm_dict.items():
            if not val:
                new_val = [-1 for _ in range(GameRunner.cm_rows)]
                GameRunner.cm_dict[key] = new_val
        for key, val in GameRunner.imp_dict.items():
            if not val:
                new_val = [-1 for _ in range(GameRunner.imp_rows)]
                GameRunner.imp_dict[key] = new_val
        cm_df = pd.DataFrame(GameRunner.cm_dict)
        imp_df = pd.DataFrame(GameRunner.imp_dict)

        # finished tasks
        cm_df.loc[cm_df['num tasks left'] == 0, 'finished tasks'] = 1
        cm_df.loc[cm_df['num tasks left'] != 0, 'finished tasks'] = 0
        for imp in imp_colors:
            imp_obj = GameRunner.all_agents[imp]
            imp_obs = GameRunner.board.get_observation(imp_obj)

            # killed
            imp_kills = sum([imp_obs[f'killed in round {i}'] for i in
                             range(1, max_rounds + 1) if imp_obs[
                                 f'killed in round {i}'] != -1])
            imp_df.loc[imp_df['color'] == imp, 'killed'] = imp_kills
        dead, alive = set(), colors.copy()
        surv_cms, surv_imps = len(cm_colors), len(imp_colors)
        for i in range(1, max_rounds + 1):
            cur_dead = GameRunner.cm_dict[f'who died in round {i}'][-1]
            if cur_dead != -1:
                dead.update(cur_dead)
                alive.difference_update(cur_dead)
                for c_dead in cur_dead:
                    if c_dead in cm_colors:
                        surv_cms -= 1
                    else:
                        surv_imps -= 1

            # impostor/crewmate ejected in round
            ejected_this_round = GameRunner.cm_dict[f'ejected in round {i}'][
                -1]
            if ejected_this_round != -1:
                if ejected_this_round in cm_colors:
                    cm_df[f'impostor ejected in round {i}'] = 0
                    imp_df[f'crewmate ejected in round {i}'] = 1
                else:
                    cm_df[f'impostor ejected in round {i}'] = 1
                    imp_df[f'crewmate ejected in round {i}'] = 0
            else:
                cm_df[f'impostor ejected in round {i}'] = 0
                imp_df[f'crewmate ejected in round {i}'] = 0

            # surviving crewmates and impostors
            cm_df[f'surviving crewmates in round {i}'] = surv_cms
            imp_df[f'surviving impostors in round {i}'] = surv_imps

            # survived round
            for a in dead:
                a_obj = GameRunner.all_agents[a]
                if a_obj.agent_type:
                    imp_df.loc[imp_df[
                                   'color'] == a_obj.color, f'survived round {i}'] = 0
                else:
                    cm_df.loc[cm_df[
                                  'color'] == a_obj.color, f'survived round {i}'] = 0
            for a in alive:
                a_obj = GameRunner.all_agents[a]
                if a_obj.agent_type:
                    imp_df.loc[imp_df[
                                   'color'] == a_obj.color, f'survived round {i}'] = 1
                else:
                    cm_df.loc[cm_df[
                                  'color'] == a_obj.color, f'survived round {i}'] = 1

            # voted impostor in round
            for c1 in cm_colors:
                voted_imp = 0
                for c2 in imp_colors:
                    if GameRunner.cm_dict[f'{c1} voted {c2} in round {i}'][
                            -1] == 1:
                        voted_imp = 1
                        break
                cm_df.loc[cm_df[
                              'color'] == c1, f'voted impostor in round {i}'] = voted_imp

            # votes received in round
            for c1 in colors:
                votes_received = 0
                for c2 in colors:
                    if GameRunner.cm_dict[f'{c2} voted {c1} in round {i}'][
                        -1] == 1 or \
                            GameRunner.imp_dict[
                                f'{c2} voted {c1} in round {i}'][-1] == 1:
                        votes_received += 1
                if c1 in cm_colors:
                    cm_df.loc[cm_df[
                                  'color'] == c1, f'votes received in round {i}'] = votes_received
                else:
                    imp_df.loc[imp_df[
                                   'color'] == c1, f'votes received in round {i}'] = votes_received

        return cm_df, imp_df
