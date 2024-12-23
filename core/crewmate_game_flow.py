import numpy as np
from scipy.spatial import distance

from config.constants import *
from core.vote_flow import seen_body
from core.game_flow import GameFlow
from core.game_runner_helper import GameRunnerHelper as Grh


class CrewmateGameFlow(GameFlow):
    __instance = None

    @staticmethod
    def get_instance():
        if CrewmateGameFlow.__instance is None:
            raise Exception
        return CrewmateGameFlow.__instance

    def __init__(self, tasks_per_crewmate):
        if CrewmateGameFlow.__instance:
            raise Exception
        super().__init__()
        self.tasks_per_crewmate = tasks_per_crewmate
        CrewmateGameFlow.__instance = self

    def get_profit_act_accord_probs(self, observ, legal_moves):
        if not observ["tasks left"]:
            return self.get_chosen_act_accord_probs(
                legal_moves, {self.dir_to_checkpoint: 1})

        if self.dir_to_task == self.dir_to_checkpoint:
            return self.get_chosen_act_accord_probs(legal_moves,
                                                    {self.dir_to_task: 1})

        profit_move_to_checkpoints = observ["time in round"] + 5 * (
                self.tasks_per_crewmate - len(observ["tasks left"]))
        if profit_move_to_checkpoints > 50:
            better_dir, worse_dir = self.dir_to_checkpoint, self.dir_to_task
        else:
            better_dir, worse_dir = self.dir_to_task, self.dir_to_checkpoint
        return self.get_chosen_act_accord_probs(legal_moves, {better_dir: 0.8,
                                                              worse_dir: 0.2})

    def seen_murder_this_round(self, observ, agents):
        for agent_color in agents:
            if observ[f"seen {agent_color} kill in round {observ['round']}"] \
                    == 1:
                return True
        return False

    def is_near_table(self, observ):
        for table_coord in table_coords:
            if distance.chebyshev(list(observ["loc"][::-1]),
                                  list(table_coord)) <= 1:
                return True
        return False

    # the main function - returns the chosen action from CrewmateGameFlow
    def get_chosen_act(self, observ, legal_moves, agents, crewmates, crewmate):
        if "do_task" in legal_moves:
            return "do_task"
        self.set_task_goal_and_dir(observ, crewmate)
        self.set_checkpoint_goal_and_dir(observ, crewmate)

        if self.seen_murder_this_round(observ, agents):
            body_color_seen = self.get_last_body_color_seen(legal_moves)
            return self.get_chosen_act_accord_probs(legal_moves,
                                                    {f"{body_color_seen}": 1})

        # seen body this round
        if seen_body(observ, observ["round"],
                     len(crewmates) - 2):
            body_color_seen = self.get_last_body_color_seen(legal_moves)
            dir_to_random = np.random.choice(Grh.get_valid_dirs(
                observ["loc"][::-1]))
            return self.get_chosen_act_accord_probs(legal_moves,
                                                    {f"{body_color_seen}": 0.9,
                                                     dir_to_random: 0.1})

        if self.is_during_first_kill_cd(observ):
            # no tasks left for agent
            if not observ["tasks left"]:
                return self.get_chosen_act_accord_probs(
                    legal_moves, {self.dir_to_checkpoint: 1})
            tasks_left_list = list({task[::-1] for task in observ["tasks "
                                                                  "left"]})
            random_task_idx = np.random.choice(len(tasks_left_list))
            random_task_coord = tasks_left_list[random_task_idx]
            return self.get_chosen_act_accord_probs(legal_moves, {
                Grh.get_dir_to_task(observ["loc"][::-1],
                                    random_task_coord): 1})

        profit_act_accord_probs = self.get_profit_act_accord_probs(
            observ, legal_moves)
        if self.is_near_table(observ):
            return self.get_chosen_act_accord_probs(legal_moves,
                                                    {"call_meeting": 0.25,
                                                     profit_act_accord_probs: 0.75})
        else:
            return profit_act_accord_probs
