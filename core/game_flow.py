from abc import abstractmethod

import numpy as np

from config.constants import *
from core.vote_flow import normalize_probs
from core.game_runner_helper import GameRunnerHelper as Grh


class GameFlow:
    def __init__(self):
        self.dir_to_task = None
        self.dir_to_checkpoint = None

    @abstractmethod
    def get_chosen_act(self, observ, legal_moves, agents, crewmates, agent):
        pass

    def get_chosen_act_accord_probs(self, legal_moves, pos_acts_with_probs):
        valid_pos_acts_with_probs = {pos_act: pos_acts_with_probs[pos_act]
                                     for pos_act in pos_acts_with_probs if
                                     Grh.is_move_valid(pos_act, legal_moves)}
        if len(pos_acts_with_probs) != len(valid_pos_acts_with_probs):
            valid_pos_acts_with_probs = normalize_probs(
                valid_pos_acts_with_probs)
        return np.random.choice(list(valid_pos_acts_with_probs.keys()),
                                p=list(valid_pos_acts_with_probs.values()))

    def is_during_first_kill_cd(self, observ):
        if observ["time in round"] <= IMPOSTOR_KILL_COOLDOWN:
            return True
        return False

    def get_last_body_color_seen(self, legal_moves):
        body_color_seen = None
        bodies_cols_to_report = [move for move in legal_moves if
                                 move.startswith("report")]
        if bodies_cols_to_report:
            body_color_seen = bodies_cols_to_report[0]
        return body_color_seen

    def set_task_goal_and_dir(self, observ, agent):
        if not agent.task_goal_coord or (agent.task_goal_coord and
                                         agent.task_goal_coord ==
                                         observ["loc"][::-1]):
            if agent.agent_type == CREWMATE:
                crewmate_tasks = {task[::-1] for task in agent.tasks}
                closest_task_coord = Grh.get_closest_task_coord(
                    observ["loc"][::-1], crewmate_tasks)
            else:
                closest_task_coord = Grh.get_closest_task_coord(
                    observ["loc"][::-1])
            self.dir_to_task = Grh.get_dir_to_task(observ["loc"][::-1],
                                                   closest_task_coord)
            agent.task_goal_coord = closest_task_coord
        elif agent.task_goal_coord != observ["loc"][::-1]:
            self.dir_to_task = Grh.get_dir_to_task(observ["loc"][::-1],
                                                   agent.task_goal_coord)

    def set_checkpoint_goal_and_dir(self, observ, agent):
        if not agent.checkpoint_goal_coord or (agent.checkpoint_goal_coord and
                                               agent.checkpoint_goal_coord ==
                                               observ["loc"][::-1]):
            closest_checkpoint_coord = Grh.get_random_checkpoint_coord(
                observ["loc"][::-1])
            self.dir_to_checkpoint = Grh.get_dir_to_task(
                observ["loc"][::-1], closest_checkpoint_coord)
            agent.checkpoint_goal_coord = closest_checkpoint_coord
        elif agent.checkpoint_goal_coord != observ["loc"][::-1]:
            self.dir_to_checkpoint = Grh.get_dir_to_task(observ["loc"][::-1],
                                                         agent.checkpoint_goal_coord)
