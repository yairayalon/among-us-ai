import numpy as np
from scipy.spatial import distance

from config.constants import *
from core.vote_flow import seen_body
from core.game_flow import GameFlow


class ImpostorGameFlow(GameFlow):
    __instance = None

    @staticmethod
    def get_instance():
        if ImpostorGameFlow.__instance is None:
            raise Exception
        return ImpostorGameFlow.__instance

    def __init__(self):
        if ImpostorGameFlow.__instance:
            raise Exception
        super().__init__()
        ImpostorGameFlow.__instance = self

    def is_during_kill_cd(self, observ):
        if observ["kill cooldown"] > 0:
            return True
        return False

    def is_any_agent_around(self, observ, agents):
        for agent_color in agents:
            # other agent is in vision range
            if observ[f"last time seen {agent_color}"] >= \
                    observ["time"]:
                return True
        return False

    def get_crewmates_and_amounts_around(self, observ, agents):
        crewmates_around = []
        num_crewmates_around, num_impostors_around = 0, 1
        for agent_color in agents:
            # other agent is in vision range and in kill range
            if observ[f"last time seen {agent_color}"] >= \
                    observ["time"] and distance.chebyshev(
                list(observ["loc"][::-1]),
                list(observ[
                         f"last loc seen {agent_color}"][::-1])) <= \
                    IMPOSTOR_KILL_RANGE:
                if agents[agent_color].agent_type == CREWMATE:
                    crewmates_around.append(agent_color)
                    num_crewmates_around += 1
                else:
                    num_impostors_around += 1
        return crewmates_around, num_crewmates_around, num_impostors_around

    # the main function - returns the chosen action from ImpostorGameFlow
    def get_chosen_act(self, observ, legal_moves, agents, crewmates, impostor):
        self.set_task_goal_and_dir(observ, impostor)
        self.set_checkpoint_goal_and_dir(observ, impostor)

        if self.is_during_first_kill_cd(observ):
            return self.get_chosen_act_accord_probs(legal_moves,
                                                    {self.dir_to_task: 1})

        # seen body this round
        if seen_body(observ, observ["round"],
                     len(crewmates) - 2):
            body_color_seen = self.get_last_body_color_seen(legal_moves)
            if self.is_any_agent_around(observ, agents):
                return self.get_chosen_act_accord_probs(legal_moves,
                                                        {f"{body_color_seen}": 0.95,
                                                         self.dir_to_task: 0.05})
            return self.get_chosen_act_accord_probs(legal_moves,
                                                    {f"report {body_color_seen}": 0.2,
                                                     self.dir_to_task: 0.8})

        crewmates_around, num_crewmates_around, num_impostors_around = \
            self.get_crewmates_and_amounts_around(observ, agents)
        if (num_crewmates_around == 1 and num_impostors_around == 1) or (
                num_crewmates_around == 1 and num_impostors_around == 2) or \
                (num_crewmates_around == 2 and num_impostors_around == 2):

            crewmate_to_kill = np.random.choice(crewmates_around)
            if num_crewmates_around == 1:
                if num_impostors_around == 1:
                    return self.get_chosen_act_accord_probs(legal_moves, {
                        f"kill {crewmate_to_kill}": 0.7,
                        self.dir_to_task: 0.3})
                elif num_impostors_around >= 2:
                    return self.get_chosen_act_accord_probs(legal_moves, {
                        f"kill {crewmate_to_kill}": 0.85,
                        self.dir_to_task: 0.15})
            elif num_crewmates_around == 2 and num_impostors_around == 2:
                return self.get_chosen_act_accord_probs(legal_moves, {
                    f"kill {crewmate_to_kill}": 0.55,
                    self.dir_to_task: 0.45})

        if self.is_during_kill_cd(observ):
            if self.dir_to_task == self.dir_to_checkpoint:
                return self.get_chosen_act_accord_probs(legal_moves,
                                                        {self.dir_to_task: 1})
            return self.get_chosen_act_accord_probs(legal_moves, {
                self.dir_to_task: 0.85,
                self.dir_to_checkpoint: 0.15})
        if self.dir_to_task == self.dir_to_checkpoint:
            return self.get_chosen_act_accord_probs(legal_moves,
                                                    {self.dir_to_task: 1})
        return self.get_chosen_act_accord_probs(legal_moves, {
            self.dir_to_task: 0.15,
            self.dir_to_checkpoint: 0.85})
