import pandas as pd
from collections import OrderedDict


class CrewmateObservations:

    def __init__(self,  observ_vertices, my_color, colors):
        max_rounds = len(colors) - 2
        self.__obs = OrderedDict()
        for key in observ_vertices:
            self.__obs[key] = -1
        self.bodies_seen_in_round = []
        self.time_seen_agent_in_round = [{} for _ in range(max_rounds)]
        for round_num in range(1, max_rounds + 1):
            self.bodies_seen_in_round.append(set())
            for color in colors.difference({my_color}):
                self.time_seen_agent_in_round[round_num - 1][color] = 0
                self.__obs[f'percentage of round {round_num} {color} was seen'] = 0
        for c in colors:
            self.__obs[f'tasks num seen on {c}'] = 0
            self.__obs[f'tasks seen on {c}'] = set()
            self.__obs[f'is dead {c}'] = 0
            self.__obs[f'seen kill {c}'] = 0
        self.__obs['current action'] = -1

    def __getitem__(self, item):
        return self.__obs[item]

    def __setitem__(self, key, value):
        self.__obs[key] = value

    def update(self, other_dict):
        self.__obs.update(other_dict)

    def as_dict(self):
        return self.__obs

    def as_data_frame(self):
        obs_list = {key: [self.__obs[key]] for key in self.__obs}
        return pd.DataFrame(obs_list)
