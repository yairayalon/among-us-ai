from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
import numpy as np

# Deprecated class!!! Added for archiving


class CrewmatePgmDecisionMaker:
    __instance = None

    @staticmethod
    def get_instance():
        if CrewmatePgmDecisionMaker.__instance is None:
            raise Exception
        return CrewmatePgmDecisionMaker.__instance

    def __init__(self, obsrv_vertices, imp_vertices, desired_vertices, colors):
        if CrewmatePgmDecisionMaker.__instance:
            raise Exception

        self.__desired_vertices_round = {'survived round': (CrewmatePgmDecisionMaker.__surv_cur_round_weight, CrewmatePgmDecisionMaker.__surv_next_round_weight),
                                         'voted impostor in round': (CrewmatePgmDecisionMaker.__vote_imp_cur_round_weight, CrewmatePgmDecisionMaker.__vote_imp_next_round_weight),
                                         'votes received in round': (CrewmatePgmDecisionMaker.__votes_received_cur_round_weight, CrewmatePgmDecisionMaker.__votes_received_next_round_weight),
                                         'surviving crewmates in round': (CrewmatePgmDecisionMaker.__surv_cms_cur_round_weight, CrewmatePgmDecisionMaker.__surv_cms_next_round_weight),
                                         'impostor ejected in round': (CrewmatePgmDecisionMaker.__eject_imp_cur_round_weight, CrewmatePgmDecisionMaker.__eject_imp_next_round_weight)}.items()

        self.__max_rounds = len(colors) - 2
        # self.__pos_vals = pos_vals
        action_vertex = 'current action'

        dependencies = []
        for v1 in obsrv_vertices:
            for v2 in desired_vertices:
                dependencies.append((v1, v2))
        for v1 in obsrv_vertices:
            for v2 in imp_vertices:
                dependencies.append((v1, v2))
        for v in desired_vertices:
            dependencies.append((action_vertex, v))
        for v1 in imp_vertices:
            for v2 in desired_vertices:
                dependencies.append((v1, v2))

        self.__observ_vertices, self.__imp_vertices, self.__desired_verticed = \
            obsrv_vertices, imp_vertices, desired_vertices

        self.pgm = BayesianModel(dependencies)
        self.inference = None

        CrewmatePgmDecisionMaker.__instance = self

    def decide_action(self, observation, actions=()):
        data = observation.as_dict()
        actions = np.array(list(actions))
        action_vals = np.array([0 for _ in actions])
        for i, action in enumerate(actions):
            data['current action'] = action
            desired_vals = self.inference.map_query(self.__desired_verticed,
                                                    data, show_progress=False)
            value = self.get_desired_value(desired_vals, observation['round'])
            action_vals[i] = value
        action_vals = np.exp(action_vals)
        action_vals = action_vals / np.sum(action_vals)
        chosen_action = np.random.choice(actions, p=action_vals)
        return chosen_action

    # CHOOSES ACTION BY MAX VALUE
    # def decide_action(self, observation, actions=()):
    #     data = observation.as_dict()
    #     max_val, max_action = float('-inf'), ''
    #     for action in actions:
    #         data['current action'] = action
    #         desired_vals = self.inference.map_query(self.__desired_verticed,
    #                                                 data, show_progress=False)
    #         value = self.get_desired_value(desired_vals, observation['round'])
    #         if value > max_val:
    #             max_val = value
    #             max_action = action
    #     return max_action

    def fit(self, data):
        # self.pgm.fit(data, state_names=self.__pos_vals)
        self.pgm.fit(data)
        self.inference = VariableElimination(self.pgm)

    def get_desired_value(self, predictions, cur_round):
        CPDM = CrewmatePgmDecisionMaker
        value = (CPDM.__win_weight(predictions['win'], cur_round) +
                 CPDM.__tasks_weight(predictions['finished tasks'], cur_round))
        for k, v in self.__desired_vertices_round:
            value += v[0](predictions[f'{k} {cur_round}'], cur_round)
            for i in range(cur_round + 1, self.__max_rounds + 1):
                value += v[1](predictions[f'{k} {i}'], cur_round, i)
        return value

    @staticmethod
    def __win_weight(win_val, cur_round):
        return win_val * 10 * cur_round

    @staticmethod
    def __tasks_weight(tasks_val, cur_round):
        return tasks_val * (cur_round + 18)

    @staticmethod
    def __surv_cur_round_weight(surv_val, cur_round):
        return surv_val * (8 + cur_round)

    @staticmethod
    def __surv_next_round_weight(surv_val, cur_round, next_round):
        return surv_val * 3 / (next_round - cur_round)

    @staticmethod
    def __vote_imp_cur_round_weight(vote_val, cur_round):
        return vote_val * (6 + cur_round)

    @staticmethod
    def __vote_imp_next_round_weight(vote_val, cur_round, next_round):
        return vote_val * 2 / (next_round - cur_round)

    @staticmethod
    def __eject_imp_cur_round_weight(eject_val, cur_round):
        return eject_val * (10 + cur_round)

    @staticmethod
    def __eject_imp_next_round_weight(eject_val, cur_round, next_round):
        return eject_val * 4 / (next_round - cur_round)

    @staticmethod
    def __votes_received_cur_round_weight(votes_val, cur_round):
        return - 2 * (votes_val + cur_round)

    @staticmethod
    def __votes_received_next_round_weight(votes_val, cur_round, next_round):
        return -2 * votes_val / (next_round - cur_round)

    @staticmethod
    def __surv_cms_cur_round_weight(surv_val, cur_round):
        return 2 * (surv_val + cur_round)

    @staticmethod
    def __surv_cms_next_round_weight(surv_val, cur_round, next_round):
        return surv_val * (next_round - cur_round)

