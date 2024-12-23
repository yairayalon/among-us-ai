from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Deprecated class!!! Added for archiving


class ImpostorPgmDecisionMaker:
    __instance = None

    @staticmethod
    def get_instance():
        if ImpostorPgmDecisionMaker.__instance is None:
            raise Exception
        return ImpostorPgmDecisionMaker.__instance

    def __init__(self, obsrv_vertices, desired_vertices, colors, pos_vals):
        if ImpostorPgmDecisionMaker.__instance:
            raise Exception

        self.__desired_vertices_round = {'survived round': (ImpostorPgmDecisionMaker.__surv_cur_round_weight, ImpostorPgmDecisionMaker.__surv_next_round_weight),
                                         'votes received in round': (ImpostorPgmDecisionMaker.__votes_received_cur_round_weight, ImpostorPgmDecisionMaker.__votes_received_next_round_weight),
                                         'surviving impostors in round': (ImpostorPgmDecisionMaker.__surv_imps_cur_round_weight, ImpostorPgmDecisionMaker.__surv_imps_next_round_weight),
                                         'crewmate ejected in round': (ImpostorPgmDecisionMaker.__eject_crew_cur_round_weight, ImpostorPgmDecisionMaker.__eject_crew_next_round_weight),
                                         'killed in round': (ImpostorPgmDecisionMaker.__num_kills_cur_round_weight, ImpostorPgmDecisionMaker.__num_kills_next_round_weight)}.items()

        self.__max_rounds = len(colors) - 2
        self.__pos_vals = pos_vals
        action_vertice = 'current action'

        dependencies = []
        for v1 in obsrv_vertices:
            for v2 in desired_vertices:
                dependencies.append((v1, v2))
        for v in obsrv_vertices:
            dependencies.append((action_vertice, v))

        self.__observ_vertices, self.__desired_verticed = \
            obsrv_vertices, desired_vertices

        self.pgm = BayesianModel(dependencies)
        self.inference = None

        ImpostorPgmDecisionMaker.__instance = self

    def decide_action(self, observation, actions=()):
        data = observation.as_dict()
        max_val, max_action = float('-inf'), ''
        for action in actions:
            data['current action'] = action
            desired_vals = self.inference.map_query(self.__desired_verticed,
                                                    data, show_progress=False)
            value = self.get_desired_value(desired_vals, observation['round'])
            if value > max_val:
                max_val = value
                max_action = action
        return max_action

    def fit(self, data):
        self.pgm.fit(data, state_names=self.__pos_vals)
        self.inference = VariableElimination(self.pgm)

    def get_desired_value(self, predictions, cur_round):
        IPDM = ImpostorPgmDecisionMaker
        value = (IPDM.__win_weight(predictions['win'], cur_round) +
                 IPDM.__killed_weight(predictions['finished tasks'], cur_round))
        for k, v in self.__desired_vertices_round:
            value += v[0](predictions[f'{k} {cur_round}'], cur_round)
            for i in range(cur_round + 1, self.__max_rounds + 1):
                value += v[1](predictions[f'{k} {i}'], cur_round, i)
        return value

    @staticmethod
    def __win_weight(win_val, cur_round):
        return win_val * 10 * cur_round

    @staticmethod
    def __killed_weight(killed_val, cur_round):
        return killed_val * 3 * cur_round

    @staticmethod
    def __surv_cur_round_weight(surv_val, cur_round):
        return surv_val * (12 + cur_round)

    @staticmethod
    def __surv_next_round_weight(surv_val, cur_round, next_round):
        return surv_val * 5 / (next_round - cur_round)

    @staticmethod
    def __eject_crew_cur_round_weight(eject_val, cur_round):
        return eject_val * (12 + cur_round)

    @staticmethod
    def __eject_crew_next_round_weight(eject_val, cur_round, next_round):
        return eject_val * 5 / (next_round - cur_round)

    @staticmethod
    def __votes_received_cur_round_weight(votes_val, cur_round):
        return -5 * votes_val

    @staticmethod
    def __votes_received_next_round_weight(votes_val, cur_round, next_round):
        return -2 * votes_val / (next_round - cur_round)

    @staticmethod
    def __surv_imps_cur_round_weight(surv_val, cur_round):
        return (8 + cur_round) * surv_val

    @staticmethod
    def __surv_imps_next_round_weight(surv_val, cur_round, next_round):
        return 3 * surv_val / (next_round - cur_round)

    @staticmethod
    def __num_kills_cur_round_weight(kills_val, cur_round):
        return (8 + cur_round) * kills_val

    @staticmethod
    def __num_kills_next_round_weight(kills_val, cur_round, next_round):
        return 3 * kills_val / (next_round - cur_round)
