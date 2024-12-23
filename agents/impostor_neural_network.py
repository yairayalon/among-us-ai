from sklearn.neural_network import MLPRegressor
import numpy as np
from scipy.special import softmax
from config.constants import *
from core.game_runner_helper import GameRunnerHelper
import os
import pickle

pred_columns = ['win', 'killed']
for i in range(1, len(COLORS) - 1):
    pred_columns.append(f'crewmate ejected in round {i}')
    pred_columns.append(f'surviving impostors in round {i}')
    pred_columns.append(f'survived round {i}')
    pred_columns.append(f'votes received in round {i}')
pred_columns = tuple(pred_columns)

actions = [str(x) for x in free_coords]
actions.extend(["move none", "move left", "move right",
                "move up", "move down", "call_meeting"])
actions.extend([f"vote {x}" for x in COLORS])
actions.extend([f"report {x}" for x in COLORS])
actions.extend([f"kill {x}" for x in COLORS])
actions.extend([f"{x} declared" for x in COLORS])
actions.append("none declared")

dicter = {}
for ind, col in enumerate(COLORS):
    dicter[col] = ind

action_mapping = {actions[i]: i for i in range(len(actions))}


class ImpostorNeuralNetwork:
    __instance = None

    def __init__(self, train):
        if ImpostorNeuralNetwork.__instance:
            raise Exception

        self.action_replace = {"current action": action_mapping}
        self.replacer = {"color": dicter}

        if train == 0:
            self.__train_model()
            self.__to_pickle()
        elif train == 1:
            self.__from_pickle()
        else:
            self.__from_pickle()
            self.train_update()
            self.__to_pickle()
        __instance = self

    @staticmethod
    def get_instance():
        return ImpostorNeuralNetwork.__instance

    def __from_pickle(self):
        infile = open("training/trained_nns/impostor1500", "rb")
        self.regr = pickle.load(infile)
        infile.close()

    def __to_pickle(self):
        out_file = open("training/trained_nns/impostor", 'wb')
        pickle.dump(self.regr, out_file)
        out_file.close()

    def __train_model(self):
        lst = [f"training/impostor_training{os.path.sep}{name}" for name in
               os.listdir('training/impostor_training')]
        df = GameRunnerHelper.read_dataframe(*lst)

        y = df.loc[:, pred_columns]
        to_drop = list(pred_columns)
        to_drop.append('Unnamed: 0')
        df = df.drop(columns=to_drop)
        df['current action'] = df['current action'].astype(str)

        df = df.replace(self.action_replace)
        df = df.replace(self.replacer)
        df = df.apply(self.str_man)
        self.regr = MLPRegressor(
            hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100),
            warm_start=True)
        self.regr = self.regr.fit(df, y)

    def train_update(self):
        lst = [f"training/impostor_training{os.path.sep}{name}" for name in
               os.listdir('training/impostor_training')]
        df = GameRunnerHelper.read_dataframe(*lst)

        y = df.loc[:, pred_columns]
        to_drop = list(pred_columns)
        to_drop.append('Unnamed: 0')
        df = df.drop(columns=to_drop)
        df['current action'] = df['current action'].astype(str)

        df = df.replace(self.action_replace)
        df = df.replace(self.replacer)
        df = df.apply(self.str_man)
        self.regr = self.regr.fit(df, y)
        self.__to_pickle()

    def decide_action(self, observation, legal_actions):
        cur_round = observation['round']
        ordered_dict = observation.as_dict()
        sample = np.array([tuple(ordered_dict.values())], dtype=object)
        sample = np.apply_along_axis(self.column_runner, axis=0, arr=sample.reshape(1, -1))
        sample = sample.reshape(1, -1)
        dict_pos_actions = dict()
        for action in legal_actions:
            sample[-1] = action_mapping[action]
            result_vec = self.regr.predict(sample)
            res = ImpostorNeuralNetwork.__win_weight(
                result_vec[0][0], cur_round) + ImpostorNeuralNetwork.__killed_weight(result_vec[0][1], cur_round)
            res += ImpostorNeuralNetwork.__eject_crew_cur_round_weight(
                result_vec[0][2], cur_round)
            res += ImpostorNeuralNetwork.__surv_imps_cur_round_weight(
                result_vec[0][3], cur_round)
            res += ImpostorNeuralNetwork.__surv_cur_round_weight(
                result_vec[0][4], cur_round)
            res += ImpostorNeuralNetwork.__votes_received_cur_round_weight(
                result_vec[0][5], cur_round)
            cur_idx = 6
            for r in range(cur_round + 1, len(COLORS) - 1):
                res += ImpostorNeuralNetwork.__eject_crew_next_round_weight(
                    result_vec[0][cur_idx], cur_round, r)
                cur_idx += 1
                res += ImpostorNeuralNetwork.__surv_imps_next_round_weight(
                    result_vec[0][cur_idx], cur_round, r)
                cur_idx += 1
                res += ImpostorNeuralNetwork.__surv_next_round_weight(
                    result_vec[0][cur_idx], cur_round, r)
                cur_idx += 1
                res += ImpostorNeuralNetwork.__votes_received_next_round_weight(
                    result_vec[0][cur_idx], cur_round, r)
                cur_idx += 1
            dict_pos_actions[action] = res
        keys = []
        values = []
        for act, val in dict_pos_actions.items():
            keys.append(act)
            values.append(val)
        vals = softmax(values)
        return np.random.choice(keys, replace=False, p=vals)

    @staticmethod
    def converter(tup_str):
        x = tup_str.split(",")
        x1 = int(x[0][1:]), int(x[1][:-1])
        return x1

    @staticmethod
    def set_converter(set_str):
        return len(eval(set_str))

    def str_man(self, entry):
        return self.column_applier(entry)

    @staticmethod
    def column_runner(entry):
        if type(entry) == np.ndarray:
            entry = entry[0]
        if type(entry) == int or type(entry) == float:
            return entry
        if type(entry) == str and entry.isnumeric():
            return float(entry)
        if entry == "-1":
            return -1
        elif entry in COLORS:
            return dicter[entry]
        elif type(entry) == str and entry.startswith("none"):
            return -1
        elif type(entry) == str and entry.startswith("{"):
            val = len(eval(entry))
            return val
        elif type(entry) == str and entry.startswith("("):
            val = ImpostorNeuralNetwork.converter(entry)
            return 25 * val[0] + val[1]
        elif type(entry) == str and entry.endswith("declared"):
            return dicter[entry.split()[0]]
        elif type(entry) == set:
            return len(entry)
        elif type(entry) == tuple:
            return 25 * entry[0] + entry[1]
        elif entry == 'set()':
            return 0
        elif type(entry) == np.str_ or type(entry) == str:
            return action_mapping[entry]
        else:
            return entry

    def column_applier(self, entry):
        return entry.apply(self.column_runner)

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
