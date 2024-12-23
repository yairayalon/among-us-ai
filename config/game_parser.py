import json


class GameParser:
    @staticmethod
    def parse_game_settings():
        f = open('config/example.json')
        dict_res = json.load(f)
        walls = {tuple(wall) for wall in dict_res['wall_locs']}
        dict_tasks = {tuple(d[0]): d[1] for d in dict_res['tasks']}
        table = tuple(dict_res['table'][0])
        dict_res['wall_locs'] = walls
        dict_res['tasks'] = dict_tasks
        dict_res['table'] = table
        f.close()
        return dict_res

    @staticmethod
    def parse_tree_json(tree_json_path):
        f = open(tree_json_path)
        dict_res = json.load(f)
        return dict_res
