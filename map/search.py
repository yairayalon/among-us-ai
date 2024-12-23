# This class is being used for A* searching the board directions
from config import constants
from core.game_runner import GameRunner


class Node:
    def __init__(self, position, parent):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


def a_star_search(start, end):
    opened = []
    closed = []
    start_node = Node(start, None)
    goal_node = Node(end, None)
    opened.append(start_node)

    while len(opened) > 0:
        opened.sort()
        current_node = opened.pop(0)
        closed.append(current_node)

        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        (x, y) = current_node.position
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for neighbor in neighbors:
            if neighbor in constants.wall_coords or neighbor in constants.table_coords:
                continue
            neighbor = Node(neighbor, current_node)
            if neighbor in closed:
                continue
            neighbor.g = abs(
                neighbor.position[0] - start_node.position[0]) + abs(
                neighbor.position[1] - start_node.position[1])
            neighbor.h = abs(
                neighbor.position[0] - goal_node.position[0]) + abs(
                neighbor.position[1] - goal_node.position[1])
            neighbor.f = neighbor.g + neighbor.h
            if add_to_open(opened, neighbor):
                opened.append(neighbor)
    return None


def add_to_open(opened, neighbor):
    for node in opened:
        if neighbor == node and neighbor.f >= node.f:
            return False
    return True


def get_directs_map():
    all_goals_coords = constants.all_tasks_coords.union(set(
        constants.checkpoints_coords))
    first_coords_to_goal = {(valid_coord, goal_coord): a_star_search(
        valid_coord, goal_coord)[0] for goal_coord in
                            all_goals_coords for valid_coord in
                            constants.valid_start_coords.difference({
                                goal_coord})}
    difference_coords = {
        coords: (first_coords_to_goal[coords][0]
                 - coords[0][0],
                 first_coords_to_goal[coords][1]
                 - coords[0][1]) for coords in
        first_coords_to_goal}
    coords_to_directs_dict = {(0, 1): 'move down', (0, -1): 'move up',
                              (1, 0): 'move right', (-1, 0): 'move left',
                              (0, 0): 'move none'}
    first_directs_to_tasks = {coords: coords_to_directs_dict[difference_coords[
        coords]] for coords in difference_coords}
    return first_directs_to_tasks


if __name__ == "__main__":
    GameRunner.init_game(False)
    all_coords = [(i, j) for i in range(GameRunner.file_data["length"]) for j
                  in range(GameRunner.file_data["width"])]
    directs_map = get_directs_map()
    print(directs_map)
