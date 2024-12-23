from map.tile import Tile
from config.constants import *
from agents.crewmate_observations import CrewmateObservations
from agents.impostor_observations import ImpostorObservations

dirs = {'move up': (-1, 0), 'move down': (1, 0), 'move left': (0, -1),
        'move right': (0, 1), 'move none': (0, 0)}


def __is_blocked_range_1(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dy) <= abs(dx) and dx <= 0


def __is_blocked_range_2(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dy) <= abs(dx) and dx <= 0 and dy <= 0


def __is_blocked_range_3(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return dy <= 0 and dx <= 0


def __is_blocked_range_4(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dx) <= abs(dy) and dx <= 0 and dy <= 0


def __is_blocked_range_5(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dx) <= abs(dy) and dy <= 0


def __is_blocked_range_6(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dx) <= abs(dy) and dx >= 0 and dy <= 0


def __is_blocked_range_7(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return dx >= 0 and dy <= 0


def __is_blocked_range_8(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dy) <= abs(dx) and dx >= 0 and dy <= 0


def __is_blocked_range_9(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dy) <= abs(dx) and dx >= 0


def __is_blocked_range_10(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dy) <= abs(dx) and dx >= 0 and dy >= 0


def __is_blocked_range_11(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return dx >= 0 and dy >= 0


def __is_blocked_range_12(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dx) <= abs(dy) and dx >= 0 and dy >= 0


def __is_blocked_range_13(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dx) <= abs(dy) and dy >= 0


def __is_blocked_range_14(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dx) <= abs(dy) and dx <= 0 and dy >= 0


def __is_blocked_range_15(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return dx <= 0 and dy >= 0


def __is_blocked_range_16(w_loc, t_loc):
    dy, dx = t_loc[0] - w_loc[0], t_loc[1] - w_loc[1]
    return abs(dy) <= abs(dx) and dx <= 0 and dy >= 0


slice_block_func = {1: __is_blocked_range_1,
                    2: __is_blocked_range_2,
                    3: __is_blocked_range_3,
                    4: __is_blocked_range_4,
                    5: __is_blocked_range_5,
                    6: __is_blocked_range_6,
                    7: __is_blocked_range_7,
                    8: __is_blocked_range_8,
                    9: __is_blocked_range_9,
                    10: __is_blocked_range_10,
                    11: __is_blocked_range_11,
                    12: __is_blocked_range_12,
                    13: __is_blocked_range_13,
                    14: __is_blocked_range_14,
                    15: __is_blocked_range_15,
                    16: __is_blocked_range_16}

wall_slice_mapping = {1: {16, 1, 2}, 2: {2}, 3: {2, 3, 4}, 4: {4},
                      5: {4, 5, 6}, 6: {6}, 7: {6, 7, 8}, 8: {8},
                      9: {8, 9, 10}, 10: {10}, 11: {10, 11, 12}, 12: {12},
                      13: {12, 13, 14}, 14: {14}, 15: {14, 15, 16}, 16: {16}}

block_slice_mapping = {0: set(), 1: {1}, 2: {1, 2, 3}, 3: {3}, 4: {3, 4, 5},
                       5: {5}, 6: {5, 6, 7}, 7: {7}, 8: {7, 8, 9}, 9: {9},
                       10: {9, 10, 11}, 11: {11}, 12: {11, 12, 13},
                       13: {13}, 14: {13, 14, 15}, 15: {15}, 16: {15, 16, 1}}


class Board:

    def __init__(self, length, width, agents, walls, tasks, table,
                 cm_observ_vertices, imp_observ_vertices):
        self.length, self.width = length, width
        self.table = table
        self.walls = walls
        self.agents = set(agents.copy())
        self.board = [[Tile() for _ in range(width)] for __ in range(length)]
        self.__num_of_agents = len(agents)
        self.__add_walls()
        self.__add_tasks(tasks)
        self.__add_agents()
        self.__add_table()
        self.dead_this_round = set()
        self.__table_tiles = {tile for tile in self.__initial_locs_generator()}
        self.__marked_kill_tiles = set()
        self.__agent_observations = {}
        colors = {cur_agent.color for cur_agent in agents}
        for agent in self.agents:
            if agent.agent_type:
                self.__agent_observations[agent] = ImpostorObservations(
                    imp_observ_vertices, agent.color, colors,
                    {a.color for a in agents if a.agent_type})
            else:
                self.__agent_observations[agent] = CrewmateObservations(
                    cm_observ_vertices, agent.color, colors)

    def __add_walls(self):
        for wall in self.walls:
            self.board[wall[0]][wall[1]].is_wall = True

    def __add_tasks(self, tasks):
        for task, rtc in tasks.items():
            self.board[task[0]][task[1]].is_task = True
            self.board[task[0]][task[1]].rtc = rtc
        self.tasks = set(tasks.keys())

    def __add_agents(self):
        gen = self.__initial_locs_generator()
        self.agent_locations = {agent: next(gen) for agent in self.agents}
        for agent, loc in self.agent_locations.items():
            if not agent.is_dead:
                self[loc[0]][loc[1]].agents = {agent}

    def __add_table(self):
        self.board[self.table[0]][self.table[1]].is_table = True
        self.board[self.table[0] + 1][self.table[1]].is_table = True
        self.board[self.table[0]][self.table[1] + 1].is_table = True
        self.board[self.table[0] + 1][self.table[1] + 1].is_table = True

    def get_task_loc(self, agent):
        return self.agent_locations[agent]

    def get_task_rtc(self, agent):
        return self.board[self.agent_locations[
            agent][0]][self.agent_locations[agent][1]].rtc

    def move_agent(self, agent, dirc):
        changed_coords = {self.agent_locations[agent]}
        self[self.agent_locations[agent][0]][
            self.agent_locations[agent][1]].agents.remove(agent)
        self.agent_locations[agent] = (self.agent_locations[agent][0] +
                                       dirs[dirc][0],
                                       self.agent_locations[agent][1] +
                                       dirs[dirc][1])
        self[self.agent_locations[agent][0]][
            self.agent_locations[agent][1]].agents.add(agent)
        changed_coords.add(self.agent_locations[agent])
        return changed_coords

    def kill_agent(self, victim, killer=None):
        self.dead_this_round.add(victim)
        for agent in self.agent_locations:
            if agent.color == victim:
                victim = agent
                break
        v_tile = self.agent_locations[victim]
        changed_coords = {v_tile}
        victim_tile = self[v_tile[0]][v_tile[1]]
        victim_tile.bodies.add(victim)
        victim_tile.agents.remove(victim)
        if killer:
            killer_obs = self.get_observation(killer)
            cur_round = killer_obs['round']
            killer_obs.num_killed_in_round[cur_round - 1] += 1
            killer_obs[f'killed in round {cur_round}'] = \
                killer_obs.num_killed_in_round[cur_round - 1]
            victim_tile.killed_by.add(killer)
            self.__marked_kill_tiles.add(v_tile)
            k_tile = self.agent_locations[killer]
            killer_tile = self[k_tile[0]][k_tile[1]]
            killer_tile.agents.remove(killer)
            victim_tile.agents.add(killer)
            self.agent_locations[killer] = v_tile
            changed_coords.add(k_tile)
        return changed_coords

    def get_pos_plays(self, agent):
        pos_plays = set()
        coords = self.agent_locations[agent]
        agent_tile = self.board[coords[0]][coords[1]]
        if agent_tile.is_task and agent.agent_type == CREWMATE and \
                coords in agent.tasks:
            pos_plays.add("do_task")
            if agent.is_performing_task:
                return pos_plays
        for dirc, vals in dirs.items():
            next_coords = (coords[0] + vals[0], coords[1] + vals[1])
            if next_coords not in self:
                continue
            next_tile = self.board[next_coords[0]][next_coords[1]]
            if next_tile.is_wall or next_tile.is_table:
                continue
            pos_plays.add(dirc)
        for (i, j) in self.__agent_vision(agent):
            tile = self[i][j]
            for body in tile.bodies:
                pos_plays.add(f"report {body.color}")
        if coords in self.__table_tiles and agent.table_calls_left > 0 and \
                agent.table_cd == 0:
            pos_plays.add("call_meeting")
        if agent.agent_type == 1 and agent.kill_cd == 0:
            for tile in self.__agent_kill_range(agent):
                for other_agent in tile.agents:
                    if other_agent.agent_type != 1 and not other_agent.is_dead:
                        pos_plays.add(f"kill {other_agent.color}")
        return pos_plays

    def board_update_crewmate_observation(self, agent, total_cycles, cur_round,
                                          cycles_in_round):
        observation = self.__agent_observations[agent]

        observation.update({'color': agent.color, 'time': total_cycles,
                            'time in round': cycles_in_round,
                            'loc': self.agent_locations[agent],
                            'round': cur_round, 'tasks left': agent.tasks,
                            'num tasks left': len(agent.tasks),
                            'table calls left': agent.table_calls_left,
                            'table cooldown': agent.table_cd})

        agents_num = 0
        seen_bodies, body_locs, body_obsrv_times = [], [], []  # New bodies
        for (i, j) in self.__agent_vision(agent):
            tile = self[i][j]
            for other_agent in tile.agents:
                if other_agent == agent:
                    continue
                agents_num += 1
                observation[f"last loc seen {other_agent.color}"] = (i, j)
                observation[f"last time seen {other_agent.color}"] = \
                    total_cycles
                if tile.is_task:
                    if observation[f"tasks seen on {other_agent.color}"] == -1:
                        observation[f"tasks seen on {other_agent.color}"] = set()
                    else:
                        observation[f"tasks seen on {other_agent.color}"].add(
                            (i, j))
                    observation[f"tasks num seen on {other_agent.color}"] = \
                        len(observation[f"tasks seen on {other_agent.color}"])
                observation.time_seen_agent_in_round[cur_round - 1][
                    other_agent.color] += 1
                observation[
                    f"percentage of round {cur_round} {other_agent.color} was seen"] = \
                    (observation.time_seen_agent_in_round[cur_round - 1][
                         other_agent.color] * 100) // cycles_in_round
            for body in tile.bodies:
                if body.color not in observation.bodies_seen_in_round[cur_round - 1]:
                    observation.bodies_seen_in_round[cur_round - 1].add(body.color)
                    observation[f"is dead {body.color}"] = 1
                    seen_bodies.append(body.color)
                    body_locs.append((i, j))
                    body_obsrv_times.append(total_cycles)
            for killer in tile.killed_by:
                observation[f"seen kill {killer.color}"] = 1
                observation[f"seen {killer.color} kill in round {cur_round}"] = 1
        for i in range(1, len(body_obsrv_times) + 1):
            j = i + len(observation.bodies_seen_in_round[cur_round - 1])
            observation[f'body {j} seen in round {cur_round}'] = seen_bodies[
                i - 1]
            observation[f'body {j} loc in round {cur_round}'] = body_locs[
                i - 1]
            observation[f'body {j} time observed in round {cur_round}'] = \
                body_obsrv_times[i - 1]
        observation['num of agents in vision'] = agents_num

    def board_update_impostor_observation(self, agent, total_cycles, cur_round,
                                          cycles_in_round):
        observation = self.__agent_observations[agent]

        observation.update({'color': agent.color, 'time': total_cycles,
                            'time in round': cycles_in_round,
                            'loc': self.agent_locations[agent],
                            'round': cur_round,
                            'table calls left': agent.table_calls_left,
                            'table cooldown': agent.table_cd})

        agents_num = 0
        seen_bodies, body_locs, body_obsrv_times = [], [], []  # New bodies
        agent_tile = self[self.agent_locations[agent][0]][
            self.agent_locations[agent][1]]
        if agent_tile.is_task:
            observation.tasks_faked.add(self.agent_locations[agent])
            observation.time_faked_tasks += 1
            observation['tasks faked'] = observation.tasks_faked
            observation['time faking tasks'] = observation.time_faked_tasks
        for (i, j) in self.__agent_vision(agent):
            tile = self[i][j]
            for other_agent in tile.agents:
                if other_agent == agent:
                    continue
                agents_num += 1
                observation[f"last loc seen {other_agent.color}"] = (i, j)
                observation[f"last time seen {other_agent.color}"] = \
                    total_cycles
                if tile.is_task:
                    if observation[f"tasks seen on {other_agent.color}"] == -1:
                        observation[f"tasks seen on {other_agent.color}"] = set()
                    else:
                        observation[f"tasks seen on {other_agent.color}"].add(
                            (i, j))
                    observation[f"tasks num seen on {other_agent.color}"] = \
                        len(observation[f"tasks seen on {other_agent.color}"])
                observation.time_seen_agent_in_round[cur_round - 1][
                    other_agent.color] += 1
                observation[
                    f"percentage of round {cur_round} {other_agent.color} was seen"] = \
                    (observation.time_seen_agent_in_round[cur_round - 1][
                         other_agent.color] * 100) // cycles_in_round
                if agent_tile.killed_by and other_agent.agent_type == 0 and \
                        abs(i - self.agent_locations[other_agent][0]) <= \
                        other_agent.vision_range and abs(
                        j - self.agent_locations[other_agent][1]) <= \
                        other_agent.vision_range:
                    observation[f'was seen killing by {other_agent.color}'] = 1
            for body in tile.bodies:
                if body.color not in observation.bodies_seen_in_round[cur_round - 1]:
                    observation.bodies_seen_in_round[cur_round - 1].add(body.color)
                    observation[f"is dead {body.color}"] = 1
                    seen_bodies.append(body.color)
                    body_locs.append((i, j))
                    body_obsrv_times.append(total_cycles)
            for killer in tile.killed_by:
                observation[f"seen kill {killer.color}"] = 1
                observation[f"seen {killer.color} kill in round {cur_round}"] = 1
        for i in range(1, len(body_obsrv_times) + 1):
            j = i + len(observation.bodies_seen_in_round[cur_round - 1])
            observation[f'body {j} seen in round {cur_round}'] = seen_bodies[
                i - 1]
            observation[f'body {j} loc in round {cur_round}'] = body_locs[
                i - 1]
            observation[f'body {j} time observed in round {cur_round}'] = \
                body_obsrv_times[i - 1]
        observation['num of agents in vision'] = agents_num

    def meeting_update_observations(self, cur_round, reporter=-1,
                                    body_reported=-1, table_user=-1,
                                    caller_loc=-1, caller_killer=-1):
        for agent in self.agents:
            if not agent.is_dead:
                obsrv = self.__agent_observations[agent]
                obsrv[f'who died in round {cur_round}'] = self.dead_this_round.copy()
                for c in obsrv[f'who died in round {cur_round}']:
                    obsrv[f'is dead {c}'] = 1
                if agent.agent_type == CREWMATE:
                    for a in self.agents:
                        obsrv[f'seen {a.color} kill in round {cur_round}'] = \
                            max(0, obsrv[f'seen {a.color} kill in round {cur_round}'])
                obsrv[f'body reported by in round {cur_round}'] = reporter
                obsrv[f'body reported in round {cur_round}'] = body_reported
                obsrv[f'table used by in round {cur_round}'] = table_user
                obsrv[
                    f'body loc according to reporter in round {cur_round}'] = caller_loc
                obsrv[
                    f'killer according to reporter in round {cur_round}'] = caller_killer
                if agent.agent_type:
                    obsrv[f'killed in round {cur_round}'] = max(
                        0, obsrv[f'killed in round {cur_round}'])

    def round_end_update_observations(self, cur_round, votes, ejected):
        """
        UPDATES THE AGENT'S PGM WITH INFO ABOUT THE ROUND (SET VALS TO FALSE WHEN NEEDED)
        """
        self.dead_this_round.add(ejected)
        for agent in self.agents:
            if not agent.is_dead:
                obsrv = self.__agent_observations[agent]
                obsrv[f'who died in round {cur_round}'] = self.dead_this_round
                obsrv[f'ejected in round {cur_round}'] = ejected
                for a1 in self.agent_locations:
                    for a2 in self.agent_locations:
                        c1, c2 = a1.color, a2.color
                        if (c1, c2) in votes:
                            obsrv[f'{c1} voted {c2} in round {cur_round}'] = 1
                        else:
                            obsrv[f'{c1} voted {c2} in round {cur_round}'] = 0
        board_after = self.reset_round()
        return board_after

    def set_observation(self, agent, key, value):
        self.__agent_observations[agent][key] = value

    def set_observations(self, agent, observations):
        self.__agent_observations[agent].update(observations)

    def get_observation(self, agent):
        return self.__agent_observations[agent]

    def get_possible_votes(self):
        votes = set()
        for agent in self.agents:
            if not agent.is_dead:
                votes.add(f'vote {agent.color}')
        return votes

    def get_possible_killers(self, agent):
        killers = set()
        for other_agent in self.agents.difference({agent}):
            if not other_agent.is_dead:
                killers.add(f"{other_agent.color} declared")
        killers.add("none declared")
        return killers

    @staticmethod
    def get_possible_body_locs():
        return str_free_coords

    def reset_round(self):
        res = set()
        for cur_agent, loc in self.agent_locations.items():
            if cur_agent.agent_type == CREWMATE:
                cur_agent.cur_task_time = 0
                cur_agent.is_performing_task = False
            self.board[loc[0]][loc[1]].bodies = set()
            self.board[loc[0]][loc[1]].agents = set()
            res.add(loc)
        self.__add_agents()
        res.update(self.__initial_locs_generator())
        self.dead_this_round = set()
        return res

    def __agent_vision(self, agent):
        coords = self.agent_locations[agent]
        walls_in_slices = {}
        for i in range(1, 17):
            walls_in_slices[i] = set()
        for i in range(coords[0] - agent.vision_range,
                       coords[0] + agent.vision_range + 1):
            for j in range(coords[1] - agent.vision_range,
                           coords[1] + agent.vision_range + 1):
                if (i, j) in self.walls:
                    wall_slice = self.__get_slice(i - coords[0],
                                                  j - coords[1])
                    for slc in wall_slice_mapping[wall_slice]:
                        walls_in_slices[slc].add((i, j))
        for i in range(coords[0] - agent.vision_range,
                       coords[0] + agent.vision_range + 1):
            for j in range(coords[1] - agent.vision_range,
                           coords[1] + agent.vision_range + 1):
                if (i, j) not in self.walls and (i, j) in self:
                    blocked = False
                    block_slice = self.__get_slice(i - coords[0],
                                                   j - coords[1])
                    for slc in block_slice_mapping[block_slice]:
                        for wall in walls_in_slices[slc]:
                            if slice_block_func[slc](wall, (i, j)):
                                blocked = True
                                break
                        if blocked:
                            break
                    yield i, j

    def remove_kill_marks(self):
        for (i, j) in self.__marked_kill_tiles:
            self[i][j].killed_by = set()
        self.__marked_kill_tiles = set()

    @staticmethod
    def __get_slice(b, a):
        if a == 0:
            if b == 0:
                return 0
            return 13 if b > 0 else 5
        if b == 0:
            return 9 if a > 0 else 1
        abs_a, abs_b = abs(a), abs(b)
        if abs_a == abs_b:
            if a == b:
                return 11 if a > 0 else 3
            return 15 if a > 0 else 7
        elif abs_a > abs_b:
            if b < 0:
                return 8 if a > b else 2
            return 10 if a > b else 16
        else:
            if a < 0:
                return 14 if b > a else 4
            return 12 if b > a else 6

    def __agent_kill_range(self, agent):
        coords = self.agent_locations[agent]
        for i in range(coords[0] - 1, coords[0] + 2):
            for j in range(coords[1] - 1, coords[1] + 2):
                if (i, j) in self:
                    yield self.board[i][j]

    def __initial_locs_generator(self):
        yield self.table[0] - 1, self.table[1] - 1
        yield self.table[0] - 1, self.table[1]
        yield self.table[0] - 1, self.table[1] + 1
        yield self.table[0] - 1, self.table[1] + 2
        yield self.table[0], self.table[1] - 1
        yield self.table[0], self.table[1] + 2
        yield self.table[0] + 1, self.table[1] - 1
        yield self.table[0] + 1, self.table[1] + 2
        yield self.table[0] + 2, self.table[1] - 1
        yield self.table[0] + 2, self.table[1]
        yield self.table[0] + 2, self.table[1] + 1
        yield self.table[0] + 2, self.table[1] + 2

    def get_table_coords(self):
        return {self.table, (self.table[0] + 1, self.table[1]),
                (self.table[0], self.table[1] + 1),
                (self.table[0] + 1, self.table[1] + 1)}

    def __contains__(self, coords):
        return self.length > coords[0] >= 0 and self.width > coords[1] >= 0

    def __getitem__(self, item):
        return self.board[item]

    def __str__(self):
        ret_str = ""
        return ret_str
