import numpy as np

CONST_WEIGHTS = [0.7, 0.15, 0.10, 0.05]


def normalize_probs(dict_probs):
    keys = []
    values = []
    sum_total = 0
    for d in dict_probs:
        keys.append(d)
        values.append(dict_probs[d])
        sum_total += dict_probs[d]
    dict_res = dict()
    for k in range(len(keys)):
        dict_res[keys[k]] = values[k] / sum_total
    return dict_res


def translate_to_probs(sorted_dict):
    opposite = {sorted_dict[key]: key for key in range(len(sorted_dict))}
    i = 0
    for o in opposite:
        if i == 0:
            opposite[o] = 0.55
        elif i == 1:
            opposite[o] = 0.3
        else:
            opposite[o] = 0.15
        i += 1
    return opposite


def find_body_location(observation, dead_agent, last_round, max_bodies):
    for i in range(1, max_bodies + 1):
        if observation[f"body {i} seen in round {last_round}"] \
                == dead_agent:
            location = observation[f"body {i} loc in round {last_round}"]
            if location and location != -1:
                return location


def find_last_vote_in_round(crewmate_obs, current, last_round, agents):
    if last_round == 1:
        return None
    for agent in agents:
        if agent.color == current.color:
            continue
        if crewmate_obs[f'{current.color} voted {agent.color} in '
                        f'round {last_round}']:
            return agent.color
    return None


def have_seen_kill(observation, living_agents, last_round, max_bodies
                   , dead_agent):
    if not dead_agent:
        return None
    for living_agent in living_agents:
        key = f"seen {living_agent.color} kill in round {last_round}"
        if observation[key] == 1:
            body = find_body_location(observation, dead_agent,
                                      last_round, max_bodies)
            return living_agent, body
    return None


def some_seen_kill(living_agents, last_round, max_bodies, board,
                   dead_agent=None):
    colors = [agent.color for agent in living_agents]
    for agent in living_agents:
        obs = board.get_observation(agent)
        res = have_seen_kill(obs, colors, last_round, max_bodies, dead_agent)
        if res:
            return res
    return None


def seen_body(observation, last_round, max_bodies):
    for i in range(1, max_bodies + 1):
        body = observation[f"body {i} seen in round {last_round}"]
        if body != -1:
            return body
    return None


def has_main_suspect(last_round, living_agents, curr_agent,
                     body_loc, board):
    if body_loc is None:
        return None
    for round in range(last_round - 5, last_round + 1):
        observation = board.get_observation(curr_agent)
        for living in living_agents:
            loc = observation[f'last loc seen {living.color}']
            if loc == -1:
                continue
            curr = abs(loc[0] - body_loc[0]) + abs(loc[1] - body_loc[1])
            if curr <= 5:
                time_seen = observation[f"last time seen {living.color}"]
                if time_seen == round:
                    return living
    return None


def probs_voting_body(observation, colors, last_round, my_color):
    reporter = None
    killer = None
    last_vote = None
    for color in colors:
        if observation[f'body reported by in round {last_round}'] \
                == color:
            reporter = color
            break
    for c in colors:
        if observation[f'killer according to reporter in round {last_round}'] == c:
            if c != my_color:
                killer = c
                break
    for c in colors:
        if observation[f"{my_color} voted {c} in round {last_round}"]:
            last_vote = c
            break
    lst = np.array([killer, last_vote, reporter])
    probs = [0.9, 0.06, 0.04]
    for i in range(len(lst)):
        if not lst[i]:
            probs[i] = 0
    probs = probs / np.sum(probs)
    dict_probs = dict()
    for i in range(len(lst)):
        if not lst[i]:
            continue
        dict_probs[lst[i]] = probs[i]
    return normalize_probs(dict_probs)


def get_suspicion_table(observation, living_agents, my_color,
                        body_cords, num_rounds, last_round):
    percentage_seen = 0
    times_he_voted_for_me = 0
    suspicion_table = dict()

    for living in living_agents:
        loc_seen = observation[f"last loc seen {living.color}"]
        if loc_seen != -1 and body_cords:
            dist = abs(loc_seen[0] - body_cords[0]) + abs(loc_seen[1] -
                                                          body_cords[1])
        else:
            dist = 0
        tasks = observation[f'tasks num seen on {living.color}']
        if tasks == -1:
            num_tasks_performed = 0

        perct_seen = observation[f'percentage of round {last_round} '
                                 f'{living.color} was seen']
        if perct_seen == -1:
            percentage_seen = 0

        for r in range(1, num_rounds + 1):
            if observation[f'{living.color} voted {my_color} in '
                           f'round {r}']:
                times_he_voted_for_me += 1
        summer = dist + 2 * tasks + percentage_seen + times_he_voted_for_me * 0.5 + 34000
        suspicion_table[living.color] = summer
        percentage_seen = 0
        times_he_voted_for_me = 0
    return suspicion_table


def get_crewmate_total_data(crewmate_obs, living_agents, last_round,
                            max_bodies, current_agent, reporter, dead,
                            board, mult=True):
    report_data = dict()
    vote_data = dict()
    killer = have_seen_kill(crewmate_obs, living_agents, last_round, max_bodies
                            , dead)
    if killer:
        vote_data = {killer[0]: 1}
        report_data = [killer[0], killer[1]]
    else:
        for living in living_agents:
            killer = have_seen_kill(board.get_observation(living),
                                    living_agents, last_round, max_bodies,
                                    dead)
            if killer:
                if killer[0].color == current_agent.color:
                    if last_round == 1:
                        agents = [agent for agent in living_agents if
                                  agent != current_agent]
                        vote_data = {agent.color: 0.4 / len(agents) for agent
                                     in agents}
                        vote_data[living.color] = 0.6
                    else:
                        last_vote = find_last_vote_in_round(
                            crewmate_obs, current_agent, last_round,
                            living_agents
                        )
                        vote_data[last_vote] = 0.6
                        vote_data[reporter] = 0.4
                else:
                    vote_data[killer[0]] = 0.9
                    last_vote = find_last_vote_in_round(
                        crewmate_obs, current_agent, last_round,
                        living_agents
                    )
                    vote_data[last_vote] = 0.06
                    vote_data[reporter] = 0.04
    if not vote_data:
        body = seen_body(crewmate_obs, last_round, max_bodies)
        if body:
            body_loc = find_body_location(crewmate_obs, dead, last_round,
                                          max_bodies)
            suspect = has_main_suspect(last_round, living_agents,
                                       current_agent,
                                       body_loc, board)

            if suspect:
                suspicion = get_suspicion_table(crewmate_obs,
                                                living_agents,
                                                current_agent.color,
                                                body_loc, last_round,
                                                last_round)
                if suspicion and report_data is None:
                    report_data = [suspect, body]
                if mult:
                    for k in suspicion:
                        suspicion[k] *= 0.15
                vote_data = {k: suspicion[k] for k in suspicion}
                vote_data[suspect] = 0.7
                last = find_last_vote_in_round(crewmate_obs, current_agent,
                                               last_round, living_agents)
            else:
                voter = [agent for agent in living_agents if
                         agent != current_agent]
                vote_data = {v: 1 / len(voter) for v in voter}
        else:
            saw_body = None
            obs = None
            for agent in living_agents:
                obs = board.get_observation(agent)
                body = seen_body(obs, last_round, max_bodies)
                if body:
                    saw_body = agent
                    break
            if saw_body:
                body_loc = find_body_location(obs, dead, last_round,
                                              max_bodies)
                suspicion_table = get_suspicion_table(obs, living_agents,
                                                      current_agent.color,
                                                      body_loc, last_round,
                                                      last_round)
                main_suspect = has_main_suspect(last_round,
                                                living_agents, current_agent,
                                                body_loc, board)
                if not main_suspect:
                    body_loc = find_body_location(obs, dead, last_round,
                                                  max_bodies)
                    suspicion_table = get_suspicion_table(obs, living_agents,
                                                          current_agent.color,
                                                          body_loc, last_round,
                                                          last_round)
                    for suspect in suspicion_table:
                        suspicion_table[suspect] *= 0.75

                    lst_vote = find_last_vote_in_round(obs,
                                                       current_agent,
                                                       last_round,
                                                       living_agents)

                    vote_data = {sus: suspicion_table[sus]
                                 for sus in suspicion_table}
                    if lst_vote:
                        vote_data[lst_vote] = 0.25
                else:
                    for sus in suspicion_table:
                        suspicion_table[sus] *= 0.2
                    last_vote = find_last_vote_in_round(
                        crewmate_obs, current_agent, last_round,
                        living_agents
                    )
                    vote_data = {sus[0]: sus[1] for sus in suspicion_table}
                    vote_data[main_suspect.color] = 0.2
                    if last_vote:
                        vote_data[last_vote] = 0.2
            else:
                body_loc = find_body_location(obs, dead, last_round,
                                              max_bodies)
                suspicion_table = get_suspicion_table(obs, living_agents,
                                                      current_agent.color,
                                                      body_loc, last_round,
                                                      last_round)
                vote_data[reporter] = 0.2
                for sus in suspicion_table:
                    suspicion_table[sus] *= 0.65
                last_vote = find_last_vote_in_round(
                    crewmate_obs, current_agent, last_round,
                    living_agents
                )
                if last_vote:
                    vote_data[last_vote] = 0.15
    if len(report_data):
        return choose_action(normalize_probs(vote_data)), report_data[0], \
               report_data[1]
    return choose_action(normalize_probs(vote_data)), None, None


def get_crew_vote(crewmate_obs, living_agents, last_round,
                  max_bodies, current_agent, reporter, dead,
                  board, mult=True):
    report_data = dict()
    vote_data = dict()
    killer = have_seen_kill(crewmate_obs, living_agents, last_round,
                            max_bodies,
                            dead)
    if killer:
        vote_data = {killer[0]: 1}
        report_data = [killer[0], killer[1]]
    else:
        for living in living_agents:
            killer = have_seen_kill(board.get_observation(living),
                                    living_agents, last_round, max_bodies,
                                    dead)
            if killer:
                if killer[0].color == current_agent.color:
                    if last_round == 1:
                        agents = [agent for agent in living_agents if
                                  agent != current_agent]
                        vote_data = {agent.color: 0.4 / len(agents) for agent
                                     in agents}
                        vote_data[living.color] = 0.6
                    else:
                        last_vote = find_last_vote_in_round(
                            crewmate_obs, current_agent, last_round,
                            living_agents
                        )
                        vote_data[last_vote] = 0.6
                        vote_data[reporter] = 0.4
                else:
                    vote_data[killer[0]] = 0.9
                    last_vote = find_last_vote_in_round(
                        crewmate_obs, current_agent, last_round,
                        living_agents
                    )
                    vote_data[last_vote] = 0.06
                    vote_data[reporter] = 0.04
    if not vote_data:
        body = seen_body(crewmate_obs, last_round, max_bodies)
        body_loc = find_body_location(crewmate_obs, dead, last_round,
                                      max_bodies)
        if body:
            body_loc = find_body_location(crewmate_obs, dead, last_round,
                                          max_bodies)
            suspect = has_main_suspect(last_round, living_agents, current_agent
                                       , body_loc, board)
            if suspect:
                body_loc = find_body_location(crewmate_obs, dead, last_round,
                                              max_bodies)
                suspicion = get_suspicion_table(crewmate_obs,
                                                living_agents,
                                                current_agent.color,
                                                body_loc, last_round,
                                                last_round)
                if mult:
                    for k in suspicion:
                        suspicion[k] *= 0.15
                vote_data = {k: suspicion[k] for k in suspicion}
                vote_data[suspect] = 0.7
                last = find_last_vote_in_round(crewmate_obs, current_agent,
                                               last_round, living_agents)
            else:
                voter = [agent for agent in living_agents if
                         agent != current_agent]
                vote_data = {v: 1 / len(voter) for v in voter}
        else:
            saw_body = None
            obs = None
            for agent in living_agents:
                obs = board.get_observation(agent)
                body = seen_body(obs, last_round, max_bodies)
                if body:
                    saw_body = agent
                    break
            if saw_body:
                body_loc = find_body_location(crewmate_obs, dead, last_round,
                                              max_bodies)
                suspicion_table = get_suspicion_table(obs, living_agents,
                                                      current_agent.color,
                                                      body_loc, last_round,
                                                      last_round)
                main_suspect = has_main_suspect(last_round,
                                                living_agents, current_agent,
                                                body_loc, board)
                if not main_suspect:

                    for suspect in suspicion_table:
                        suspicion_table[suspect] *= 0.75

                    lst_vote = find_last_vote_in_round(obs,
                                                       current_agent,
                                                       last_round,
                                                       living_agents)

                    vote_data = {sus: suspicion_table[sus]
                                 for sus in suspicion_table}
                    if lst_vote:
                        vote_data[lst_vote] = 0.25
                else:
                    for sus in suspicion_table:
                        suspicion_table[sus] *= 0.2
                    last_vote = find_last_vote_in_round(
                        crewmate_obs, current_agent, last_round,
                        living_agents
                    )
                    vote_data = {sus[0]: sus[1] for sus in suspicion_table}
                    vote_data[main_suspect.color] = 0.2
                    if last_vote:
                        vote_data[last_vote] = 0.2
            else:
                body_loc = find_body_location(obs, dead, last_round,
                                              max_bodies)
                suspicion_table = get_suspicion_table(obs, living_agents,
                                                      current_agent.color,
                                                      body_loc, last_round,
                                                      last_round)
                vote_data[reporter] = 0.2
                for sus in suspicion_table:
                    suspicion_table[sus] *= 0.65
                last_vote = find_last_vote_in_round(
                    crewmate_obs, current_agent, last_round,
                    living_agents
                )
                if last_vote:
                    vote_data[last_vote] = 0.15

    t = choose_action(normalize_probs(vote_data))
    if type(t) == str or type(t) == np.str_:
        return t
    return t.color


############################IMPOSTOR VOTING#################################

def considerations(obs, last_round, myself, impostor_partner, agents):
    dict_res = dict()
    for round in range(1, last_round + 1):
        counter_for_myself = 0
        counter_for_partner = 0
        for agent in agents:
            if agent == myself or agent == impostor_partner:
                continue
            if obs[f"{agent.color} voted {myself.color} in round {round}"]:
                counter_for_myself += 1
            if obs[f"{agent.color} voted {impostor_partner.color} in round "
                   f"{round}"]:
                counter_for_partner += 1
            dict_res[agent.color] = 4 * counter_for_myself + counter_for_partner
    sorted_dict = list({k: v for k, v in sorted(dict_res.items(),
                                                key=lambda item: item[1],
                                                reverse=True)})
    return translate_to_probs(sorted_dict[:3])


def impostor_vote_flow(obs, living_agents, last_round,
                       max_bodies, current_agent, reporter, dead,
                       impostors, board):
    vote_data = dict()
    report_data = dict()
    body_report_data = dict()
    impostor = None
    for age in living_agents:
        if age == current_agent:
            continue
        if age.agent_type:
            impostor = age
            break
    if not impostor:
        l = [k for k in living_agents if k != current_agent]
        return l[np.random.choice(len(l))].color, -1, -1
    consider = considerations(obs, last_round, current_agent,
                              impostor, living_agents)
    if reporter.color == current_agent.color:
        poss_killers = board.get_possible_killers(current_agent)
        pos_bodies = board.get_possible_body_locs()
        for l in living_agents:
            if l.agent_type:
                if l in poss_killers:
                    poss_killers.remove(l)
        report_data = {killer.color: 1 / len(poss_killers)
                       for killer in poss_killers}
        body_report_data = {body: 1 / len(pos_bodies) for body in pos_bodies}
        vote_data[reporter.color] = 1
    else:
        for con in consider:
            consider[con] *= 0.10
        killer = None
        for agent in living_agents:
            obs = board.get_observation(agent)
            killer = have_seen_kill(obs, living_agents,
                                    last_round, max_bodies, dead)
            if killer:
                if killer[0] == current_agent:
                    for con in consider:
                        consider[con] *= 0.10
                    vote_data = {con: consider[con] for con in consider}
                    vote_data[agent.color] = 0.9
                    break
        if not killer:
            vote_data = {con: consider[con] for con in consider}
            vote_data[killer[0]] = 0.9
        else:
            obs = board.get_observation(reporter)
            body = seen_body(obs, last_round, max_bodies)
            if body:
                killer = obs[f'killer according to reporter in round '
                             f'{last_round}']
                if killer[0] == current_agent:
                    for con in consider:
                        consider[con] *= 0.2
                    vote_data = {c: consider[c] for c in consider}
                    vote_data[reporter.color] = 0.8
                elif killer[0] in impostors:
                    for con in consider:
                        consider[con] *= 0.1
                    vote_data = {c: consider[c] for c in consider}
                    vote_data[impostor.color] = 0.5
                    vote_data[reporter.color] = 0.4
                else:
                    for con in consider:
                        consider[con] *= 0.3
                    vote_data = {c: consider[c] for c in consider}
                    vote_data[killer[0].color] = 0.7
            else:
                vote_data = {c: consider for c in consider}
    val1 = None
    if report_data:
        val1 = choose_action(report_data)
    val2 = None
    if body_report_data:
        val2 = choose_action(body_report_data)
    return choose_action(normalize_probs(vote_data)), val1, val2


def impostor_vote(obs, living_agents, last_round,
                  max_bodies, current_agent, reporter, dead,
                  impostors, board):
    impostor = None
    for age in living_agents:
        if age == current_agent:
            continue
        if age.agent_type:
            impostor = age
            break
    if not impostor:
        l = [k for k in living_agents if k != current_agent]
        return l[np.random.choice(len(l))].color
    consider = considerations(obs, last_round, current_agent,
                              impostor, living_agents)
    if reporter.color == current_agent.color:
        poss_killers = board.get_possible_killers(current_agent)
        pos_bodies = board.get_possible_body_locs()
        for l in living_agents:
            if l.agent_type:
                if l in poss_killers:
                    poss_killers.remove(l)
        report_data = {killer: 1 / len(poss_killers)
                       for killer in poss_killers}
        body_report_data = {body: 1 / len(pos_bodies) for body in pos_bodies}
        vote_data = {v: 1 / len(living_agents) for v in living_agents
                     if v not in impostors}

    else:
        for con in consider:
            consider[con] *= 0.10
        killer = None
        for agent in living_agents:
            obs = board.get_observation(agent)
            killer = have_seen_kill(obs, living_agents,
                                    last_round, max_bodies, dead)
            if killer:
                if killer[0] == current_agent:
                    for con in consider:
                        consider[con] *= 0.10
                    vote_data = {con: consider[con] for con in consider}
                    vote_data[agent.color] = 0.9
                    break
        if not killer:
            vote_data = {con: consider[con] for con in consider}
        else:
            obs = board.get_observation(reporter)
            body = seen_body(obs, last_round, max_bodies)
            if body:
                killer = obs[f'killer according to reporter in round '
                             f'{last_round}']
                if killer == current_agent:
                    for con in consider:
                        consider[con] *= 0.2
                    vote_data = {c: consider[c] for c in consider}
                    vote_data[reporter.color] = 0.8
                elif killer in impostors:
                    for con in consider:
                        consider[con] *= 0.1
                    vote_data = {c: consider[c] for c in consider}
                    vote_data[impostor.color] = 0.5
                    vote_data[reporter.color] = 0.4
                else:
                    for con in consider:
                        consider[con] *= 0.3
                    vote_data = {c: consider[c] for c in consider}
                    vote_data[killer] = 0.7
            else:
                vote_data = {c: consider for c in consider}
    return choose_action(normalize_probs(vote_data))


def choose_action(dict_probs):
    lst = list(dict_probs.items())
    first = [it[0] for it in lst]
    second = [it[1] for it in lst]

    choice = np.random.choice(first, replace=False, p=second)
    return choice
