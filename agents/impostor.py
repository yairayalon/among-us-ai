from agents.agent import Agent


class Impostor(Agent):

    def __init__(self, color, agent_type, vision_range, kill_cd, decider_type):
        super().__init__(color, agent_type, vision_range, decider_type)
        self.kill_cd = kill_cd
