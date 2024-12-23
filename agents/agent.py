from config.constants import *


class Agent:

    def __init__(self, color, agent_type, vision_range, decider_type):
        """
        :param color:  Agent color (identifies the player)-string
        :param agent_type:   Impostor or Crew mate-string
        :param is already called meeting, limited for game.
        """
        self.color = color
        self.agent_type = agent_type
        self.vision_range = vision_range
        self.is_dead = False
        self.table_calls_left = 1
        self.cur_task_time = 0
        self.table_cd = TABLE_CD
        self.decider_type = decider_type
        self.task_goal_coord = None
        self.checkpoint_goal_coord = None
