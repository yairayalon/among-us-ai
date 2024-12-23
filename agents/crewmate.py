from agents.agent import Agent


class Crewmate(Agent):

    def __init__(self, color, agent_type, vision_range, tasks, decider_type):
        super().__init__(color, agent_type, vision_range, decider_type)
        self.board = None
        self.cur_task_time = 0
        self.is_performing_task = False
        self.tasks = {task for task in tasks}

    def do_task(self):
        self.is_performing_task = True
        self.cur_task_time += 1
        if self.cur_task_time == self.board.get_task_rtc(self):
            self.cur_task_time = 0
            self.tasks.remove(self.board.get_task_loc(self))
            self.is_performing_task = False
