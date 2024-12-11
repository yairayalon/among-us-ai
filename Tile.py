class Tile:
    """
    This class represents any tile in the board
    """

    def __init__(self, is_wall=False, is_task=False, rtc=-1, is_table=False):
        """
        A constructor for Tile
        :param is_wall: Is this a wall tile
        :param is_task: Is this a task tile
        :param rtc: Time to complete if it's a task tile
        :param is_table: Is this a table tile
        """
        self.is_wall = is_wall
        self.is_task = is_task
        self.rtc = rtc
        self.is_table = is_table
        self.agents = set()
        self.bodies = set()
        self.killed_by = set()
