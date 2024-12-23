import sys

from core.game_runner import GameRunner
import tkinter as tk
from tkinter import Frame, Canvas
from config.constants import winner_ids


class BoardGUI(Frame):
    def __init__(self, row=25, column=25, times=2):
        self.root = tk.Tk()
        Frame.__init__(self, self.root)
        self.pack(expand='yes', fill='both')
        self.canvas = Canvas(self)
        self.canvas.config(width=1280, height=768, bg='red')
        self.canvas.pack(expand='yes', fill='both')
        self.ROWS = row
        self.COLUMNS = column
        self.label_frame_list = []
        self.voting_board = None
        self.times = times
        self.__reset()

    def __reset(self):
        for r in range(0, self.ROWS):
            for c in range(0, self.COLUMNS):
                color = self.choose_color(GameRunner.board[r][c])
                label = tk.LabelFrame(master=self.canvas, width=51.2,
                                      height=25, bg=color)
                self.label_frame_list.append(label)
                self.label_frame_list[-1].grid(row=r, column=c)

    def update_board(self):
        winning = GameRunner.run_game()
        board = GameRunner.board
        if len(winning) == 2:
            for w in sorted(winning[0].keys()):
                for j in winning[0][w]:
                    color = self.choose_color(board[j[0]][j[1]])
                    self.label_frame_list[self.ROWS * j[0] + j[1]]. \
                        config(bg=color)
                    self.label_frame_list[self.ROWS * j[0] + j[1]].grid(
                        row=j[0], column=j[1])
            if winning[1]:
                for cor in winning[1]:
                    color = self.choose_color(board[cor[0]][cor[1]])
                    self.label_frame_list[self.ROWS * cor[0] + cor[1]]. \
                        config(bg=color)
                    self.label_frame_list[self.ROWS * cor[0] + cor[1]].grid(
                        row=cor[0], column=cor[1])
        else:
            for w in winning[1]:
                for j in sorted(winning[1][w]):
                    color = self.choose_color(board[j[0]][j[1]])
                    self.label_frame_list[self.ROWS * j[0] + j[1]]. \
                        config(bg=color)
                    self.label_frame_list[self.ROWS * j[0] + j[1]].grid(
                        row=j[0], column=j[1])
            if winning[2]:
                for cor in winning[2]:
                    color = self.choose_color(board[cor[0]][cor[1]])
                    self.label_frame_list[self.ROWS * cor[0] + cor[1]]. \
                        config(bg=color)
                    self.label_frame_list[self.ROWS * cor[0] + cor[1]].grid(
                        row=cor[0], column=cor[1])
            print(f"{winner_ids[winning[0]]} win")
            self.root.destroy()
            sys.exit(0)
        self.root.after(50, self.update_board)

    def choose_color(self, tile):
        if tile.is_wall:
            return "black"
        elif tile.is_table:
            return "#654321"
        elif tile.is_task:
            return "light green"
        elif tile.bodies:
            # return "white"
            return "#FF9999"  # "#66FF66"
        elif tile.agents:
            return list(tile.agents)[0].color
        return "white"
