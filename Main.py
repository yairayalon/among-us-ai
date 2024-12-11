from GameRunner import GameRunner
from BoardGUI import BoardGUI

if __name__ == '__main__':
    # Choose the type of run you want, and comment the other ones!
    # All runs generate CSV files documenting the games in the provided dirs
    # (used for training)

    # NONE GUI VERSION: CSV Generation
    # GameRunner.init_game(False)
    # for i in range(100):
    #     a1 = 0
    #     while not a1:
    #         a1, a2, a3 = GameRunner.run_game()
    #     print(i, a1)

    # NONE GUI VERSION: Statistics + CSV Generation
    # GameRunner.init_game(False)
    # cm_wins, imp_wins = 0, 0
    # total_games = 100
    # for i in range(total_games):
    #     a1 = 0
    #     while not a1:
    #         a1, a2, a3 = GameRunner.run_game()
    #     cm_wins += max(0, a1)
    #     imp_wins -= min(0, a1)
    #     print(i, a1)
    # print(f"Crewmates' won {cm_wins} games out of {total_games}")
    # print(f"Crewmates' win rate is {cm_wins / total_games}")
    # print(f"Impostors' won {imp_wins} games out of {total_games}")
    # print(f"Impostors' win rate is {imp_wins / total_games}")

    # SINGLE GAME GUI VERSION
    GameRunner.init_game(True)
    bg = BoardGUI()
    bg.root.after(50, bg.update_board)
    bg.root.mainloop()
