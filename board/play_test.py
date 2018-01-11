from gomokuZero.board.play import Player, Game


player_1 = Player()
player_2 = Player()
game = Game(player_1, player_2, time_delay=1)
game.play()
