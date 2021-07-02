import gym
import ffai
from botbowlcurriculum.Curriculum import get_empty_game_turn, Lecture

from ffai import GotebotWrapper
import numpy as np

def test_empty_turn():
    game = get_empty_game_turn(home_receiving=True)
    assert game.get_receiving_team() == game.state.home_team
    assert game.state.available_actions[0].team == game.state.home_team

    game = get_empty_game_turn(home_receiving=False)
    assert game.get_receiving_team() == game.state.away_team
    assert game.state.available_actions[0].team == game.state.home_team

    game = get_empty_game_turn(turn=3, home_receiving=False)
    assert game.get_receiving_team() == game.state.away_team
    assert game.state.available_actions[0].team == game.state.home_team
    assert game.state.home_team.state.turn == 3

    game = get_empty_game_turn(turn=3, home_receiving=True)
    assert game.get_receiving_team() == game.state.home_team
    assert game.state.available_actions[0].team == game.state.home_team
    assert game.state.home_team.state.turn == 3


def test_sublevel_calculation():
    lect = Lecture("", [6,3,4])

    for level in range(lect.max_level+1):
        lect.level = level
        a1, a2, a3 = lect.get_sublevels()


