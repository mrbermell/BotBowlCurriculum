import botbowlcurriculum
import gym
import ffai
from botbowlcurriculum.Curriculum import Lecture, Academy
from botbowlcurriculum.utils import get_empty_game_turn

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
    sub1 = 6
    sub2 = 3
    sub3 = 4

    lect = Lecture("", [sub1, sub2, sub3])

    for level in range(lect.max_level+1):
        lect.level = level
        a1, a2, a3 = lect.get_sublevels()
        assert 0 <= a1 < sub1 and 0 <= a2 < sub2 and 0 <= a3 < sub3

def test_academy():
    dummy_lecture = Lecture("dummy lecture", [10])
    academy = Academy([dummy_lecture])

    outcome = np.array([0, 0, 1], dtype=np.int)
    academy.log_training(outcome)

    outcome = np.random.randint(0, 2, size=[30, 3])
    outcome[:,0:2] = 0

    academy.log_training(outcome)

    academy.evaluate()
    reports = academy.get_report_dicts()


def test_probs_and_levels():
    academy = botbowlcurriculum.make_academy()
    prob_lvls = academy.get_probs_and_levels()
    assert prob_lvls.shape == (len(academy), 2)
    assert (prob_lvls[:, 0] == 0).all()
    assert np.round(np.sum(prob_lvls[:,1]), 7) == 1.0

