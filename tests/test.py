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

    reports = academy.evaluate()


def test_probs_and_levels():
    academy = botbowlcurriculum.make_academy()
    prob_lvls = academy.get_probs_and_levels()
    assert prob_lvls.shape == (len(academy), 2)
    assert (prob_lvls[:, 0] == 0).all()
    assert np.round(np.sum(prob_lvls[:,1]), 7) == 1.0

def test_full_academy():
    academy = botbowlcurriculum.make_academy()
    num_outcomes = len(academy)
    levels_seen = set()

    for i in range(100):
        prob_lvls = academy.get_probs_and_levels()

        for j in range(len(academy)):
            assert prob_lvls[j, 0] == academy.lect_histo[j].lecture.get_level()

            levels_seen.add(int(prob_lvls[j, 0]))

        outcome = np.zeros((num_outcomes, 3), dtype=np.int)
        outcome[:,0] = np.array(list(range(num_outcomes)))
        outcome[:,1] = prob_lvls[:,0]
        outcome[:,2] = 1

        academy.log_training(outcome)
        if i > 1 and i % 10 == 0:
            academy.evaluate()

    #assert {0, 1, 2, 3, 4} in levels_seen
    assert 0 in levels_seen
    assert 1 in levels_seen
    assert 2 in levels_seen
    assert 3 in levels_seen
    assert 4 in levels_seen

