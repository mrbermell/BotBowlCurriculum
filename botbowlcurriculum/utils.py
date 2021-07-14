from collections import Iterable
from copy import deepcopy
import random
from random import randint

import numpy as np
from ffai.core import Square, D3, D6, D8, BBDie, Action, Agent, ActionType, load_rule_set, load_team_by_filename, Game, load_config, Turn

class DoNothingBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)

    def act(self, game):
        for action_type in [ActionType.END_TURN, ActionType.END_SETUP, ActionType.END_PLAYER_TURN,
                            ActionType.SELECT_NONE, ActionType.HEADS, ActionType.KICK, ActionType.SELECT_DEFENDER_DOWN,
                            ActionType.SELECT_DEFENDER_STUMBLES, ActionType.SELECT_ATTACKER_DOWN,
                            ActionType.SELECT_PUSH, ActionType.SELECT_BOTH_DOWN, ActionType.DONT_USE_REROLL,
                            ActionType.DONT_USE_APOTHECARY, ActionType.SETUP_FORMATION_LINE,
                            ActionType.SETUP_FORMATION_ZONE, ActionType.SETUP_FORMATION_SPREAD,
                            ActionType.SETUP_FORMATION_WEDGE]:
            for action in game.state.available_actions:
                if action_type == ActionType.END_SETUP and not game.is_setup_legal(game.get_agent_team(game.actor)):
                    continue
                if action.action_type == action_type:
                    return Action(action_type)
        # Take random action
        action_choice = game.rnd.choice(game.state.available_actions)
        position = game.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = game.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
        return Action(action_choice.action_type, position=position, player=player)

    def new_game(self, game, team):
        pass

    def end_game(self, game):
        pass

game_turn_memoized = {}

def get_empty_game_turn(config="bot-bowl-iii", home_receiving=None, turn=0, clear_board=True, away_agent=None):
    if type(config) == str:
        config = load_config(config)
    config.competition_mode = False
    config.fast_mode = True
    seed = np.random.randint(0, 2 ** 32)

    pitch_size = config.pitch_max

    key = f"{pitch_size} {turn} {clear_board} {home_receiving}"
    if key in game_turn_memoized:
        game = deepcopy(game_turn_memoized[key])
        game.set_seed(seed)
        if away_agent is not None:
            game.replace_away_agent(away_agent)
        return game

    D3.FixedRolls = []
    D6.FixedRolls = [3, 4, 3, 4, 3, 4, 3, 4, 3]  # No crazy kickoff or broken armors
    D8.FixedRolls = []
    BBDie.FixedRolls = []

    ruleset = load_rule_set(config.ruleset)

    size_suffix = f"-{pitch_size}" if pitch_size != 11 else ""
    hometeam = "human"
    awayteam = "human"
    home = load_team_by_filename(hometeam + size_suffix, ruleset, board_size=pitch_size)
    away = load_team_by_filename(awayteam + size_suffix, ruleset, board_size=pitch_size)
    game = Game(seed, home, away, home_agent=Agent("human1", human=True), away_agent=Agent("human2", human=True), config=config)
    game.init()
    game.step(Action(ActionType.START_GAME))

    if home_receiving is not None:
        game.step(Action(ActionType.HEADS))
        if (game.state.coin_toss_winner == home and home_receiving) or (game.state.coin_toss_winner == away and not home_receiving):
            game.step(Action(ActionType.RECEIVE))
        else:
            game.step(Action(ActionType.KICK))


    if turn > 0:
        while type(game.get_procedure()) is not Turn or game.state.home_team.state.turn != turn or game.state.available_actions[0].team != home:
            game.step(game._forced_action())

        if clear_board:
            game.clear_board()

    if away_agent is not None:
        game.replace_away_agent(away_agent)
    else:
        game.replace_away_agent(DoNothingBot("Do nothing bot"))

    D6.FixedRolls = []
    game.step()
    game_turn_memoized[key] = deepcopy(game)

    return game


def get_players(game, team):

    if team == "home":
        players = game.state.home_team.players
    elif team == "away":
        players = game.state.away_team.players
    elif team == "both":
        players = game.state.away_team.players + game.state.home_team.players
    else:
        assert False, f"team='{team}' not valid"
    num = len(players)
    return random.sample(players, num)

def get_boundary_square(game, steps, from_position):
    """
    :param game:
    :param steps:
    :param from_position:
    :return: position that is 'steps' away from 'from_position'
    checks are done so it's square is available
    """

    steps = int(steps)

    if steps == 0:
        if game.state.pitch.board[from_position.y][from_position.x] is None:
            return from_position
        else:
            steps += 1


    board_x_max = len(game.state.pitch.board[0]) - 2
    board_y_max = len(game.state.pitch.board) - 2

    assert steps > 0

    avail_squares = steps * 8

    squares_per_side = 2 * steps

    i = 0
    while True:
        i += 1
        assert i < 5000

        sq_index = randint(0, avail_squares - 1)
        steps_along_side = sq_index % squares_per_side

        # up, including left corner
        if sq_index // squares_per_side == 0:
            dx = - steps + steps_along_side
            dy = - steps
        # right, including upper corner
        elif sq_index // squares_per_side == 1:
            dx = + steps
            dy = - steps + steps_along_side
        # down, including right corner
        elif sq_index // squares_per_side == 2:
            dx = + steps - steps_along_side
            dy = + steps
            # left, including lower corner
        elif sq_index // squares_per_side == 3:
            dx = - steps
            dy = + steps - steps_along_side
        else:
            assert False

        position = Square(from_position.x + dx, from_position.y + dy)
        x = position.x
        y = position.y

        if x < 1 or x > board_x_max or y < 1 or y > board_y_max:
            continue

        if game.state.pitch.board[y][x] is None:  # it should y first, don't ask.
            break

    return position


def scatter_ball(game, steps, from_position):
    # scatters ball a certain amount of steps away for original position
    # checks are done so it's not out of bounds or on a player
    if steps > 0:
        pos = get_boundary_square(game, steps, from_position)
    else:
        pos = from_position

    game.get_ball().move_to(pos)
    game.get_ball().is_carried = False


def set_player_state(player, p_used=None, p_down=None):
    if p_down is not None:
        player.state.up = not random.random() < p_down

    if p_used is not None and player.state.up:
        player.state.used = random.random() < p_used


def move_player_within_square(game, player, x, y, give_ball=False, p_used=None, p_down=None):
    # places the player at a random position within the given square.

    assert isinstance(give_ball, bool)

    board_x_max = len(game.state.pitch.board[0]) - 2
    board_y_max = len(game.state.pitch.board) - 2

    xx = sorted(x) if isinstance(x, Iterable) else (x, x)
    yy = sorted(y) if isinstance(y, Iterable) else (y, y)

    x_min = max(xx[0], 1)
    x_max = min(xx[1], board_x_max)
    y_min = max(yy[0], 1)
    y_max = min(yy[1], board_y_max)

    assert x_min <= x_max
    assert y_min <= y_max

    i = 0

    while True:
        i += 1
        assert i < 5000

        x = randint(x_min, x_max)
        y = randint(y_min, y_max)

        # if x < 1 or x > board_x_max or y < 1 or y > board_y_max:
        #    continue

        if game.state.pitch.board[y][x] is None:
            break

    game.put(player, Square(x, y))
    if give_ball == True:
        game.get_ball().move_to(player.position)
        game.get_ball().is_carried = True

    set_player_state(player, p_used=p_used, p_down=p_down)


def move_player_out_of_square(game, player, x, y, p_used=None, p_down=None):
    # places the player at a random position that is not in the given square.

    xx = x if isinstance(x, Iterable) else (x, x)
    yy = y if isinstance(y, Iterable) else (y, y)

    x_min = xx[0]
    x_max = xx[1]
    y_min = yy[0]
    y_max = yy[1]

    board_x_max = len(game.state.pitch.board[0]) - 2
    board_y_max = len(game.state.pitch.board) - 2

    i = 0

    while True:
        x = randint(1, board_x_max)
        y = randint(1, board_y_max)

        if x_min <= x <= x_max and y_min <= y <= y_max:
            i += 1
            assert i < 5000
            continue

        if game.state.pitch.board[y][x] is None:
            break

    game.put(player, Square(x, y))
    set_player_state(player, p_used=p_used, p_down=p_down)


def move_players_out_of_square(game, players, x, y, p_used=None, p_down=None):
    for p in players:
        move_player_out_of_square(game, p, x, y, p_used=p_used, p_down=p_down)


def swap_game(game):
    moved_players = []
    board_x_max = len(game.state.pitch.board[0]) - 2

    player_to_move = [p for p in get_players(game, 'both') if p.position is not None]
    for p in player_to_move:

        if p in moved_players:
            continue

        old_x = p.position.x
        new_x = 27 - old_x

        potential_swap_p = game.state.pitch.board[p.position.y][new_x]
        if potential_swap_p is not None:
            game.move(potential_swap_p, Square(0, 0))

        game.move(p, Square(new_x, p.position.y))

        if potential_swap_p is not None:
            game.move(potential_swap_p, Square(old_x, p.position.y))
            moved_players.append(potential_swap_p)

            # ball_pos = game.get_ball().position

    # ball_new_x  = 27-ball_pos.x
    # ball_y      = ball_pos.y

    # game.get_ball().move_to( Square(ball_new_x, ball_y) )

    # assert game.get_ball().position.x ==  ball_new_x
    pass

def get_game_data(game):
    x_max = len(game.state.pitch.board[0]) - 2
    y_max = len(game.state.pitch.board) - 2

    return x_max, y_max, get_players(game, "home"), get_players(game, "away")

def pop_role(players, role):
    for i, player in enumerate(players):
        if player.role.name == role:
            return players.pop(i)

    assert False, f"No role='{role}' in the list"