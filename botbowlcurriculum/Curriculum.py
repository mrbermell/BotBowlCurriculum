import numpy as np
from functools import reduce

from ffai.core.load import *
from operator import mul
import statsmodels.stats.proportion as stats

class Lecture:
    def __init__(self, name, sub_levels):
        self.name = name
        self.level = 0
        self.exceptions_thrown = 0

        assert len(sub_levels) in list(range(1,10))
        self.sub_levels = sub_levels
        self.max_level = reduce(mul, self.sub_levels, 1) - 1 # -1 because level=0 is the first level.

    # MUST override in subclass
    def reset_game(self, config):
        """
        :paran config: integer of pitch size, (currently 3,5,7,11)
        :return: return a fully initialized game object, with opp_agent initialized
        """
        raise NotImplementedError("Must be overridden by subclass")

    # Overwrite to change from default 8/9.
    @property
    def expected_success_rate(self):
        """
        Not sure how to use this. TODO TBD
        """
        return 8/9 # 3+ with re-roll may fail.

    # Overwrite to change behavior. Default is winner of game
    def _evaluate(self, game):
        """ To be overwritten!
        :param game: game object to be judged
        :return: int describing outcome. -1 = failed, 0=draw, 1=success.
        """
        outcome = game.state.home_team.state.score - game.state.away_team.state.score
        return outcome // abs(outcome) if outcome != 0 else 0

    def evaluate(self, game):
        """
        :param game: game object to be judged
        :return: array shape=(2,), dtype=np.int. Containing [level, outcome]
        """
        level = self.get_level()
        outcome = self._evaluate(game)
        return np.array((level, outcome), dtype=np.int)

    def get_sublevels(self):
        num_sublvls = len(self.sub_levels)
        current_sub_level = [0]*num_sublvls
        level = self.get_level()

        for i in range(num_sublvls):
            denominator = reduce(mul, self.sub_levels[:i], 1)
            current_sub_level[i] = (level//denominator) % self.sub_levels[i]

        return tuple(current_sub_level)

    def increase_level(self):
        self.level += 1 * (self.level < self.max_level)

    def decrease_level(self):
        self.level -= 1 * (0 < self.level)

    def get_diff(self):
        return min(self.level, self.max_level) / self.max_level

    def get_level(self):
        return min(int(self.level), self.max_level)


class LectureHistory:
    MAX_SIZE = 200

    def __init__(self, lecture):
        self.lecture = lecture
        self.outcomes = None
        self.index = None
        self.episodes = 0
        self.steps = 0
        self.max_acheived = -1

        self.reset_self()

    def reset_self(self):
        self.outcomes = np.zeros((self.MAX_SIZE,), dtype=np.int)
        self.index = 0

    def log(self, level, outcome):
        i = self.index

        if level == self.lecture.get_level() and i < self.MAX_SIZE:
            self.outcomes[i] = 1 if outcome == 1 else 0
            self.episodes += 1
            self.index += 1

            if outcome == 1 and self.max_acheived < level:
                self.max_acheived = level

    def evaluate(self):
        if self.index <= 5:
            return

        target_p = self.lecture.expected_success_rate
        outcomes = self.outcomes[:self.index]
        trials = len(outcomes)
        successes = sum(outcomes)
        outcome_p = successes / trials

        conf_low, conf_high = stats.proportion_confint(successes, trials, method='wilson', alpha=0.03)

        # increase level
        if outcome_p >= target_p:
            self.lecture.increase_level()
            self.reset_self()

        # decrease level
        elif conf_high < target_p:
            self.lecture.decrease_level()
            self.reset_self()

        # reset
        elif trials > 0.75 * self.MAX_SIZE:
            self.reset_self()

        # continue


    def report(self, with_name=False):
        lvl = str(self.lecture.get_level())
        max_lvl = self.lecture.max_level
        avg = self.outcomes[:self.index].mean() if self.index>0 else 0
        # prob        = self.lec_prob_soft[lec_index]
        # reward      = self.rewards[lec_index,:].mean()

        s = f"ep={self.episodes}, steps={self.steps}, lvl= {lvl} ({self.max_acheived})/{max_lvl}), avg={avg}"
        return s

    def get_progress_score(self):
        score = 10.0 # Entropy term
        return score


class Academy:
    def __init__(self, lectures):
        self.lect_histo = [LectureHistory(lecture) for lecture in lectures]
        self.lec_prob = None

        self._update_probs()

    def _update_probs(self):
        scores = np.array([lect_hist.get_progress_score() for lect_hist in self.lect_histo])
        self.lec_prob = scores / sum(scores)
        assert round(sum(self.lec_prob), 3) == 1.0

    def evaluate(self):
        for lect in self.lect_histo:
            lect.evaluate()
        self._update_probs()

    def get_next_lecture(self):
        rand_int = np.random.choice(list(range(len(self.lect_histo))), 1, p=self.lec_prob)[0]
        return self.lect_histo[rand_int], rand_int

    def log_training(self, outcomes):
        if outcomes.shape == (3,):
            outcomes = np.expand_dims(outcomes, axis=0)

        assert len(outcomes.shape) == 2 and outcomes.shape[1]==3, f"Shape='{outcomes.shape}' is wrong"

        for outcome in outcomes:
            index, level, success = np.round(outcome)
            self.lect_histo[index].log(level, success)

    def __len__(self):
        return len(self.lect_histo)

    def report(self, filename=None):
        # render plots

        max_name_len = max([len(l.lecture.name) for l in self.lect_histo])

        s = ""
        for l in self.lect_histo:
            name = l.lecture.name
            extra_spaces = max_name_len - len(name)

            s += l.lecture.name +": "+ " " * extra_spaces
            s += l.report() + "\n"

        return s


if __name__ == "__main__":
    import statsmodels.stats.proportion as stats
    import matplotlib.pyplot as plt

    p = np.linspace(0.0, 1, 400)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 1.05, 0.1))

    alpha = 0.1

    for n in [5, 10, 20, 50, 100]:
        p = np.zeros(n+1)
        y = np.zeros(n+1)
        for i in range(n+1):
            p[i] = i/n
            y[i] = stats.proportion_confint(n*p[i], n, method='wilson',
                                            alpha=alpha)[0]

        plt.plot(p, y, 'o-', label=f"n={n}")

    plt.plot([0, 1], [0, 1], '--', label=f"truth")
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("observed p")
    plt.ylabel("confidence p")
    plt.title(f"confidence = {1-alpha}")
    plt.show()

