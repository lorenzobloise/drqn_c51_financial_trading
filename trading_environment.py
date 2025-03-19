from common import hot_encoding, save_data_structure
from data_preprocessing import Data
import numpy as np
import copy

class TradingEnvironment:

    def __init__(self, data: Data, initial_value=100000):
        self.initial_value = initial_value
        self.portfolio = [float(initial_value)]
        self.actions = []
        self.prev_close = None
        self.data = data
        self.spread = 0.005
        self.trade_size = 10000

    def merge_state_action(self, state, a_variable):
        if state is None:
            return []
        T = len(state)
        actions_for_state = self.actions[self.data.n:][:T-1]
        actions_for_state.append(a_variable)
        diff = T - len(actions_for_state)
        if diff > 0:
            actions_for_state.extend([a_variable] * diff)
        result = []
        for s, a in zip(state, actions_for_state):
            new_s = copy.deepcopy(s)
            new_s.extend(hot_encoding(a))
            result.append(new_s)
        result = np.asarray(result)
        return result

    # Returns: state
    def reset(self) -> object:
        self.portfolio = [float(self.initial_value)]
        self.data.reset()
        self.actions.append(0)
        closing, state_initial = self.data.next()
        self.prev_close = closing
        return self.merge_state_action(state_initial, 0)

    # Returns: actions, rewards, new_states, selected new_state, done
    def step(self, action, step) -> object:
        actions = [-1, 0, 1]
        v_old = self.portfolio[-1]
        try:
            closing, state_next = self.data.next()
            done = False
        except:
            state_next = None
            done = True
        # Added: if state_next is None, returns
        if done:
            return actions, [], [], [], done # actions, rewards, new_states, new_states[action+1], done
        new_states = []
        for a in actions:
            new_states.append(self.merge_state_action(state_next, a))
        current_closed = closing
        if self.prev_close is not None:
            current_open = self.prev_close
            self.prev_close = current_closed
        else:
            raise Exception("No previous close price saved!")
        v_new = []
        for a in actions:
            commission = self.trade_size * np.abs(a - self.actions[-1]) * self.spread
            v_new.append(v_old + a * self.trade_size * (current_closed - current_open) - commission)
        v_new = np.asarray(v_new)
        rewards = []
        for i in range(len(v_new)):
            if v_new[i] * v_old > 0 and v_old != 0:
                rewards.append(np.log(v_new[i] / v_old))
            else:
                rewards.append(-1)
        rewards = np.asarray(rewards)
        if (step + 1) % 1000 == 0:
            print(float(v_new[action+1]))
        self.actions.append(int(action))
        self.portfolio.append(float(v_new[action+1]))
        return actions, rewards, new_states, new_states[action+1], done

    def store_test_result(self, args):
        save_data_structure(self.actions, args.action_path + '/' + args.stock + '_action.json')
        save_data_structure(self.portfolio, args.portfolio_path + '/' + args.stock + '_portfolio.json')