class TradingAgent:

    def act(self, state):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def store(self, state, actions, new_states, rewards, action, step):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def optimize(self, step):
        raise NotImplementedError("This method must be implemented by a subclass.")