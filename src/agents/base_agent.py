class BaseAgent():
    """Main class for implementing reinforcement learning agents

    It must be extended by a subclass and the select_action method must be 
    overriden. 
    """
    def __init__(self) -> None:
        pass

    def select_action(self, observation):
        raise NotImplementedError("You should implement this method in a subclass")