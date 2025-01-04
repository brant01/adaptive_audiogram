from enum import Enum

class Decision(Enum):
    """
    Decision class
    """
    Yes = 0
    No = 1
    
class Ear(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3

class TrialType(Enum):
    """
    Trial type class
    """
    Test = 0
    Train = 1
    
