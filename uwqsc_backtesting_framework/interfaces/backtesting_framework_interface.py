"""
Interface for a Backtesting Framework Algorithm
"""

from abc import ABC

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import IAlgorithm


class IBacktestingFramework(ABC):
    """
    Interface for a Backtesting Framework Algorithm
    """

    def __init__(self,
                 name: str,
                 algorithm: IAlgorithm):
        self._name = name
        self.algorithm = algorithm
