"""
Historical data task modules.

This package contains tasks for fetching and processing historical market data.
Module organization:
- base: Common utilities and mixins
- contracts: Contract resolution tasks
- data_fetching: Data retrieval tasks  
- data_processing: Data normalization and saving tasks

Note: Full refactoring from monolithic historical_data.py is planned for Phase 2.2
"""

# For backward compatibility, expose all tasks from the main historical_data module
from ..historical_data import *