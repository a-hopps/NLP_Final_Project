import math
import tensorflow as tf
import numpy as np
import random

def format_time(elapsed_time):
    """
    Convert elapsed time in seconds to a formatted string in the format hh:mm:ss.

    Args:
        elapsed_time (float): The elapsed time in seconds.
    Returns:
        str: A formatted string representing the elapsed time in the format hh:mm:ss.
    """

    # Convert total seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    hours = int(math.ceil(hours))
    minutes = int(math.ceil(minutes))
    seconds = int(math.ceil(seconds))

    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Format time as hh:mm:ss
    return formatted_time

def set_random_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)