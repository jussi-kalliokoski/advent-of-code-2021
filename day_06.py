EXAMPLE_INPUT = """
3,4,3,1,2
"""

PUZZLE_INPUT = """
1,1,1,1,2,1,1,4,1,4,3,1,1,1,1,1,1,1,1,4,1,3,1,1,1,5,1,3,1,4,1,2,1,1,5,1,1,1,1,1,1,1,1,1,1,3,4,1,5,1,1,1,1,1,1,1,1,1,3,1,4,1,1,1,1,3,5,1,1,2,1,1,1,1,4,4,1,1,1,4,1,1,4,2,4,4,5,1,1,1,1,2,3,1,1,4,1,5,1,1,1,3,1,1,1,1,5,5,1,2,2,2,2,1,1,2,1,1,1,1,1,3,1,1,1,2,3,1,5,1,1,1,2,2,1,1,1,1,1,3,2,1,1,1,4,3,1,1,4,1,5,4,1,4,1,1,1,1,1,1,1,1,1,1,2,2,4,5,1,1,1,1,5,4,1,3,1,1,1,1,4,3,3,3,1,2,3,1,1,1,1,1,1,1,1,2,1,1,1,5,1,3,1,4,3,1,3,1,5,1,1,1,1,3,1,5,1,2,4,1,1,4,1,4,4,2,1,2,1,3,3,1,4,4,1,1,3,4,1,1,1,2,5,2,5,1,1,1,4,1,1,1,1,1,1,3,1,5,1,2,1,1,1,1,1,4,4,1,1,1,5,1,1,5,1,2,1,5,1,1,1,1,1,1,1,1,1,1,1,1,3,2,4,1,1,2,1,1,3,2
"""

import tensorflow as tf
import tfutils

def __parse(text: tf.Tensor) -> tf.Tensor:
    numbers = tf.strings.to_number(tf.strings.split(tf.strings.strip(text), ","), out_type=tf.int64)
    return tf.concat([tf.shape(tf.where(tf.equal(numbers, i)), out_type=tf.int64)[0] for i in range(0, 9)], 0)

def __increment_day(fish_per_counter: tf.Tensor) -> tf.Tensor:
    return tf.concat([fish_per_counter[1:7], fish_per_counter[0:1] + fish_per_counter[7:8], fish_per_counter[8:9], fish_per_counter[0:1]], 0)

def __number_of_fish_after_days(text: tf.Tensor, days: tf.Tensor) -> tf.Tensor:
    fish_per_counter = tfutils.reduce_fn(
        lambda current, _: __increment_day(current),
        tf.range(0, days),
        __parse(text),
    )
    return tf.math.reduce_sum(fish_per_counter, 0)

assert __number_of_fish_after_days(EXAMPLE_INPUT, 18) == tf.constant(26, dtype=tf.int64)
assert __number_of_fish_after_days(EXAMPLE_INPUT, 80) == tf.constant(5934, dtype=tf.int64)
assert __number_of_fish_after_days(EXAMPLE_INPUT, 256) == tf.constant(26984457539)

assert __number_of_fish_after_days(PUZZLE_INPUT, 80) == tf.constant(386536, dtype=tf.int64)
print("Part 1:", __number_of_fish_after_days(PUZZLE_INPUT, 80).numpy())

assert __number_of_fish_after_days(PUZZLE_INPUT, 256) == tf.constant(1732821262171)
print("Part 2:", __number_of_fish_after_days(PUZZLE_INPUT, 256).numpy())
