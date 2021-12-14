EXAMPLE_INPUT = """
5483143223
2745854711
5264556173
6141336146
6357385478
4167524645
2176841721
6882881134
4846848554
5283751526
"""

PUZZLE_INPUT = """
3322874652
5636588857
7755117548
5854121833
2856682477
3124873812
1541372254
8634383236
2424323348
2265635842
"""

import tensorflow as tf

def __parse(text: tf.Tensor) -> tf.Tensor:
    return tf.strings.to_number(
        tf.strings.bytes_split(tf.strings.split(tf.strings.strip(text), "\n")),
        out_type=tf.int64,
    ).to_tensor()

def __flash(m: tf.Tensor) -> tf.Tensor:
    i64 = tf.constant(0, dtype=tf.int64)
    w = tf.shape(m, out_type=tf.int64)[1]
    h = tf.shape(m, out_type=tf.int64)[0]
    m += 1
    def adjacent_flashing(flashing, x, y):
        return tf.cond(
            x >= i64 and x < w and y >= i64 and y < h,
            lambda: tf.where(flashing[y][x], i64 + 1, i64),
            lambda: i64,
        )
    def next_flash(flashing, flashed, m):
        flashed = flashed | flashing
        m = tf.map_fn(
            lambda y: tf.map_fn(
                lambda x: m[y][x] + tf.reduce_sum(tf.stack([
                    adjacent_flashing(flashing, x + dx, y + dy)
                    for dx, dy in [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
                ])),
                tf.range(i64, w),
            ),
            tf.range(i64, h),
        )
        m = tf.where(flashed, tf.fill(tf.shape(m), i64), m)
        flashing = m > 9
        return flashing, flashed, m
    flashing, flashed, m = tf.while_loop(
        lambda flashing, flashed, m: tf.reduce_any(flashing),
        next_flash,
        [m > 9, tf.fill(tf.shape(m), False), m],
    )
    return m, flashed

@tf.function
def __flashes_after_100_steps(text: tf.Tensor):
    m = __parse(text)
    def step(i, m, s):
        i += 1
        m, flashed = __flash(m)
        s += tf.shape(tf.where(flashed), out_type=tf.int64)[0]
        return i, m, s
    _, m, s = tf.while_loop(
        lambda i, m, s: i < 100,
        step,
        [0, m, tf.constant(0, dtype=tf.int64)],
    )
    return s

@tf.function
def __first_simultaneous_flash(text: tf.Tensor):
    m = __parse(text)
    def step(i, m):
        i += 1
        m, _ = __flash(m)
        return i, m
    s, _ = tf.while_loop(
        lambda i, m: not tf.reduce_all(m == 0),
        step,
        [0, m],
    )
    return s

assert __flashes_after_100_steps(EXAMPLE_INPUT) == 1656
assert __flashes_after_100_steps(PUZZLE_INPUT) == 1613
print("Part 1:", __flashes_after_100_steps(PUZZLE_INPUT).numpy())

assert __first_simultaneous_flash(EXAMPLE_INPUT) == 195
assert __first_simultaneous_flash(PUZZLE_INPUT) == 510
print("Part 2:", __first_simultaneous_flash(PUZZLE_INPUT).numpy())
