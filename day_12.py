EXAMPLE_INPUT = """
start-A
start-b
A-c
A-b
b-d
A-end
b-end
"""

PUZZLE_INPUT = """
pf-pk
ZQ-iz
iz-NY
ZQ-end
pf-gx
pk-ZQ
ZQ-dc
NY-start
NY-pf
NY-gx
ag-ZQ
pf-start
start-gx
BN-ag
iz-pf
ag-FD
pk-NY
gx-pk
end-BN
ag-pf
iz-pk
pk-ag
iz-end
iz-BN
"""

import tensorflow as tf

def __parse(text: tf.Tensor) -> tf.Tensor:
    connection_strs = tf.strings.split(tf.strings.split(tf.strings.strip(text), "\n"), "-").to_tensor()
    caves, _ = tf.unique(tf.reshape(connection_strs, [-1]))
    caves = tf.concat([
        tf.map_fn(
            lambda idx: caves[idx],
            tf.reshape(tf.where(caves != "end"), [-1]),
            dtype=tf.string,
        ),
        tf.constant(["end"]),
    ], 0)
    caves = tf.concat([
        tf.constant(["start"]),
        tf.map_fn(
            lambda idx: caves[idx],
            tf.reshape(tf.where(caves != "start"), [-1]),
            dtype=tf.string,
        ),
    ], 0)
    connections = tf.map_fn(
        lambda c1: tf.map_fn(
            lambda c2: tf.reduce_any(
                tf.map_fn(
                    lambda s: (s[0] == caves[c1] and s[1] == caves[c2]) or (s[0] == caves[c2] and s[1] == caves[c1]),
                    connection_strs,
                    dtype=tf.bool,
                ),
            ),
            tf.range(0, tf.shape(caves, out_type=tf.int64)[0]),
            dtype=tf.bool,
        ),
        tf.range(0, tf.shape(caves, out_type=tf.int64)[0]),
        dtype=tf.bool,
    )
    big_caves = caves == tf.strings.upper(caves)
    return caves, connections, big_caves

@tf.function
def __count_distinct_paths(text: tf.Tensor, max_visit_count: tf.Tensor) -> tf.Tensor:
    caves, connections, big_caves = __parse(text)
    start = tf.constant(0, dtype=tf.int64)
    end = tf.shape(caves, out_type=tf.int64)[0] - 1
    def traverse(path, stack, small_cave_explored_twice, paths_found):
        def next_connection():
            return path, tf.concat([stack[:-1], tf.stack([stack[-1] + 1])], 0), small_cave_explored_twice, paths_found
        def drop():
            return path[:-1], stack[:-1], small_cave_explored_twice[:-1], paths_found
        def add():
            path, stack, small_cave_explored_twice, paths_found = next_connection()
            return path, stack, small_cave_explored_twice, paths_found + 1
        def deeper(visit_count):
            return (
                tf.concat([path, stack[-1:]], 0),
                tf.concat([stack[:-1], stack[-1:] + 1, tf.constant([1], dtype=tf.int64)], 0),
                tf.concat([small_cave_explored_twice, tf.stack([small_cave_explored_twice[-1] or visit_count == 1])], 0),
                paths_found,
            )
        def check_visit_count():
            visit_count = tf.cond(
                big_caves[stack[-1]],
                lambda: tf.constant(0, dtype=tf.int64),
                lambda: tf.shape(tf.where(path == stack[-1]), out_type=tf.int64)[0],
            )
            return tf.cond(
                visit_count == max_visit_count or (small_cave_explored_twice[-1] and visit_count == 1),
                next_connection,
                lambda: deeper(visit_count),
            )
        def check_end():
            return tf.cond(stack[-1] == end, add, check_visit_count)
        def check_valid():
            return tf.cond(not connections[path[-1]][stack[-1]] or stack[-1] == start, next_connection, check_end)
        return tf.cond(stack[-1] > end, drop, check_valid)
    path, stack, small_cave_explored_twice, paths_found = tf.while_loop(
        lambda path, stack, small_cave_explored_twice, paths_found: tf.shape(path) > 0,
        traverse,
        [tf.stack([start]), tf.constant([1], dtype=tf.int64), tf.constant([False]), tf.constant(0, dtype=tf.int64)],
        shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])],
    )
    return paths_found

assert __count_distinct_paths(EXAMPLE_INPUT, 1) == 10
assert __count_distinct_paths(PUZZLE_INPUT, 1) == 5212
print("Part 1:", __count_distinct_paths(PUZZLE_INPUT, 1).numpy())

assert __count_distinct_paths(EXAMPLE_INPUT, 2) == 36
assert __count_distinct_paths(PUZZLE_INPUT, 2) == 134862
print("Part 2:", __count_distinct_paths(PUZZLE_INPUT, 2).numpy())
