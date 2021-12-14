EXAMPLE_INPUT = """
NNCB

CH -> B
HH -> N
CB -> H
NH -> C
HB -> C
HC -> B
HN -> C
NN -> C
BH -> H
NC -> B
NB -> B
BN -> B
BB -> N
BC -> B
CC -> N
CN -> C
"""

PUZZLE_INPUT = """
NNSOFOCNHBVVNOBSBHCB

HN -> S
FK -> N
CH -> P
VP -> P
VV -> C
PB -> H
CP -> F
KO -> P
KN -> V
NO -> K
NF -> N
CO -> P
HO -> H
VH -> V
OV -> C
VS -> F
PK -> H
OS -> S
BF -> S
SN -> P
NK -> N
SV -> O
KB -> O
ON -> O
FN -> H
FO -> N
KV -> S
CS -> C
VO -> O
SP -> O
VK -> H
KP -> S
SK -> N
NC -> B
PN -> N
HV -> O
HS -> C
CN -> N
OO -> V
FF -> B
VC -> V
HK -> K
CC -> H
BO -> H
SC -> O
HH -> C
BV -> P
OB -> O
FC -> H
PO -> C
FV -> C
BK -> F
HB -> B
NH -> P
KF -> N
BP -> H
KK -> O
OH -> K
CB -> H
CK -> C
OK -> H
NN -> F
VF -> N
SO -> K
OP -> F
NP -> B
FS -> S
SH -> O
FP -> O
SF -> V
HF -> N
KC -> K
SB -> V
FH -> N
SS -> C
BB -> C
NV -> K
OC -> S
CV -> N
HC -> P
BC -> N
OF -> K
BH -> N
NS -> K
BN -> F
PC -> C
CF -> N
HP -> F
BS -> O
PF -> S
PV -> B
KH -> K
VN -> V
NB -> N
PH -> V
KS -> B
PP -> V
PS -> C
VB -> N
FB -> N
"""

import tensorflow as tf
from typing import Tuple

def __parse(text: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    text = tf.strings.strip(text)
    sections = tf.strings.split(text, "\n\n")
    template_str = tf.strings.bytes_split(sections[0])
    template_pairs = tf.map_fn(
        lambda idx: tf.strings.join([template_str[idx - 1], template_str[idx]]),
        tf.range(1, tf.shape(template_str, out_type=tf.int64)[0]),
        dtype=tf.string,
    )
    lines = tf.strings.split(sections[1], "\n")
    mappings_str = tf.strings.split(lines, " -> ").to_tensor()
    chars, _ = tf.unique(tf.reshape(mappings_str[:, 1], [-1]))
    mappings = tf.map_fn(
        lambda idx: tf.stack([
            tf.where(mappings_str[:, 0] == tf.strings.join([tf.strings.bytes_split(mappings_str[idx][0])[0], mappings_str[idx][1]]))[0][0],
            tf.where(mappings_str[:, 0] == tf.strings.join([mappings_str[idx][1], tf.strings.bytes_split(mappings_str[idx][0])[1]]))[0][0],
        ]),
        tf.range(0, tf.shape(mappings_str, out_type=tf.int64)[0]),
    )
    values = tf.map_fn(
        lambda idx: tf.where(chars == tf.strings.bytes_split(mappings_str[idx][0])[0])[0][0],
        tf.range(0, tf.shape(mappings_str, out_type=tf.int64)[0]),
    )
    template = tf.map_fn(
        lambda idx: tf.shape(tf.where(mappings_str[idx][0] == template_pairs), out_type=tf.int64)[0],
        tf.range(0, tf.shape(mappings_str, out_type=tf.int64)[0]),
    )
    last_value = tf.where(chars == template_str[-1])[0][0]
    return template, mappings, values, last_value

@tf.function
def __get_variance_after_steps(text: tf.Tensor, steps: tf.Tensor) -> tf.Tensor:
    template, mappings, values, last_value = __parse(text)
    template = tf.while_loop(
        lambda i, _: i < steps,
        lambda i, template: (
            i + 1,
            tf.map_fn(
                lambda idx: tf.math.reduce_sum(tf.map_fn(
                    lambda idx2: template[idx2],
                    tf.reshape(
                        tf.concat([
                            tf.where(mappings[:, 0] == idx),
                            tf.where(mappings[:, 1] == idx),
                        ], 0),
                        [-1],
                    ),
                )),
                tf.range(0, tf.shape(template, out_type=tf.int64)[0]),
            ),
        ),
        [tf.constant(0, dtype=tf.int64), template],
    )[1]
    occurrence = tf.map_fn(
        lambda value: tf.math.reduce_sum(
            tf.map_fn(
                lambda idx: template[idx],
                tf.reshape(tf.where(values == value), [-1]),
            ),
        ) + tf.cond(value == last_value, lambda: tf.constant(1, dtype=tf.int64), lambda: tf.constant(0, dtype=tf.int64)),
        values,
    )
    occurrence = tf.sort(occurrence)
    return occurrence[-1] - occurrence[0]

assert __get_variance_after_steps(EXAMPLE_INPUT, 10) == 1588
assert __get_variance_after_steps(PUZZLE_INPUT, 10) == 3906
print("Part 1:", __get_variance_after_steps(PUZZLE_INPUT, 10).numpy())

assert __get_variance_after_steps(EXAMPLE_INPUT, 40) == 2188189693529
assert __get_variance_after_steps(PUZZLE_INPUT, 40) == 4441317262452
print("Part 2:", __get_variance_after_steps(PUZZLE_INPUT, 40).numpy())
