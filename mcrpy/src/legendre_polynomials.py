import tensorflow as tf

ZERO = tf.constant(0.0, dtype=tf.float64)
ONE = tf.constant(1.0, dtype=tf.float64)


@tf.function
def l_0_0(x: tf.Tensor) -> tf.Tensor:
    return ONE


@tf.function
def l_1_0(x: tf.Tensor) -> tf.Tensor:
    return x


@tf.function
def l_1_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2))


@tf.function
def l_2_0(x: tf.Tensor) -> tf.Tensor:
    return (3 * x**2) / 2 - 1 / 2


@tf.function
def l_2_1(x: tf.Tensor) -> tf.Tensor:
    return -3 * x * (1 - x**2) ** (1 / 2)


@tf.function
def l_2_2(x: tf.Tensor) -> tf.Tensor:
    return 3 - 3 * x**2


@tf.function
def l_3_0(x: tf.Tensor) -> tf.Tensor:
    return (5 * x**3) / 2 - (3 * x) / 2


@tf.function
def l_3_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2)) * ((15 * x**2) / 2 - 3 / 2)


@tf.function
def l_3_2(x: tf.Tensor) -> tf.Tensor:
    return -15 * x * (x**2 - 1)


@tf.function
def l_3_3(x: tf.Tensor) -> tf.Tensor:
    return -15 * (1 - x**2) ** (3 / 2)


@tf.function
def l_4_0(x: tf.Tensor) -> tf.Tensor:
    return (35 * x**4) / 8 - (15 * x**2) / 4 + 3 / 8


@tf.function
def l_4_1(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (1 / 2) * ((15 * x) / 2 - (35 * x**3) / 2)


@tf.function
def l_4_2(x: tf.Tensor) -> tf.Tensor:
    return -(x**2 - 1) * ((105 * x**2) / 2 - 15 / 2)


@tf.function
def l_4_3(x: tf.Tensor) -> tf.Tensor:
    return -105 * x * (1 - x**2) ** (3 / 2)


@tf.function
def l_4_4(x: tf.Tensor) -> tf.Tensor:
    return 105 * (x**2 - 1) ** 2


@tf.function
def l_5_0(x: tf.Tensor) -> tf.Tensor:
    return (63 * x**5) / 8 - (35 * x**3) / 4 + (15 * x) / 8


@tf.function
def l_5_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2)) * ((315 * x**4) / 8 - (105 * x**2) / 4 + 15 / 8)


@tf.function
def l_5_2(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) * ((105 * x) / 2 - (315 * x**3) / 2)


@tf.function
def l_5_3(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (3 / 2)) * ((945 * x**2) / 2 - 105 / 2)


@tf.function
def l_5_4(x: tf.Tensor) -> tf.Tensor:
    return 945 * x * (x**2 - 1) ** 2


@tf.function
def l_5_5(x: tf.Tensor) -> tf.Tensor:
    return -945 * (1 - x**2) ** (5 / 2)


@tf.function
def l_6_0(x: tf.Tensor) -> tf.Tensor:
    return (231 * x**6) / 16 - (315 * x**4) / 16 + (105 * x**2) / 16 - 5 / 16


@tf.function
def l_6_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2)) * ((693 * x**5) / 8 - (315 * x**3) / 4 + (105 * x) / 8)


@tf.function
def l_6_2(x: tf.Tensor) -> tf.Tensor:
    return -(x**2 - 1) * ((3465 * x**4) / 8 - (945 * x**2) / 4 + 105 / 8)


@tf.function
def l_6_3(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (3 / 2) * ((945 * x) / 2 - (3465 * x**3) / 2)


@tf.function
def l_6_4(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 2 * ((10395 * x**2) / 2 - 945 / 2)


@tf.function
def l_6_5(x: tf.Tensor) -> tf.Tensor:
    return -10395 * x * (1 - x**2) ** (5 / 2)


@tf.function
def l_6_6(x: tf.Tensor) -> tf.Tensor:
    return -10395 * (x**2 - 1) ** 3


@tf.function
def l_7_0(x: tf.Tensor) -> tf.Tensor:
    return (429 * x**7) / 16 - (693 * x**5) / 16 + (315 * x**3) / 16 - (35 * x) / 16


@tf.function
def l_7_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2)) * ((3003 * x**6) / 16 - (3465 * x**4) / 16 + (945 * x**2) / 16 - 35 / 16)


@tf.function
def l_7_2(x: tf.Tensor) -> tf.Tensor:
    return -(x**2 - 1) * ((9009 * x**5) / 8 - (3465 * x**3) / 4 + (945 * x) / 8)


@tf.function
def l_7_3(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (3 / 2)) * ((45045 * x**4) / 8 - (10395 * x**2) / 4 + 945 / 8)


@tf.function
def l_7_4(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 2) * ((10395 * x) / 2 - (45045 * x**3) / 2)


@tf.function
def l_7_5(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (5 / 2)) * ((135135 * x**2) / 2 - 10395 / 2)


@tf.function
def l_7_6(x: tf.Tensor) -> tf.Tensor:
    return -135135 * x * (x**2 - 1) ** 3


@tf.function
def l_7_7(x: tf.Tensor) -> tf.Tensor:
    return -135135 * (1 - x**2) ** (7 / 2)


@tf.function
def l_8_0(x: tf.Tensor) -> tf.Tensor:
    return (6435 * x**8) / 128 - (3003 * x**6) / 32 + (3465 * x**4) / 64 - (315 * x**2) / 32 + 35 / 128


@tf.function
def l_8_1(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (1 / 2) * (-(6435 * x**7) / 16 + (9009 * x**5) / 16 - (3465 * x**3) / 16 + (315 * x) / 16)


@tf.function
def l_8_2(x: tf.Tensor) -> tf.Tensor:
    return -(x**2 - 1) * ((45045 * x**6) / 16 - (45045 * x**4) / 16 + (10395 * x**2) / 16 - 315 / 16)


@tf.function
def l_8_3(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (3 / 2)) * ((135135 * x**5) / 8 - (45045 * x**3) / 4 + (10395 * x) / 8)


@tf.function
def l_8_4(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 2 * ((675675 * x**4) / 8 - (135135 * x**2) / 4 + 10395 / 8)


@tf.function
def l_8_5(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (5 / 2) * ((135135 * x) / 2 - (675675 * x**3) / 2)


@tf.function
def l_8_6(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 3) * ((2027025 * x**2) / 2 - 135135 / 2)


@tf.function
def l_8_7(x: tf.Tensor) -> tf.Tensor:
    return -2027025 * x * (1 - x**2) ** (7 / 2)


@tf.function
def l_8_8(x: tf.Tensor) -> tf.Tensor:
    return 2027025 * (x**2 - 1) ** 4


@tf.function
def l_9_0(x: tf.Tensor) -> tf.Tensor:
    return (12155 * x**9) / 128 - (6435 * x**7) / 32 + (9009 * x**5) / 64 - (1155 * x**3) / 32 + (315 * x) / 128


@tf.function
def l_9_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2)) * (
        (109395 * x**8) / 128 - (45045 * x**6) / 32 + (45045 * x**4) / 64 - (3465 * x**2) / 32 + 315 / 128
    )


@tf.function
def l_9_2(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) * (-(109395 * x**7) / 16 + (135135 * x**5) / 16 - (45045 * x**3) / 16 + (3465 * x) / 16)


@tf.function
def l_9_3(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (3 / 2)) * ((765765 * x**6) / 16 - (675675 * x**4) / 16 + (135135 * x**2) / 16 - 3465 / 16)


@tf.function
def l_9_4(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 2 * ((2297295 * x**5) / 8 - (675675 * x**3) / 4 + (135135 * x) / 8)


@tf.function
def l_9_5(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (5 / 2)) * ((11486475 * x**4) / 8 - (2027025 * x**2) / 4 + 135135 / 8)


@tf.function
def l_9_6(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 3 * ((2027025 * x) / 2 - (11486475 * x**3) / 2)


@tf.function
def l_9_7(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (7 / 2)) * ((34459425 * x**2) / 2 - 2027025 / 2)


@tf.function
def l_9_8(x: tf.Tensor) -> tf.Tensor:
    return 34459425 * x * (x**2 - 1) ** 4


@tf.function
def l_9_9(x: tf.Tensor) -> tf.Tensor:
    return -34459425 * (1 - x**2) ** (9 / 2)


@tf.function
def l_10_0(x: tf.Tensor) -> tf.Tensor:
    return (
        (46189 * x**10) / 256
        - (109395 * x**8) / 256
        + (45045 * x**6) / 128
        - (15015 * x**4) / 128
        + (3465 * x**2) / 256
        - 63 / 256
    )


@tf.function
def l_10_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2)) * (
        (230945 * x**9) / 128 - (109395 * x**7) / 32 + (135135 * x**5) / 64 - (15015 * x**3) / 32 + (3465 * x) / 128
    )


@tf.function
def l_10_2(x: tf.Tensor) -> tf.Tensor:
    return -(x**2 - 1) * (
        (2078505 * x**8) / 128 - (765765 * x**6) / 32 + (675675 * x**4) / 64 - (45045 * x**2) / 32 + 3465 / 128
    )


@tf.function
def l_10_3(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (3 / 2) * (
        -(2078505 * x**7) / 16 + (2297295 * x**5) / 16 - (675675 * x**3) / 16 + (45045 * x) / 16
    )


@tf.function
def l_10_4(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 2 * ((14549535 * x**6) / 16 - (11486475 * x**4) / 16 + (2027025 * x**2) / 16 - 45045 / 16)


@tf.function
def l_10_5(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (5 / 2)) * ((43648605 * x**5) / 8 - (11486475 * x**3) / 4 + (2027025 * x) / 8)


@tf.function
def l_10_6(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 3) * ((218243025 * x**4) / 8 - (34459425 * x**2) / 4 + 2027025 / 8)


@tf.function
def l_10_7(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (7 / 2) * ((34459425 * x) / 2 - (218243025 * x**3) / 2)


@tf.function
def l_10_8(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 4 * ((654729075 * x**2) / 2 - 34459425 / 2)


@tf.function
def l_10_9(x: tf.Tensor) -> tf.Tensor:
    return -654729075 * x * (1 - x**2) ** (9 / 2)


@tf.function
def l_10_10(x: tf.Tensor) -> tf.Tensor:
    return -654729075 * (x**2 - 1) ** 5


@tf.function
def l_11_0(x: tf.Tensor) -> tf.Tensor:
    return (
        (88179 * x**11) / 256
        - (230945 * x**9) / 256
        + (109395 * x**7) / 128
        - (45045 * x**5) / 128
        + (15015 * x**3) / 256
        - (693 * x) / 256
    )


@tf.function
def l_11_1(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (1 / 2)) * (
        (969969 * x**10) / 256
        - (2078505 * x**8) / 256
        + (765765 * x**6) / 128
        - (225225 * x**4) / 128
        + (45045 * x**2) / 256
        - 693 / 256
    )


@tf.function
def l_11_2(x: tf.Tensor) -> tf.Tensor:
    return -(x**2 - 1) * (
        (4849845 * x**9) / 128
        - (2078505 * x**7) / 32
        + (2297295 * x**5) / 64
        - (225225 * x**3) / 32
        + (45045 * x) / 128
    )


@tf.function
def l_11_3(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (3 / 2)) * (
        (43648605 * x**8) / 128 - (14549535 * x**6) / 32 + (11486475 * x**4) / 64 - (675675 * x**2) / 32 + 45045 / 128
    )


@tf.function
def l_11_4(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 2) * (
        -(43648605 * x**7) / 16 + (43648605 * x**5) / 16 - (11486475 * x**3) / 16 + (675675 * x) / 16
    )


@tf.function
def l_11_5(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (5 / 2)) * (
        (305540235 * x**6) / 16 - (218243025 * x**4) / 16 + (34459425 * x**2) / 16 - 675675 / 16
    )


@tf.function
def l_11_6(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 3) * ((916620705 * x**5) / 8 - (218243025 * x**3) / 4 + (34459425 * x) / 8)


@tf.function
def l_11_7(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (7 / 2)) * ((4583103525 * x**4) / 8 - (654729075 * x**2) / 4 + 34459425 / 8)


@tf.function
def l_11_8(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 4) * ((654729075 * x) / 2 - (4583103525 * x**3) / 2)


@tf.function
def l_11_9(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (9 / 2)) * ((13749310575 * x**2) / 2 - 654729075 / 2)


@tf.function
def l_11_10(x: tf.Tensor) -> tf.Tensor:
    return -13749310575 * x * (x**2 - 1) ** 5


@tf.function
def l_11_11(x: tf.Tensor) -> tf.Tensor:
    return -13749310575 * (1 - x**2) ** (11 / 2)


@tf.function
def l_12_0(x: tf.Tensor) -> tf.Tensor:
    return (
        (676039 * x**12) / 1024
        - (969969 * x**10) / 512
        + (2078505 * x**8) / 1024
        - (255255 * x**6) / 256
        + (225225 * x**4) / 1024
        - (9009 * x**2) / 512
        + 231 / 1024
    )


@tf.function
def l_12_1(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (1 / 2) * (
        -(2028117 * x**11) / 256
        + (4849845 * x**9) / 256
        - (2078505 * x**7) / 128
        + (765765 * x**5) / 128
        - (225225 * x**3) / 256
        + (9009 * x) / 256
    )


@tf.function
def l_12_2(x: tf.Tensor) -> tf.Tensor:
    return -(x**2 - 1) * (
        (22309287 * x**10) / 256
        - (43648605 * x**8) / 256
        + (14549535 * x**6) / 128
        - (3828825 * x**4) / 128
        + (675675 * x**2) / 256
        - 9009 / 256
    )


@tf.function
def l_12_3(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (3 / 2)) * (
        (111546435 * x**9) / 128
        - (43648605 * x**7) / 32
        + (43648605 * x**5) / 64
        - (3828825 * x**3) / 32
        + (675675 * x) / 128
    )


@tf.function
def l_12_4(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 2 * (
        (1003917915 * x**8) / 128
        - (305540235 * x**6) / 32
        + (218243025 * x**4) / 64
        - (11486475 * x**2) / 32
        + 675675 / 128
    )


@tf.function
def l_12_5(x: tf.Tensor) -> tf.Tensor:
    return (1 - x**2) ** (5 / 2) * (
        -(1003917915 * x**7) / 16 + (916620705 * x**5) / 16 - (218243025 * x**3) / 16 + (11486475 * x) / 16
    )


@tf.function
def l_12_6(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 3) * (
        (7027425405 * x**6) / 16 - (4583103525 * x**4) / 16 + (654729075 * x**2) / 16 - 11486475 / 16
    )


@tf.function
def l_12_7(x: tf.Tensor) -> tf.Tensor:
    return -((1 - x**2) ** (7 / 2)) * ((21082276215 * x**5) / 8 - (4583103525 * x**3) / 4 + (654729075 * x) / 8)


@tf.function
def l_12_8(x: tf.Tensor) -> tf.Tensor:
    return (x**2 - 1) ** 4 * ((105411381075 * x**4) / 8 - (13749310575 * x**2) / 4 + 654729075 / 8)


@tf.function
def l_12_9(x: tf.Tensor) -> tf.Tensor:
    return ((13749310575 * x) / 2 - (105411381075 * x**3) / 2) * (1 - x**2) ** (9 / 2)


@tf.function
def l_12_10(x: tf.Tensor) -> tf.Tensor:
    return -((x**2 - 1) ** 5) * ((316234143225 * x**2) / 2 - 13749310575 / 2)


@tf.function
def l_12_11(x: tf.Tensor) -> tf.Tensor:
    return -316234143225 * x * (1 - x**2) ** (11 / 2)


@tf.function
def l_12_12(x: tf.Tensor) -> tf.Tensor:
    return 316234143225 * (x**2 - 1) ** 6


luts = [
    [l_0_0],
    [l_1_0, l_1_1],
    [
        l_2_0,
        l_2_1,
        l_2_2,
    ],
    [
        l_3_0,
        l_3_1,
        l_3_2,
        l_3_3,
    ],
    [
        l_4_0,
        l_4_1,
        l_4_2,
        l_4_3,
        l_4_4,
    ],
    [
        l_5_0,
        l_5_1,
        l_5_2,
        l_5_3,
        l_5_4,
        l_5_5,
    ],
    [
        l_6_0,
        l_6_1,
        l_6_2,
        l_6_3,
        l_6_4,
        l_6_5,
        l_6_6,
    ],
    [
        l_7_0,
        l_7_1,
        l_7_2,
        l_7_3,
        l_7_4,
        l_7_5,
        l_7_6,
        l_7_7,
    ],
    [
        l_8_0,
        l_8_1,
        l_8_2,
        l_8_3,
        l_8_4,
        l_8_5,
        l_8_6,
        l_8_7,
        l_8_8,
    ],
    [
        l_9_0,
        l_9_1,
        l_9_2,
        l_9_3,
        l_9_4,
        l_9_5,
        l_9_6,
        l_9_7,
        l_9_8,
        l_9_9,
    ],
    [
        l_10_0,
        l_10_1,
        l_10_2,
        l_10_3,
        l_10_4,
        l_10_5,
        l_10_6,
        l_10_7,
        l_10_8,
        l_10_9,
        l_10_10,
    ],
    [
        l_11_0,
        l_11_1,
        l_11_2,
        l_11_3,
        l_11_4,
        l_11_5,
        l_11_6,
        l_11_7,
        l_11_8,
        l_11_9,
        l_11_10,
        l_11_11,
    ],
    [
        l_12_0,
        l_12_1,
        l_12_2,
        l_12_3,
        l_12_4,
        l_12_5,
        l_12_6,
        l_12_7,
        l_12_8,
        l_12_9,
        l_12_10,
        l_12_11,
        l_12_12,
    ],
]


def legendre_lut(x: tf.Tensor, m: int, l: int) -> tf.Tensor:
    return luts[l][m](x)
