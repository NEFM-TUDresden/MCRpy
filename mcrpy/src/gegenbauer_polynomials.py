import tensorflow as tf

ZERO = tf.constant(0.0, dtype=tf.float64)
ONE = tf.constant(1.0, dtype=tf.float64)

coeffs = [
    [
        [1],
        [0],
    ],
    [
        [1],
        [2, 0],
        [4, 0, -1],
        [8, 0, -4, 0],
        [16, 0, -12, 0, 1],
        [32, 0, -32, 0, 6, 0],
        [64, 0, -80, 0, 24, 0, -1],
        [128, 0, -192, 0, 80, 0, -8, 0],
        [256, 0, -448, 0, 240, 0, -40, 0, 1],
        [512, 0, -1024, 0, 672, 0, -160, 0, 10, 0],
        [1024, 0, -2304, 0, 1792, 0, -560, 0, 60, 0, -1],
        [2048, 0, -5120, 0, 4608, 0, -1792, 0, 280, 0, -12, 0],
        [4096, 0, -11264, 0, 11520, 0, -5376, 0, 1120, 0, -84, 0, 1],
    ],
    [
        [1],
        [4, 0],
        [12, 0, -2],
        [32, 0, -12, 0],
        [80, 0, -48, 0, 3],
        [192, 0, -160, 0, 24, 0],
        [448, 0, -480, 0, 120, 0, -4],
        [1024, 0, -1344, 0, 480, 0, -40, 0],
        [2304, 0, -3584, 0, 1680, 0, -240, 0, 5],
        [5120, 0, -9216, 0, 5376, 0, -1120, 0, 60, 0],
        [11264, 0, -23040, 0, 16128, 0, -4480, 0, 420, 0, -6],
        [24576, 0, -56320, 0, 46080, 0, -16128, 0, 2240, 0, -84, 0],
        [53248, 0, -135168, 0, 126720, 0, -53760, 0, 10080, 0, -672, 0, 7],
    ],
    [
        [1],
        [6, 0],
        [24, 0, -3],
        [80, 0, -24, 0],
        [240, 0, -120, 0, 6],
        [672, 0, -480, 0, 60, 0],
        [1792, 0, -1680, 0, 360, 0, -10],
        [4608, 0, -5376, 0, 1680, 0, -120, 0],
        [11520, 0, -16128, 0, 6720, 0, -840, 0, 15],
        [28160, 0, -46080, 0, 24192, 0, -4480, 0, 210, 0],
        [67584, 0, -126720, 0, 80640, 0, -20160, 0, 1680, 0, -21],
        [159744, 0, -337920, 0, 253440, 0, -80640, 0, 10080, 0, -336, 0],
        [372736, 0, -878592, 0, 760320, 0, -295680, 0, 50400, 0, -3024, 0, 28],
    ],
    [
        [1],
        [8, 0],
        [40, 0, -4],
        [160, 0, -40, 0],
        [560, 0, -240, 0, 10],
        [1792, 0, -1120, 0, 120, 0],
        [5376, 0, -4480, 0, 840, 0, -20],
        [15360, 0, -16128, 0, 4480, 0, -280, 0],
        [42240, 0, -53760, 0, 20160, 0, -2240, 0, 35],
        [112640, 0, -168960, 0, 80640, 0, -13440, 0, 560, 0],
        [292864, 0, -506880, 0, 295680, 0, -67200, 0, 5040, 0, -56],
        [745472, 0, -1464320, 0, 1013760, 0, -295680, 0, 33600, 0, -1008, 0],
        [1863680, 0, -4100096, 0, 3294720, 0, -1182720, 0, 184800, 0, -10080, 0, 84],
    ],
    [
        [1],
        [10, 0],
        [60, 0, -5],
        [280, 0, -60, 0],
        [1120, 0, -420, 0, 15],
        [4032, 0, -2240, 0, 210, 0],
        [13440, 0, -10080, 0, 1680, 0, -35],
        [42240, 0, -40320, 0, 10080, 0, -560, 0],
        [126720, 0, -147840, 0, 50400, 0, -5040, 0, 70],
        [366080, 0, -506880, 0, 221760, 0, -33600, 0, 1260, 0],
        [1025024, 0, -1647360, 0, 887040, 0, -184800, 0, 12600, 0, -126],
        [2795520, 0, -5125120, 0, 3294720, 0, -887040, 0, 92400, 0, -2520, 0],
        [7454720, 0, -15375360, 0, 11531520, 0, -3843840, 0, 554400, 0, -27720, 0, 210],
    ],
    [
        [1],
        [12, 0],
        [84, 0, -6],
        [448, 0, -84, 0],
        [2016, 0, -672, 0, 21],
        [8064, 0, -4032, 0, 336, 0],
        [29568, 0, -20160, 0, 3024, 0, -56],
        [101376, 0, -88704, 0, 20160, 0, -1008, 0],
        [329472, 0, -354816, 0, 110880, 0, -10080, 0, 126],
        [1025024, 0, -1317888, 0, 532224, 0, -73920, 0, 2520, 0],
        [3075072, 0, -4612608, 0, 2306304, 0, -443520, 0, 27720, 0, -252],
        [8945664, 0, -15375360, 0, 9225216, 0, -2306304, 0, 221760, 0, -5544, 0],
        [25346048, 0, -49201152, 0, 34594560, 0, -10762752, 0, 1441440, 0, -66528, 0, 462],
    ],
    [
        [1],
        [14, 0],
        [112, 0, -7],
        [672, 0, -112, 0],
        [3360, 0, -1008, 0, 28],
        [14784, 0, -6720, 0, 504, 0],
        [59136, 0, -36960, 0, 5040, 0, -84],
        [219648, 0, -177408, 0, 36960, 0, -1680, 0],
        [768768, 0, -768768, 0, 221760, 0, -18480, 0, 210],
        [2562560, 0, -3075072, 0, 1153152, 0, -147840, 0, 4620, 0],
        [8200192, 0, -11531520, 0, 5381376, 0, -960960, 0, 55440, 0, -462],
        [25346048, 0, -41000960, 0, 23063040, 0, -5381376, 0, 480480, 0, -11088, 0],
        [76038144, 0, -139403264, 0, 92252160, 0, -26906880, 0, 3363360, 0, -144144, 0, 924],
    ],
    [
        [1],
        [16, 0],
        [144, 0, -8],
        [960, 0, -144, 0],
        [5280, 0, -1440, 0, 36],
        [25344, 0, -10560, 0, 720, 0],
        [109824, 0, -63360, 0, 7920, 0, -120],
        [439296, 0, -329472, 0, 63360, 0, -2640, 0],
        [1647360, 0, -1537536, 0, 411840, 0, -31680, 0, 330],
        [5857280, 0, -6589440, 0, 2306304, 0, -274560, 0, 7920, 0],
        [19914752, 0, -26357760, 0, 11531520, 0, -1921920, 0, 102960, 0, -792],
        [65175552, 0, -99573760, 0, 52715520, 0, -11531520, 0, 960960, 0, -20592, 0],
        [206389248, 0, -358465536, 0, 224040960, 0, -61501440, 0, 7207200, 0, -288288, 0, 1716],
    ],
    [
        [1],
        [18, 0],
        [180, 0, -9],
        [1320, 0, -180, 0],
        [7920, 0, -1980, 0, 45],
        [41184, 0, -15840, 0, 990, 0],
        [192192, 0, -102960, 0, 11880, 0, -165],
        [823680, 0, -576576, 0, 102960, 0, -3960, 0],
        [3294720, 0, -2882880, 0, 720720, 0, -51480, 0, 495],
        [12446720, 0, -13178880, 0, 4324320, 0, -480480, 0, 12870, 0],
        [44808192, 0, -56010240, 0, 23063040, 0, -3603600, 0, 180180, 0, -1287],
        [154791936, 0, -224040960, 0, 112020480, 0, -23063040, 0, 1801800, 0, -36036, 0],
        [515973120, 0, -851355648, 0, 504092160, 0, -130690560, 0, 14414400, 0, -540540, 0, 3003],
    ],
    [
        [1],
        [20, 0],
        [220, 0, -10],
        [1760, 0, -220, 0],
        [11440, 0, -2640, 0, 55],
        [64064, 0, -22880, 0, 1320, 0],
        [320320, 0, -160160, 0, 17160, 0, -220],
        [1464320, 0, -960960, 0, 160160, 0, -5720, 0],
        [6223360, 0, -5125120, 0, 1201200, 0, -80080, 0, 715],
        [24893440, 0, -24893440, 0, 7687680, 0, -800800, 0, 20020, 0],
        [94595072, 0, -112020480, 0, 43563520, 0, -6406400, 0, 300300, 0, -2002],
        [343982080, 0, -472975360, 0, 224040960, 0, -43563520, 0, 3203200, 0, -60060, 0],
        [1203937280, 0, -1891901440, 0, 1064194560, 0, -261381120, 0, 27227200, 0, -960960, 0, 5005],
    ],
    [
        [1],
        [22, 0],
        [264, 0, -11],
        [2288, 0, -264, 0],
        [16016, 0, -3432, 0, 66],
        [96096, 0, -32032, 0, 1716, 0],
        [512512, 0, -240240, 0, 24024, 0, -286],
        [2489344, 0, -1537536, 0, 240240, 0, -8008, 0],
        [11202048, 0, -8712704, 0, 1921920, 0, -120120, 0, 1001],
        [47297536, 0, -44808192, 0, 13069056, 0, -1281280, 0, 30030, 0],
        [189190144, 0, -212838912, 0, 78414336, 0, -10890880, 0, 480480, 0, -3003],
        [722362368, 0, -945950720, 0, 425677824, 0, -78414336, 0, 5445440, 0, -96096, 0],
        [2648662016, 0, -3972993024, 0, 2128389120, 0, -496624128, 0, 49008960, 0, -1633632, 0, 8008],
    ],
    [
        [1],
        [24, 0],
        [312, 0, -12],
        [2912, 0, -312, 0],
        [21840, 0, -4368, 0, 78],
        [139776, 0, -43680, 0, 2184, 0],
        [792064, 0, -349440, 0, 32760, 0, -364],
        [4073472, 0, -2376192, 0, 349440, 0, -10920, 0],
        [19348992, 0, -14257152, 0, 2970240, 0, -174720, 0, 1365],
        [85995520, 0, -77395968, 0, 21385728, 0, -1980160, 0, 43680, 0],
        [361181184, 0, -386979840, 0, 135442944, 0, -17821440, 0, 742560, 0, -4368],
        [1444724736, 0, -1805905920, 0, 773959680, 0, -135442944, 0, 8910720, 0, -148512, 0],
        [5538111488, 0, -7945986048, 0, 4063288320, 0, -902952960, 0, 84651840, 0, -2673216, 0, 12376],
    ],
    [
        [1],
        [26, 0],
        [364, 0, -13],
        [3640, 0, -364, 0],
        [29120, 0, -5460, 0, 91],
        [198016, 0, -58240, 0, 2730, 0],
        [1188096, 0, -495040, 0, 43680, 0, -455],
        [6449664, 0, -3564288, 0, 495040, 0, -14560, 0],
        [32248320, 0, -22573824, 0, 4455360, 0, -247520, 0, 1820],
        [150492160, 0, -128993280, 0, 33860736, 0, -2970240, 0, 61880, 0],
        [662165504, 0, -677214720, 0, 225738240, 0, -28217280, 0, 1113840, 0, -6188],
        [2769055744, 0, -3310827520, 0, 1354429440, 0, -225738240, 0, 14108640, 0, -222768, 0],
        [11076222976, 0, -15229806592, 0, 7449361920, 0, -1580167680, 0, 141086400, 0, -4232592, 0, 18564],
    ],
]
coeffs = [[[tf.constant(c, dtype=tf.float64) for c in coeffs_ll] for coeffs_ll in coeffs_l] for coeffs_l in coeffs]

def g_13_0(x: tf.Tensor):
    return 1


def g_13_1(x: tf.Tensor):
    return 26 * x


def g_13_2(x: tf.Tensor):
    return 364 * x**2 - 13


def g_13_3(x: tf.Tensor):
    return 3640 * x**3 - 364 * x


def g_13_4(x: tf.Tensor):
    return 29120 * x**4 - 5460 * x**2 + 91


def g_13_5(x: tf.Tensor):
    return 198016 * x**5 - 58240 * x**3 + 2730 * x


def g_13_6(x: tf.Tensor):
    return 1188096 * x**6 - 495040 * x**4 + 43680 * x**2 - 455


def g_13_7(x: tf.Tensor):
    return 6449664 * x**7 - 3564288 * x**5 + 495040 * x**3 - 14560 * x


def g_13_8(x: tf.Tensor):
    return 32248320 * x**8 - 22573824 * x**6 + 4455360 * x**4 - 247520 * x**2 + 1820


def g_13_9(x: tf.Tensor):
    return 150492160 * x**9 - 128993280 * x**7 + 33860736 * x**5 - 2970240 * x**3 + 61880 * x


def g_13_10(x: tf.Tensor):
    return 662165504 * x**10 - 677214720 * x**8 + 225738240 * x**6 - 28217280 * x**4 + 1113840 * x**2 - 6188


def g_13_11(x: tf.Tensor):
    return 2769055744 * x**11 - 3310827520 * x**9 + 1354429440 * x**7 - 225738240 * x**5 + 14108640 * x**3 - 222768 * x


def g_13_12(x: tf.Tensor):
    return (
        11076222976 * x**12
        - 15229806592 * x**10
        + 7449361920 * x**8
        - 1580167680 * x**6
        + 141086400 * x**4
        - 4232592 * x**2
        + 18564
    )


@tf.function
def g_0_0(x):
    return ONE


@tf.function
def g_0_1(x):
    return ZERO


@tf.function
def g_1_0(x: tf.Tensor):
    return ONE


@tf.function
def g_1_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(2.0, dtype=tf.float64), x)


# def g_1_1( x: tf.Tensor): return tf.math.polyval([2, 0], x)


@tf.function
def g_1_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(4.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 1


@tf.function
def g_1_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(8.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(4.0, dtype=tf.float64), x
    )


@tf.function
def g_1_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(16.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(12.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 1
    )


@tf.function
def g_1_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(32.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(32.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(6.0, dtype=tf.float64), x)
    )


@tf.function
def g_1_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(64.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(80.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(24.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 1
    )


@tf.function
def g_1_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(128.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(192.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(80.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(8.0, dtype=tf.float64), x)
    )


@tf.function
def g_1_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(256.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(448.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(240.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(40.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 1
    )


@tf.function
def g_1_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(512.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(1024.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(672.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(160.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(10.0, dtype=tf.float64), x)
    )


@tf.function
def g_1_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1024.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(2304.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(1792.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(560.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(60.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 1
    )


@tf.function
def g_1_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(2048.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(5120.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(4608.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(1792.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(280.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(12.0, dtype=tf.float64), x)
    )


@tf.function
def g_1_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(4096.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(11264.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(11520.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(5376.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(1120.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(84.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 1
    )


@tf.function
def g_2_0(x: tf.Tensor):
    return ONE


@tf.function
def g_2_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(4.0, dtype=tf.float64), x)


@tf.function
def g_2_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(12.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 2


@tf.function
def g_2_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(32.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(12.0, dtype=tf.float64), x
    )


@tf.function
def g_2_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(80.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(48.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 3
    )


@tf.function
def g_2_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(192.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(160.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(24.0, dtype=tf.float64), x)
    )


@tf.function
def g_2_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(448.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(480.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(120.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 4
    )


@tf.function
def g_2_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1024.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(1344.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(480.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(40.0, dtype=tf.float64), x)
    )


@tf.function
def g_2_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(2304.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(3584.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(1680.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(240.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 5
    )


@tf.function
def g_2_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(5120.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(9216.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(5376.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(1120.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(60.0, dtype=tf.float64), x)
    )


@tf.function
def g_2_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(11264.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(23040.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(16128.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(4480.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(420.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 6
    )


@tf.function
def g_2_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(24576.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(56320.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(46080.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(16128.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(2240.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(84.0, dtype=tf.float64), x)
    )


@tf.function
def g_2_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(53248.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(135168.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(126720.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(53760.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(10080.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(672.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 7
    )


@tf.function
def g_3_0(x: tf.Tensor):
    return ONE


@tf.function
def g_3_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(6.0, dtype=tf.float64), x)


@tf.function
def g_3_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(24.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 3


@tf.function
def g_3_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(80.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(24.0, dtype=tf.float64), x
    )


@tf.function
def g_3_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(240.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(120.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 6
    )


@tf.function
def g_3_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(672.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(480.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(60.0, dtype=tf.float64), x)
    )


@tf.function
def g_3_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1792.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(1680.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(360.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 10
    )


@tf.function
def g_3_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(4608.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(5376.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(1680.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(120.0, dtype=tf.float64), x)
    )


@tf.function
def g_3_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(11520.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(16128.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(6720.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(840.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 15
    )


@tf.function
def g_3_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(28160.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(46080.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(24192.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(4480.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(210.0, dtype=tf.float64), x)
    )


@tf.function
def g_3_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(67584.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(126720.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(80640.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(20160.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(1680.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 21
    )


@tf.function
def g_3_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(159744.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(337920.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(253440.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(80640.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(10080.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(336.0, dtype=tf.float64), x)
    )


@tf.function
def g_3_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(372736.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(878592.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(760320.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(295680.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(50400.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(3024.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 28
    )


@tf.function
def g_4_0(x: tf.Tensor):
    return ONE


@tf.function
def g_4_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(8.0, dtype=tf.float64), x)


@tf.function
def g_4_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(40.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 4


@tf.function
def g_4_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(160.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(40.0, dtype=tf.float64), x
    )


@tf.function
def g_4_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(560.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(240.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 10
    )


@tf.function
def g_4_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1792.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(1120.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(120.0, dtype=tf.float64), x)
    )


@tf.function
def g_4_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(5376.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(4480.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(840.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 20
    )


@tf.function
def g_4_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(15360.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(16128.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(4480.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(280.0, dtype=tf.float64), x)
    )


@tf.function
def g_4_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(42240.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(53760.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(20160.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(2240.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 35
    )


@tf.function
def g_4_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(112640.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(168960.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(80640.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(13440.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(560.0, dtype=tf.float64), x)
    )


@tf.function
def g_4_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(292864.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(506880.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(295680.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(67200.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(5040.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 56
    )


@tf.function
def g_4_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(745472.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(1464320.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(1013760.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(295680.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(33600.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(1008.0, dtype=tf.float64), x)
    )


@tf.function
def g_4_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1863680.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(4100096.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(3294720.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(1182720.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(184800.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(10080.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 84
    )


@tf.function
def g_5_0(x: tf.Tensor):
    return ONE


@tf.function
def g_5_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(10.0, dtype=tf.float64), x)


@tf.function
def g_5_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(60.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 5


@tf.function
def g_5_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(280.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(60.0, dtype=tf.float64), x
    )


@tf.function
def g_5_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1120.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(420.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 15
    )


@tf.function
def g_5_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(4032.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(2240.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(210.0, dtype=tf.float64), x)
    )


@tf.function
def g_5_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(13440.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(10080.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(1680.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 35
    )


@tf.function
def g_5_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(42240.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(40320.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(10080.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(560.0, dtype=tf.float64), x)
    )


@tf.function
def g_5_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(126720.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(147840.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(50400.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(5040.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 70
    )


@tf.function
def g_5_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(366080.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(506880.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(221760.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(33600.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(1260.0, dtype=tf.float64), x)
    )


@tf.function
def g_5_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1025024.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(1647360.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(887040.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(184800.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(12600.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 126
    )


@tf.function
def g_5_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(2795520.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(5125120.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(3294720.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(887040.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(92400.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(2520.0, dtype=tf.float64), x)
    )


@tf.function
def g_5_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(7454720.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(15375360.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(11531520.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(3843840.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(554400.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(27720.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 210
    )


@tf.function
def g_6_0(x: tf.Tensor):
    return ONE


@tf.function
def g_6_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(12.0, dtype=tf.float64), x)


@tf.function
def g_6_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(84.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 6


@tf.function
def g_6_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(448.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(84.0, dtype=tf.float64), x
    )


@tf.function
def g_6_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(2016.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(672.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 21
    )


@tf.function
def g_6_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(8064.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(4032.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(336.0, dtype=tf.float64), x)
    )


@tf.function
def g_6_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(29568.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(20160.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(3024.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 56
    )


@tf.function
def g_6_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(101376.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(88704.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(20160.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(1008.0, dtype=tf.float64), x)
    )


@tf.function
def g_6_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(329472.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(354816.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(110880.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(10080.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 126
    )


@tf.function
def g_6_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1025024.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(1317888.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(532224.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(73920.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(2520.0, dtype=tf.float64), x)
    )


@tf.function
def g_6_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(3075072.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(4612608.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(2306304.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(443520.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(27720.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 252
    )


@tf.function
def g_6_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(8945664.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(15375360.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(9225216.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(2306304.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(221760.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(5544.0, dtype=tf.float64), x)
    )


@tf.function
def g_6_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(25346048.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(49201152.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(34594560.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(10762752.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(1441440.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(66528.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 462
    )


@tf.function
def g_7_0(x: tf.Tensor):
    return ONE


@tf.function
def g_7_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(14.0, dtype=tf.float64), x)


@tf.function
def g_7_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(112.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 7


@tf.function
def g_7_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(672.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(112.0, dtype=tf.float64), x
    )


@tf.function
def g_7_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(3360.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(1008.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 28
    )


@tf.function
def g_7_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(14784.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(6720.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(504.0, dtype=tf.float64), x)
    )


@tf.function
def g_7_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(59136.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(36960.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(5040.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 84
    )


@tf.function
def g_7_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(219648.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(177408.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(36960.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(1680.0, dtype=tf.float64), x)
    )


@tf.function
def g_7_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(768768.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(768768.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(221760.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(18480.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 210
    )


@tf.function
def g_7_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(2562560.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(3075072.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(1153152.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(147840.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(4620.0, dtype=tf.float64), x)
    )


@tf.function
def g_7_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(8200192.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(11531520.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(5381376.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(960960.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(55440.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 462
    )


@tf.function
def g_7_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(25346048.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(41000960.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(23063040.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(5381376.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(480480.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(11088.0, dtype=tf.float64), x)
    )


@tf.function
def g_7_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(76038144.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(139403264.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(92252160.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(26906880.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(3363360.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(144144.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 924
    )


@tf.function
def g_8_0(x: tf.Tensor):
    return ONE


@tf.function
def g_8_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(16.0, dtype=tf.float64), x)


@tf.function
def g_8_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(144.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 8


@tf.function
def g_8_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(960.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(144.0, dtype=tf.float64), x
    )


@tf.function
def g_8_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(5280.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(1440.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 36
    )


@tf.function
def g_8_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(25344.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(10560.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(720.0, dtype=tf.float64), x)
    )


@tf.function
def g_8_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(109824.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(63360.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(7920.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 120
    )


@tf.function
def g_8_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(439296.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(329472.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(63360.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(2640.0, dtype=tf.float64), x)
    )


@tf.function
def g_8_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1647360.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(1537536.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(411840.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(31680.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 330
    )


@tf.function
def g_8_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(5857280.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(6589440.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(2306304.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(274560.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(7920.0, dtype=tf.float64), x)
    )


@tf.function
def g_8_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(19914752.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(26357760.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(11531520.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(1921920.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(102960.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 792
    )


@tf.function
def g_8_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(65175552.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(99573760.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(52715520.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(11531520.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(960960.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(20592.0, dtype=tf.float64), x)
    )


@tf.function
def g_8_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(206389248.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(358465536.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(224040960.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(61501440.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(7207200.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(288288.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 1716
    )


@tf.function
def g_9_0(x: tf.Tensor):
    return ONE


@tf.function
def g_9_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(18.0, dtype=tf.float64), x)


@tf.function
def g_9_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(180.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 9


@tf.function
def g_9_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(1320.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(180.0, dtype=tf.float64), x
    )


@tf.function
def g_9_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(7920.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(1980.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 45
    )


@tf.function
def g_9_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(41184.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(15840.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(990.0, dtype=tf.float64), x)
    )


@tf.function
def g_9_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(192192.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(102960.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(11880.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 165
    )


@tf.function
def g_9_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(823680.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(576576.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(102960.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(3960.0, dtype=tf.float64), x)
    )


@tf.function
def g_9_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(3294720.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(2882880.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(720720.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(51480.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 495
    )


@tf.function
def g_9_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(12446720.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(13178880.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(4324320.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(480480.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(12870.0, dtype=tf.float64), x)
    )


@tf.function
def g_9_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(44808192.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(56010240.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(23063040.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(3603600.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(180180.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 1287
    )


@tf.function
def g_9_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(154791936.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(224040960.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(112020480.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(23063040.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(1801800.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(36036.0, dtype=tf.float64), x)
    )


@tf.function
def g_9_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(515973120.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(851355648.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(504092160.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(130690560.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(14414400.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(540540.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 3003
    )


@tf.function
def g_10_0(x: tf.Tensor):
    return ONE


@tf.function
def g_10_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(20.0, dtype=tf.float64), x)


@tf.function
def g_10_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(220.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 10


@tf.function
def g_10_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(1760.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(220.0, dtype=tf.float64), x
    )


@tf.function
def g_10_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(11440.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(2640.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 55
    )


@tf.function
def g_10_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(64064.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(22880.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(1320.0, dtype=tf.float64), x)
    )


@tf.function
def g_10_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(320320.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(160160.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(17160.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 220
    )


@tf.function
def g_10_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1464320.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(960960.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(160160.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(5720.0, dtype=tf.float64), x)
    )


@tf.function
def g_10_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(6223360.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(5125120.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(1201200.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(80080.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 715
    )


@tf.function
def g_10_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(24893440.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(24893440.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(7687680.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(800800.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(20020.0, dtype=tf.float64), x)
    )


@tf.function
def g_10_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(94595072.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(112020480.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(43563520.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(6406400.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(300300.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 2002
    )


@tf.function
def g_10_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(343982080.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(472975360.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(224040960.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(43563520.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(3203200.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(60060.0, dtype=tf.float64), x)
    )


@tf.function
def g_10_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1203937280.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(1891901440.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(1064194560.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(261381120.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(27227200.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(960960.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 5005
    )


@tf.function
def g_11_0(x: tf.Tensor):
    return ONE


@tf.function
def g_11_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(22.0, dtype=tf.float64), x)


@tf.function
def g_11_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(264.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 11


@tf.function
def g_11_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(2288.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(264.0, dtype=tf.float64), x
    )


@tf.function
def g_11_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(16016.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(3432.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 66
    )


@tf.function
def g_11_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(96096.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(32032.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(1716.0, dtype=tf.float64), x)
    )


@tf.function
def g_11_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(512512.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(240240.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(24024.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 286
    )


@tf.function
def g_11_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(2489344.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(1537536.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(240240.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(8008.0, dtype=tf.float64), x)
    )


@tf.function
def g_11_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(11202048.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(8712704.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(1921920.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(120120.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 1001
    )


@tf.function
def g_11_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(47297536.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(44808192.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(13069056.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(1281280.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(30030.0, dtype=tf.float64), x)
    )


@tf.function
def g_11_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(189190144.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(212838912.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(78414336.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(10890880.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(480480.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 3003
    )


@tf.function
def g_11_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(722362368.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(945950720.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(425677824.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(78414336.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(5445440.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(96096.0, dtype=tf.float64), x)
    )


@tf.function
def g_11_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(2648662016.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(3972993024.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(2128389120.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(496624128.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(49008960.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(1633632.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 8008
    )


@tf.function
def g_12_0(x: tf.Tensor):
    return ONE


@tf.function
def g_12_1(x: tf.Tensor):
    return tf.math.multiply(tf.constant(24.0, dtype=tf.float64), x)


@tf.function
def g_12_2(x: tf.Tensor):
    return tf.math.multiply(tf.constant(312.0, dtype=tf.float64), tf.math.pow(x, 2.0)) - 12


@tf.function
def g_12_3(x: tf.Tensor):
    return tf.math.multiply(tf.constant(2912.0, dtype=tf.float64), tf.math.pow(x, 3.0)) - tf.math.multiply(
        tf.constant(312.0, dtype=tf.float64), x
    )


@tf.function
def g_12_4(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(21840.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(4368.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 78
    )


@tf.function
def g_12_5(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(139776.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(43680.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(2184.0, dtype=tf.float64), x)
    )


@tf.function
def g_12_6(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(792064.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(349440.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(32760.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 364
    )


@tf.function
def g_12_7(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(4073472.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(2376192.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(349440.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(10920.0, dtype=tf.float64), x)
    )


@tf.function
def g_12_8(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(19348992.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(14257152.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(2970240.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(174720.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 1365
    )


@tf.function
def g_12_9(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(85995520.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        - tf.math.multiply(tf.constant(77395968.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        + tf.math.multiply(tf.constant(21385728.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        - tf.math.multiply(tf.constant(1980160.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        + tf.math.multiply(tf.constant(43680.0, dtype=tf.float64), x)
    )


@tf.function
def g_12_10(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(361181184.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        - tf.math.multiply(tf.constant(386979840.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        + tf.math.multiply(tf.constant(135442944.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        - tf.math.multiply(tf.constant(17821440.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        + tf.math.multiply(tf.constant(742560.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        - 4368
    )


@tf.function
def g_12_11(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(1444724736.0, dtype=tf.float64), tf.math.pow(x, 11.0))
        - tf.math.multiply(tf.constant(1805905920.0, dtype=tf.float64), tf.math.pow(x, 9.0))
        + tf.math.multiply(tf.constant(773959680.0, dtype=tf.float64), tf.math.pow(x, 7.0))
        - tf.math.multiply(tf.constant(135442944.0, dtype=tf.float64), tf.math.pow(x, 5.0))
        + tf.math.multiply(tf.constant(8910720.0, dtype=tf.float64), tf.math.pow(x, 3.0))
        - tf.math.multiply(tf.constant(148512.0, dtype=tf.float64), x)
    )


@tf.function
def g_12_12(x: tf.Tensor):
    return (
        tf.math.multiply(tf.constant(5538111488.0, dtype=tf.float64), tf.math.pow(x, 12.0))
        - tf.math.multiply(tf.constant(7945986048.0, dtype=tf.float64), tf.math.pow(x, 10.0))
        + tf.math.multiply(tf.constant(4063288320.0, dtype=tf.float64), tf.math.pow(x, 8.0))
        - tf.math.multiply(tf.constant(902952960.0, dtype=tf.float64), tf.math.pow(x, 6.0))
        + tf.math.multiply(tf.constant(84651840.0, dtype=tf.float64), tf.math.pow(x, 4.0))
        - tf.math.multiply(tf.constant(2673216.0, dtype=tf.float64), tf.math.pow(x, 2.0))
        + 12376
    )


luts = [
    {
        0: g_0_0,
        1: g_0_1,
        2: g_0_1,
        3: g_0_1,
        4: g_0_1,
        5: g_0_1,
        6: g_0_1,
        7: g_0_1,
        8: g_0_1,
        9: g_0_1,
        10: g_0_1,
        11: g_0_1,
        12: g_0_1,
    },
    {
        0: g_1_0,
        1: g_1_1,
        2: g_1_2,
        3: g_1_3,
        4: g_1_4,
        5: g_1_5,
        6: g_1_6,
        7: g_1_7,
        8: g_1_8,
        9: g_1_9,
        10: g_1_10,
        11: g_1_11,
        12: g_1_12,
    },
    {
        0: g_2_0,
        1: g_2_1,
        2: g_2_2,
        3: g_2_3,
        4: g_2_4,
        5: g_2_5,
        6: g_2_6,
        7: g_2_7,
        8: g_2_8,
        9: g_2_9,
        10: g_2_10,
        11: g_2_11,
        12: g_2_12,
    },
    {
        0: g_3_0,
        1: g_3_1,
        2: g_3_2,
        3: g_3_3,
        4: g_3_4,
        5: g_3_5,
        6: g_3_6,
        7: g_3_7,
        8: g_3_8,
        9: g_3_9,
        10: g_3_10,
        11: g_3_11,
        12: g_3_12,
    },
    {
        0: g_4_0,
        1: g_4_1,
        2: g_4_2,
        3: g_4_3,
        4: g_4_4,
        5: g_4_5,
        6: g_4_6,
        7: g_4_7,
        8: g_4_8,
        9: g_4_9,
        10: g_4_10,
        11: g_4_11,
        12: g_4_12,
    },
    {
        0: g_5_0,
        1: g_5_1,
        2: g_5_2,
        3: g_5_3,
        4: g_5_4,
        5: g_5_5,
        6: g_5_6,
        7: g_5_7,
        8: g_5_8,
        9: g_5_9,
        10: g_5_10,
        11: g_5_11,
        12: g_5_12,
    },
    {
        0: g_6_0,
        1: g_6_1,
        2: g_6_2,
        3: g_6_3,
        4: g_6_4,
        5: g_6_5,
        6: g_6_6,
        7: g_6_7,
        8: g_6_8,
        9: g_6_9,
        10: g_6_10,
        11: g_6_11,
        12: g_6_12,
    },
    {
        0: g_7_0,
        1: g_7_1,
        2: g_7_2,
        3: g_7_3,
        4: g_7_4,
        5: g_7_5,
        6: g_7_6,
        7: g_7_7,
        8: g_7_8,
        9: g_7_9,
        10: g_7_10,
        11: g_7_11,
        12: g_7_12,
    },
    {
        0: g_8_0,
        1: g_8_1,
        2: g_8_2,
        3: g_8_3,
        4: g_8_4,
        5: g_8_5,
        6: g_8_6,
        7: g_8_7,
        8: g_8_8,
        9: g_8_9,
        10: g_8_10,
        11: g_8_11,
        12: g_8_12,
    },
    {
        0: g_9_0,
        1: g_9_1,
        2: g_9_2,
        3: g_9_3,
        4: g_9_4,
        5: g_9_5,
        6: g_9_6,
        7: g_9_7,
        8: g_9_8,
        9: g_9_9,
        10: g_9_10,
        11: g_9_11,
        12: g_9_12,
    },
    {
        0: g_10_0,
        1: g_10_1,
        2: g_10_2,
        3: g_10_3,
        4: g_10_4,
        5: g_10_5,
        6: g_10_6,
        7: g_10_7,
        8: g_10_8,
        9: g_10_9,
        10: g_10_10,
        11: g_10_11,
        12: g_10_12,
    },
    {
        0: g_11_0,
        1: g_11_1,
        2: g_11_2,
        3: g_11_3,
        4: g_11_4,
        5: g_11_5,
        6: g_11_6,
        7: g_11_7,
        8: g_11_8,
        9: g_11_9,
        10: g_11_10,
        11: g_11_11,
        12: g_11_12,
    },
    {
        0: g_12_0,
        1: g_12_1,
        2: g_12_2,
        3: g_12_3,
        4: g_12_4,
        5: g_12_5,
        6: g_12_6,
        7: g_12_7,
        8: g_12_8,
        9: g_12_9,
        10: g_12_10,
        11: g_12_11,
        12: g_12_12,
    },
    {
        0: g_13_0,
        1: g_13_1,
        2: g_13_2,
        3: g_13_3,
        4: g_13_4,
        5: g_13_5,
        6: g_13_6,
        7: g_13_7,
        8: g_13_8,
        9: g_13_9,
        10: g_13_10,
        11: g_13_11,
        12: g_13_12,
    },
]


def gegenbauer_lut_ref(x: tf.Tensor, nu: int, n: int) -> tf.Tensor:  # (172) # parameter nu, order n
    x = tf.where(x == ONE, x - 1e-9, x)
    return luts[nu][n](x)


def gegenbauer_lut(x: tf.Tensor, nu: int, n: int) -> tf.Tensor:  # (172) # parameter nu, order n
    x = tf.where(x == ONE, x - 1e-9, x)
    gegenbauer_coeffs = coeffs[nu][n]
    return tf.math.polyval(gegenbauer_coeffs, x)


def gegenbauer_compute(x: tf.Tensor):
    raise NotImplementedError


def test_gegenbauer():
    x = tf.random.uniform((100, 100, 100), minval=0, maxval=1, dtype=tf.float64, seed=2)
    for nu in range(len(coeffs)):
        for n in range(len(coeffs[nu])):
            deviation = gegenbauer_lut_ref(x, nu, n) - gegenbauer_lut(x, nu, n)
            max_deviation = tf.reduce_max(tf.abs(deviation))
            assert max_deviation < 1e-5


def profile_gegenbauer():
    x = tf.random.uniform((100, 100, 100), minval=0, maxval=1, dtype=tf.float64, seed=2)
    import time

    for nu in range(len(coeffs)):
        for n in range(len(coeffs[nu])):
            t0 = time.perf_counter()
            _ = gegenbauer_lut_ref(x, nu, n)
            t1 = time.perf_counter()
            _ = gegenbauer_lut(x, nu, n)
            t2 = time.perf_counter()
            print(f"{(t1 - t0) / (t2 - t1)} times faster")


if __name__ == "__main__":
    # x = tf.constant([0, 0.1], dtype=tf.float64)
    # print(gegenbauer_lut(x, 1, 1))
    test_gegenbauer()
    # profile_gegenbauer()
