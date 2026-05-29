from copy import deepcopy
from typing import List, Tuple

import tensorflow as tf
import numpy as np

from mcrpy.orientation import Orientation, Hypersphere
from mcrpy.src.Symmetry import Symmetry

from mcrpy.src.legendre_derivatives import legendre_derivative
from mcrpy.src.gegenbauer_derivatives import gegenbauer_derivative
from mcrpy.src.gegenbauer_polynomials import gegenbauer_lut
from mcrpy.src.legendre_polynomials import legendre_lut

ONE = tf.constant(1.0, dtype=tf.float64)
EPS = 1e-9


def make_shsh_encoding(
    symmetry: Symmetry, orientation_representation_dimension: int, desired_shape_2d: Tuple[int]
) -> callable:
    z_l = []
    for i_function, (n, lam) in enumerate(symmetry.expansion_table):
        if i_function >= orientation_representation_dimension:
            break
        z_l.append(symmetrize_z(symmetry, n, lam))
    n_functions = len(z_l)
    assert n_functions == orientation_representation_dimension
    shape = [1] + [*desired_shape_2d] + [orientation_representation_dimension]

    @tf.function
    def shsh_encoding(x: Orientation) -> tf.Tensor:
        x_hs = x.astype(Hypersphere)
        s = tf.reshape(tf.stack([zi(x_hs) for zi in z_l], axis=-1), shape)
        return s

    return shsh_encoding


def symmetrize_z(symmetry: Symmetry, n: int, lam: int) -> callable:
    def make_partial_z(n_z: int, l_z: int, m_z: int, i_z: bool) -> callable:
        def partial_z(ori: Orientation) -> tf.Tensor:
            return z_unsymm(ori, n_z, l_z, m_z, i_z)

        return partial_z

    def make_add_partial_z(coeff_list: List) -> callable:
        term_list = [make_partial_z(n, l_z, m_z, i_z) for _, l_z, m_z, i_z in coeff_list]
        factor_list = [f_z for f_z, _, _, _ in coeff_list]

        def add_partial_z(ori: Orientation) -> tf.Tensor:
            ori = ori.astype(Hypersphere)
            return tf.math.reduce_sum([f * t(ori) for f, t in zip(factor_list, term_list)], axis=0, keepdims=True)

        return add_partial_z

    key = (n, lam)
    et = symmetry.expansion_table
    if key not in et:
        raise NotImplementedError("Combination of n and lambda not implemented - either too high or wrong")
    return make_add_partial_z(et[key])


def z_symm(ori: Orientation, symmetry: Symmetry, n: int, lam: int) -> tf.Tensor:
    return symmetrize_z(symmetry, n, lam)(ori)


def z_unsymm(ori: Orientation, n: int, l: int, m: int, i: bool) -> tf.Tensor:
    ori_hypersphere = ori.astype(Hypersphere)
    omega = ori_hypersphere.omega
    theta = ori_hypersphere.theta
    phi = ori_hypersphere.phi
    if m == 0 and i:
        return tf.zeros(tf.shape(phi), dtype=tf.float64)
    order_factor = compute_order_factor(
        tf.constant(n, dtype=tf.float64), tf.constant(l, dtype=tf.float64), tf.constant(m, dtype=tf.float64)
    )
    omega_half = 0.5 * omega
    omega_term = tf.math.pow(tf.sin(omega_half), l) * gegenbauer_lut(tf.cos(omega_half), l + 1, n - l)
    theta_term = legendre_lut(tf.cos(theta), m, l)
    if i:
        phi_term = tf.sin(m * phi)
    else:
        phi_term = ONE / tf.sqrt(2.0 * ONE) if m == 0 else tf.cos(m * phi)
    return order_factor * omega_term * theta_term * phi_term


@tf.function
def compute_order_factor(n, l, m):
    return (
        tf.math.pow(-ONE, m)
        * (tf.math.pow(2.0 * ONE, l) * factorial(l))
        / np.pi
        * tf.sqrt(
            (2.0 * ONE * l + ONE)
            * factorial(l - m)
            * (n + ONE)
            * factorial(n - l)
            / (factorial(l + m) * factorial(n + l + ONE))
        )
    )


@tf.function
def compute_gegenbauer_order_factor(n, nu):
    return (
        tf.math.pow(-2.0 * ONE, n)
        / factorial(n)
        * (gamma(nu + n) * gamma(2.0 * nu + n) / (gamma(nu) * gamma(2.0 * nu + 2.0 * n)))
    )


def gegenbauer(x: tf.Tensor, nu: int, n: int) -> tf.Tensor:  # (172) # parameter nu, order n
    x = tf.where(x == ONE, x - 1e-9, x)
    order_factor = compute_gegenbauer_order_factor(tf.constant(n, dtype=tf.float64), tf.constant(nu, dtype=tf.float64))
    poly_term = tf.math.pow(1 - tf.math.square(x), 0.5 - nu)
    derivative_term = gegenbauer_derivative(x, nu, n)
    return order_factor * poly_term * derivative_term


def legendre(x: tf.Tensor, m: int, l: int) -> tf.Tensor:  # (171)
    x = tf.where(x == ONE, x - 1e-9, x)
    cordon_shortly_phase_factor = tf.math.pow(-ONE, m)
    poly_term = ONE / (tf.math.pow(2.0 * ONE, l) * factorial(l)) * tf.math.pow(ONE - tf.math.square(x), m * 0.5)
    derivative_term = legendre_derivative(x, l, m)
    return cordon_shortly_phase_factor * poly_term * derivative_term


def factorial(x: tf.Tensor) -> tf.Tensor:  # x in N
    tf.assert_equal(tf.reduce_all(x >= 0), True)
    return gamma(x + 1)


def gamma(x: tf.Tensor) -> tf.Tensor:  # x in N
    x = tf.cast(x, tf.float64)
    return tf.exp(tf.math.lgamma(x))


def stereographic_projection(x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, is_normalized: bool = False) -> Tuple[tf.Tensor]:
    if not is_normalized:
        inv_norm = ONE / tf.math.sqrt(tf.math.square(x) + tf.math.square(y) + tf.math.square(z))
        x = x * inv_norm
        y = y * inv_norm
        z = z * inv_norm
    return x / (ONE + z), y / (ONE + z)


def plot_3d(
    f_index: Tuple[int] = (8, 1), savefig: bool = True, n_samples: int = 20, omega_deg: float = 0.2, OFFSET: float = 1.0
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib
    from mcrpy.src.Symmetry import Cubic

    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 10,
            "figure.titlesize": "medium",
            "text.usetex": "True",
            "pgf.rcfonts": "False",
            "pgf.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{mathrsfs}",
        }
    )

    count = n_samples
    phi = np.linspace(0, 2 * np.pi, n_samples)
    theta = np.linspace(0, np.pi, n_samples)
    s = (len(phi), len(theta))
    x = np.zeros(s)
    y = np.zeros(s)
    z = np.zeros(s)
    fcolors = np.zeros(s)
    z_symmetrized = symmetrize_z(Cubic, *f_index)
    for i, phi_i in enumerate(phi):
        for j, theta_j in enumerate(theta):
            ori = Hypersphere(tf.constant([omega_deg * np.pi / 180.0, theta_j, phi_i], dtype=tf.float64))
            val = z_symmetrized(ori)
            fcolors[i, j] = val
            r = OFFSET
            x[i, j] = r * np.sin(theta_j) * np.cos(phi_i)
            y[i, j] = r * np.sin(theta_j) * np.sin(phi_i)
            z[i, j] = r * np.cos(theta_j)
    fig = plt.figure(figsize=(3, 2.4))
    ax = fig.add_subplot([0, 0, 1, 1], projection="3d", aspect="equal")
    minn = -1
    maxx = 1
    norm = matplotlib.colors.Normalize(minn, maxx)
    cmap = "seismic"  # "Reds" if args.reference is None else "bwr"
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    fcolors = m.to_rgba(fcolors)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax.plot_surface(x, y, z, rcount=count, ccount=count, facecolors=fcolors)  # , shading='gouraud')
    ax.set_xlabel(r"$q_1 / | \vec q \, |$")
    ax.set_ylabel(r"$q_2 / | \vec q \, |$")
    ax.set_zlabel(r"$q_3 / | \vec q \, |$")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.7, wspace=0.2, hspace=0.2)
    if not savefig:
        plt.show()
    else:
        plt.savefig(f"plot_shsh_{f_index[0]}_{f_index[1]}_omega_{omega_deg}.png", bbox_inches="tight", dpi=600)
    plt.close()


def plot_3d_unsymm(
    n, l, m, savefig: bool = True, n_samples: int = 20, omega: float = 0.2, OFFSET: float = 1.0, compl: bool = False
):
    import matplotlib
    import matplotlib.pyplot as plt

    count = n_samples
    phi = np.linspace(0, 2 * np.pi, n_samples)
    theta = np.linspace(0, np.pi, n_samples)
    s = (len(phi), len(theta))
    x = np.zeros(s)
    y = np.zeros(s)
    z = np.zeros(s)
    cols = np.zeros(s)
    for i, phi_i in enumerate(phi):
        for j, theta_j in enumerate(theta):
            ori = Hypersphere(tf.constant([omega, theta_j, phi_i], dtype=tf.float64))
            cols[i, j] = z_unsymm(ori, n=n, l=l, m=m, i=compl)  # const
            r = OFFSET
            x[i, j] = r * np.sin(theta_j) * np.cos(phi_i)
            y[i, j] = r * np.sin(theta_j) * np.sin(phi_i)
            z[i, j] = r * np.cos(theta_j)
    fig = plt.figure(figsize=(3, 2.4))
    ax = fig.add_subplot(projection="3d")
    fcolors = cols
    minn, maxx = fcolors.min(), fcolors.max()
    minn = minn - np.abs(0.01 * minn) - EPS
    maxx = maxx + np.abs(0.01 * maxx) + EPS
    norm = matplotlib.colors.Normalize(minn, maxx)
    cmap = "bwr"  # "Reds" if args.reference is None else "bwr"
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fcolors = mappable.to_rgba(fcolors)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax.plot_surface(x, y, z, rcount=count, ccount=count, facecolors=fcolors)
    cbar = plt.colorbar(mappable)
    plt.tight_layout()
    if not savefig:
        plt.show()
    else:
        plt.savefig(f"plot_shsh_unsymm_n_{n}_l_{l}_m_{m}_i_{compl}_omega_{omega}.png", bbox_inches="tight", dpi=600)
    plt.close()


def plot_projection_unsymm(
    n, l, m, omega_deg: int = 170, n_samples: int = 100, axis: bool = False, compl: bool = False, savefig: bool = True
):  # sourcery skip: extract-method
    # mesh 2D grid
    phi_proj = np.arange(0, 2.0 * (1 + 1 / n_samples) * np.pi, 2 * np.pi / n_samples)
    r_proj = np.arange(0, 1 * (1 + 1 / n_samples), 1 / n_samples)
    phi_proj_mg, r_proj_mg = np.meshgrid(phi_proj, r_proj)
    # convert to hyperspherical coords
    omega = omega_deg / np.pi * np.ones(phi_proj_mg.shape)
    phi = phi_proj_mg
    theta = 2 * np.arctan(r_proj_mg)
    ori = Hypersphere(np.stack([omega, theta, phi], axis=-1).astype(np.float64))
    # compute bf values
    f = z_unsymm(ori, n, l, m, compl)
    assert f.shape == (n_samples + 1, n_samples + 1, 1)
    f = f[:, :, 0]
    # make plot
    if axis:
        import matplotlib

        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "font.size": 10,
                "figure.titlesize": "medium",
                "text.usetex": "True",
                "pgf.rcfonts": "False",
                "pgf.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{mathrsfs}",
            }
        )
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[2.5, 2.5] if axis else [3.5, 3.5])
    ax = fig.add_axes([0.01, 0.01, 0.99, 0.99], polar=True)
    ax.pcolormesh(
        phi_proj_mg, r_proj_mg, f, cmap="seismic", vmin=-1, vmax=1, shading="gouraud"
    )  # shading gouraud oder nearest
    # ax.pcolormesh(phi_proj_mg, r_proj_mg, f, edgecolors='face', cmap='seismic', vmin=-1, vmax=1, shading='gouraud') # shading gouraud oder nearest
    if axis:
        ax.set_xticks([np.pi * i for i in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]])
        ax.set_yticks([])
        plt.text(0.5 * 45 / 180 * np.pi, 1.1, r"$\phi$")  # , transform=ax.transAxes)
        plt.text(0.9, 1.05, r"$\omega = $" + f"{omega_deg:.1f}°", transform=ax.transAxes)
        plt.text(0.62, 0.51, r"$\theta$", transform=ax.transAxes)
        plt.text(0.5, 0.51, r"0°", transform=ax.transAxes)
        plt.text(0.72, 0.51, r"60°", transform=ax.transAxes)
        plt.text(0.92, 0.51, r"90°", transform=ax.transAxes)
        plt.polar([i / 180 * np.pi for i in [-1, 1]], [0.5, 0.5], "k-", lw=0.5)
        plt.polar([-0.5 * np.pi, 0.5 * np.pi], [0.01, 0.01], "k-", lw=0.5)
        plt.polar([0 / 180 * np.pi] * 2, [0, 1], "k-", lw=0.5)
    else:
        plt.axis("off")
        plt.title(r"$\omega = $" + f"{omega_deg:.1f}°")
    if savefig:
        addition = "" if axis else "_noaxis"
        plt.savefig(
            f"slice_shsh_unsymm_{n}_{l}_{m}_{compl}_omega_{omega_deg:08.1f}{addition}.png", bbox_inches="tight", dpi=600
        )
    else:
        plt.show()
    plt.close()


def plot_projection(
    f_index: Tuple[int] = (8, 1), omega_deg: int = 170, n_samples: int = 100, axis: bool = False, savefig: bool = True
):  # sourcery skip: extract-method
    from mcrpy.src.Symmetry import Cubic

    # mesh 2D grid
    phi_proj = np.arange(0, 2.0 * (1 + 1 / n_samples) * np.pi, 2 * np.pi / n_samples)
    r_proj = np.arange(0, 1 * (1 + 1 / n_samples), 1 / n_samples)
    phi_proj_mg, r_proj_mg = np.meshgrid(phi_proj, r_proj)
    # convert to hyperspherical coords
    omega = omega_deg / 180.0 * np.pi * np.ones(phi_proj_mg.shape)
    phi = phi_proj_mg
    theta = 2 * np.arctan(r_proj_mg)
    ori = Hypersphere(np.stack([omega, theta, phi], axis=-1).astype(np.float64))
    # compute bf values
    f = z_symm(ori, Cubic, *f_index)
    assert f.shape == (1, n_samples + 1, n_samples + 1, 1)
    f = f[0, :, :, 0]
    # make plot
    if axis:
        import matplotlib

        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "font.size": 14,
                # 'font.size': 10,
                "figure.titlesize": "medium",
                "text.usetex": "True",
                "pgf.rcfonts": "False",
                "pgf.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{mathrsfs}",
            }
        )
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[2.5, 2.5] if axis else [3.5, 3.5])
    ax = fig.add_axes([0.01, 0.01, 0.99, 0.99], polar=True)
    if axis:
        ax.set_xticks([np.pi * i for i in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]])
        ax.set_yticks([])
    ax.pcolormesh(
        phi_proj_mg, r_proj_mg, f, cmap="seismic", vmin=-1, vmax=1, shading="gouraud", zorder=100
    )  # shading gouraud oder nearest
    # ax.pcolormesh(phi_proj_mg, r_proj_mg, f, edgecolors='face', cmap='seismic', vmin=-1, vmax=1, shading='gouraud') # shading gouraud oder nearest
    if axis:
        plt.text(0.5 * 45 / 180 * np.pi, 1.1, r"$\varphi$", zorder=1000)  # , transform=ax.transAxes)
        # plt.text(0.85, 1.05, r'$\omega = $' + f'{omega_deg}°', transform=ax.transAxes, zorder=1000)
        # plt.text(0.85, 1.05, r'$\omega = $' + f'{omega_deg:.1f}°', transform=ax.transAxes)
        plt.text(0.62, 0.51, r"$\theta$", transform=ax.transAxes, zorder=1000)
        plt.text(0.5, 0.51, r"0°", transform=ax.transAxes, zorder=1000)
        plt.text(0.72, 0.51, r"60°", transform=ax.transAxes, zorder=1000)
        plt.text(0.92, 0.51, r"90°", transform=ax.transAxes, zorder=1000)
        plt.polar([i / 180 * np.pi for i in [-1, 1]], [0.5, 0.5], "k-", lw=0.5, zorder=1000)
        plt.polar([-0.5 * np.pi, 0.5 * np.pi], [0.01, 0.01], "k-", lw=0.5, zorder=1000)
        plt.polar([0 / 180 * np.pi] * 2, [0, 1], "k-", lw=0.5, zorder=1000)
    else:
        # pass
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.axis('off') # wenn nichtmal schwarzer kreis erwünscht
        # plt.title(r'$\omega = $' + f'{omega_deg:.1f}°')
    if savefig:
        addition = "" if axis else "_noaxis"
        plt.savefig(
            f"slice_shsh_{f_index[0]}_{f_index[1]}_omega_{omega_deg:08.1f}{addition}.png", bbox_inches="tight", dpi=150
        )  # dpi 600
    else:
        plt.show()
    plt.close()


def save_plots(in_3d: bool = False):
    from mcrpy.src.Symmetry import Cubic

    if in_3d:
        for omega_deg in range(125, 185, 10):
            for index in Cubic.expansion_table:
                plot_3d(index, savefig=True, n_samples=20, omega_deg=omega_deg)
    else:
        for index in Cubic.expansion_table:
            # for omega_deg in range(0, 361, 10): # no skipping
            #     plot_projection(index, savefig=True, n_samples=500, omega_deg=omega_deg, axis=False) # 100 samples
            plot_projection(index, savefig=True, n_samples=500, omega_deg=180, axis=True)  # 100 samples


def test_symmetry():
    from mcrpy.src.Symmetry import Cubic
    from mcrpy.orientation.test_orientation import _generate_axis_testdata

    ori = _generate_axis_testdata(shape=[30, 10]).astype(Hypersphere)
    symmetry = Cubic
    symmetric_equivalents = Cubic.apply_to_orientation(ori)
    for index in Cubic.expansion_table:
        z_ori = z_symm(ori, symmetry, *index)
        error = tf.reduce_max(
            tf.concat([z_symm(se, symmetry, *index) - z_ori for se in symmetric_equivalents], axis=0), axis=0
        )
        assert tf.reduce_all(error < 0.001)

