import math
import random
from collections.abc import Iterable, Callable, Sequence

import matplotlib.pyplot as plt


def raw_sequence(size: int) -> Iterable[int, int]:
    x: int = 0
    for n in range(size + 1):
        yield n / size, x
        x = x + random.randint(0, 1)


def sequence_scaled_1(size: int) -> Iterable[float, float]:
    for t, x in raw_sequence(size):
        yield t, x - size * t / 2


def sequence_scaled_2(size: int) -> Iterable[float, float]:
    for t, x in raw_sequence(size):
        yield t, (x - size * t / 2) / math.sqrt(size / 4)


def plot(
    seq: Callable[[int], Iterable[float, float]],
    y_label: str,
    filename: str,
    y_lim: tuple[float, float],
    sqrt: bool = False,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for size, ax in zip([10, 100, 1000, 10000], axs.flat):
        for _ in range(20):
            xys = list(seq(size))
            ax.plot([x for x, _ in xys], [y for _, y in xys])
        ax.set(xlabel="$t$", ylabel=y_label.format(size=size), title=f"$n={size}$")
        ax.grid()
        ax.set_ylim(*y_lim)
        if sqrt:
            ax.plot(
                [i / 100 for i in range(101)],
                [1.96 * math.sqrt(i / 100) for i in range(101)],
                color="black",
                linestyle="dashed",
                label=f"$\\pm 1.96\\sqrt{{t}}$",
            )
            ax.plot(
                [i / 100 for i in range(101)],
                [-1.96 * math.sqrt(i / 100) for i in range(101)],
                color="black",
                linestyle="dashed",
            )
            ax.legend()
    plt.tight_layout()
    plt.savefig(filename, transparent=True)


def plot_increments(
    seq: Callable[[int], Iterable[float, float]],
    size: int,
    filename: str,
    lines: bool = True,
    reset: bool = True,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    xyss = [list(seq(size)) for _ in range(20)]
    for xys in xyss:
        xys_part1 = [(x, y) for x, y in xys if 0.0 <= x <= 0.5]
        xys_part2 = [(x, y) for x, y in xys if 0.5 <= x <= 1.0]
        if lines:
            xys_part1 = [xys_part1[0], xys_part1[-1]]
            xys_part2 = [xys_part2[0], xys_part2[-1]]
        for xys_part, ax in zip([xys_part1, xys_part2], axs.flat):
            y0 = xys_part[0][1] if reset else 0.0
            ax.plot([x for x, _ in xys_part], [y - y0 for _, y in xys_part])
            ax.set(xlabel="$t$", ylabel="$W_n(t)$", title=f"$n={size}$")
            ax.set_ylim(-3.0, 3.0)
            ax.grid()
    plt.tight_layout()
    plt.savefig(filename, transparent=True)


if __name__ == "__main__":
    seed = 6
    random.seed(seed)
    plot_increments(sequence_scaled_2, 1000, "increment1.png", lines=False, reset=False)
    random.seed(seed)
    plot_increments(sequence_scaled_2, 1000, "increment2.png", lines=False, reset=True)
    random.seed(seed)
    plot_increments(sequence_scaled_2, 1000, "increment3.png", lines=True, reset=True)
    random.seed(seed)
    plot(raw_sequence, "$X_n(t)$", "binom1.png", (-0.5, 20.5))
    random.seed(seed)
    plot(sequence_scaled_1, "$X_n(t)-nt/2$", "binom2.png", (-10.5, 10.5))
    random.seed(seed)
    plot(
        sequence_scaled_2,
        "$(X_n(t)-nt/2)/\\sqrt{{n/4}}$",
        "binom3.png",
        y_lim=(-3.5, 3.5),
        sqrt=True,
    )
