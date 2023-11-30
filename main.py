import math
import random
from collections.abc import Iterable, Callable

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


if __name__ == "__main__":
    seed = 5
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
