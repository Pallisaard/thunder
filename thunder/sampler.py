from typing import List, TypedDict
import numpy as np

from thunder.utils import AutoInit


class SampleAndRemaining(TypedDict):
    samples: List[int]
    remaining: List[int]


def unpack_sample_and_remaining(sample_and_remaining: SampleAndRemaining):
    sample = sample_and_remaining["samples"]
    remaining = sample_and_remaining["remaining"]
    return sample, remaining


class BaseSampler(AutoInit):
    def __init__(
        self, N: int, k: int = 1, replacement: bool = False, infinite: bool = False
    ):
        if N <= 0:
            raise ValueError("N must be a positive integer")
        if k <= 0:
            raise ValueError("k must be a positive integer")

        self.N = N
        self.k = k
        self.replacement = replacement
        self.infinite = infinite
        self.remaining_values = list(range(N))
        self.sampled_count = 0

    def sample(self, possible_values: List[int]) -> SampleAndRemaining:
        raise NotImplementedError(
            "Subclasses must implement `sample(self, possible_values)`."
        )

    def __iter__(self):
        return self

    def __next__(self):
        if not self.infinite and self.sampled_count >= self.N:
            raise StopIteration
        samples, remaining = unpack_sample_and_remaining(
            self.sample(self.remaining_values)
        )
        self.sampled_count += len(samples)
        self.remaining_values = remaining
        return samples


class SequentialSampler(BaseSampler):
    """replacement is ignored"""

    sample_idx: int = 0

    def sample(self, possible_values) -> SampleAndRemaining:
        # If we use replacement, throw error because it doesn't make sense to use replacement.
        # if infinite, iterate in a circle.
        # If not infinite, Iterate through the array..
        if self.infinite:
            n_samples = (
                self.k if self.infinite else min(self.k, self.N - self.sampled_count)
            )
            samples = possible_values[:n_samples]
            remaining = possible_values[n_samples:] + samples
            return {"samples": samples, "remaining": remaining}
        elif len(possible_values) == 0:
            raise StopIteration
        elif len(possible_values) < self.k:
            return {"samples": possible_values, "remaining": []}
        samples = possible_values[: self.k]
        remaining = possible_values[self.k :]
        return {"samples": samples, "remaining": remaining}

    def filter_remaining_values(
        self, remaining_values: List[int], samples: List[int]
    ) -> List[int]:
        return remaining_values


class RandomSampler(BaseSampler):
    def sample(self, possible_values) -> SampleAndRemaining:
        if self.replacement:
            n_samples = (
                self.k if self.infinite else min(self.k, self.N - self.sampled_count)
            )
            samples: List[int] = np.random.randint(0, self.N, size=n_samples).tolist()
            return {"samples": samples, "remaining": possible_values}
        elif len(possible_values) < self.k:
            new_remaining: List[int] = []
            return {"samples": possible_values, "remaining": new_remaining}
        else:
            samples: List[int] = np.random.choice(
                possible_values, size=self.k, replace=False
            ).tolist()
            new_remaining = [x for x in possible_values if x not in samples]
            return {"samples": samples, "remaining": new_remaining}


class PoissonSampler(BaseSampler):
    def __init__(
        self,
        N: int,
        lam: float,
        k: int = 1,
        replacement: bool = True,
        infinite: bool = False,
    ):
        super().__init__(N, k, replacement, infinite)
        self.lam = lam

    def sample(self, possible_values: List[int]) -> SampleAndRemaining:
        if self.replacement:
            n_samples = (
                self.k if self.infinite else min(self.k, self.N - self.sampled_count)
            )
            samples: List[int] = np.random.poisson(self.lam, size=n_samples).tolist()
            return {"samples": samples, "remaining": possible_values}
        elif len(possible_values) == 0:
            raise StopIteration
        elif len(possible_values) < self.k:
            return {"samples": possible_values, "remaining": []}
        samples: List[int] = np.random.choice(
            self.remaining_values, size=self.k, replace=False
        ).tolist()
        new_remaining = [val for val in self.remaining_values if val not in samples]
        return {"samples": samples, "remaining": new_remaining}
