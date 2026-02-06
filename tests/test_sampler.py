import pytest
from thunder.data.sampler import PoissonSampler, RandomSampler, SequentialSampler


def test_random_sampler_no_replacement():
    sampler = RandomSampler(N=10, k=3, replacement=False, infinite=False)
    all_samples = []
    for sample in sampler:
        all_samples.extend(sample)
    assert len(all_samples) == 10
    assert len(set(all_samples)) == 10
    assert set(all_samples) == set(range(10))


def test_random_sampler_with_replacement():
    sampler = RandomSampler(N=10, k=3, replacement=True, infinite=False)
    all_samples = []
    for sample in sampler:
        all_samples.extend(sample)
    assert len(all_samples) == 10
    assert all(0 <= x < 10 for x in all_samples)


def test_random_sampler_infinite():
    sampler = RandomSampler(N=10, k=3, replacement=True, infinite=True)
    samples = []
    for _, sample in zip(range(100), sampler):
        samples.extend(sample)
    assert len(samples) == 300
    assert all(0 <= x < 10 for x in samples)


def test_random_sampler_remaining_values_less_than_k():
    sampler = RandomSampler(N=5, k=3, replacement=False, infinite=False)
    all_samples = []
    for sample in sampler:
        all_samples.extend(sample)
    assert len(all_samples) == 5
    assert len(set(all_samples)) == 5
    assert set(all_samples) == set(range(5))


if __name__ == "__main__":
    pytest.main()


def test_poisson_sampler_initialization():
    with pytest.raises(ValueError):
        PoissonSampler(N=-1, lam=3.0, k=1)
    with pytest.raises(ValueError):
        PoissonSampler(N=10, lam=3.0, k=0)


def test_poisson_sampler_with_replacement():
    sampler = PoissonSampler(N=10, lam=3.0, k=3, replacement=True, infinite=False)
    all_samples = []
    for sample in sampler:
        all_samples.extend(sample)
    assert len(all_samples) == 10
    assert all(isinstance(x, int) for x in all_samples)


def test_poisson_sampler_without_replacement():
    sampler = PoissonSampler(N=10, lam=3.0, k=3, replacement=False, infinite=False)
    all_samples = []
    for sample in sampler:
        all_samples.extend(sample)
    assert len(all_samples) == 10
    assert len(set(all_samples)) == 10
    assert set(all_samples) == set(range(10))


def test_poisson_sampler_infinite():
    sampler = PoissonSampler(N=10, lam=3.0, k=3, replacement=True, infinite=True)
    samples = []
    for _, sample in zip(range(100), sampler):
        samples.extend(sample)
    assert len(samples) == 300
    assert all(isinstance(x, int) for x in samples)


def test_poisson_sampler_remaining_values_less_than_k():
    sampler = PoissonSampler(N=5, lam=3.0, k=3, replacement=False, infinite=False)
    all_samples = []
    for sample in sampler:
        all_samples.extend(sample)
    assert len(all_samples) == 5
    assert len(set(all_samples)) == 5
    assert set(all_samples) == set(range(5))


def test_sequential_sampler_initialization():
    with pytest.raises(ValueError):
        SequentialSampler(N=-1, k=1)
    with pytest.raises(ValueError):
        SequentialSampler(N=10, k=0)


def test_sequential_sampler_no_replacement():
    sampler = SequentialSampler(N=10, k=3, replacement=False, infinite=False)
    all_samples = []
    for sample in sampler:
        all_samples.extend(sample)
    assert len(all_samples) == 10
    assert all_samples == list(range(10))


def test_sequential_sampler_infinite():
    sampler = SequentialSampler(N=10, k=3, replacement=False, infinite=True)
    samples = []
    for i, sample in zip(range(100), sampler):
        assert len(sample) == 3
        samples.extend(sample)
    assert len(samples) == 300
    assert samples[:100] == list(range(10)) * 10


def test_sequential_sampler_remaining_values_less_than_k():
    sampler = SequentialSampler(N=5, k=3, replacement=False, infinite=False)
    all_samples = []
    for i, sample in enumerate(sampler):
        assert len(sample) == (3 if i == 0 else 2)
        all_samples.extend(sample)
    assert len(all_samples) == 5
    assert all_samples == list(range(5))


if __name__ == "__main__":
    pytest.main()
