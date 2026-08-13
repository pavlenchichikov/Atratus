"""Training determinism: the same config must produce the same model.

Before GTRADE_SEED the nets were built with no seed at all unless seed-averaging
was on, so re-running one config gave a different model every time. That is what
made the research agent's neural_lift unreadable - it measured the reseed, not
the genome. These are the checks that fail if that regresses.
"""

import numpy as np

from core import net_hygiene as nh


def _tiny_net():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input((4, 3)),
        tf.keras.layers.LSTM(6),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def test_seeded_build_is_reproducible_and_seed_sensitive():
    import train_hybrid as th
    seed = nh.asset_seed("SP500", 1, 0)
    a = [w.numpy() for w in th._seeded_build(seed, _tiny_net).weights]
    b = [w.numpy() for w in th._seeded_build(seed, _tiny_net).weights]
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), \
        "same seed must give byte-identical initial weights"
    other = nh.asset_seed("NVDA", 1, 0)
    c = [w.numpy() for w in th._seeded_build(other, _tiny_net).weights]
    assert any(not np.array_equal(x, y) for x, y in zip(a, c)), \
        "a different asset must not train from the same initialization"


def test_dataset_shuffle_needs_the_explicit_seed():
    """The positive control for the `seed=` added to every training shuffle:
    without it the batch order differs between two identical runs."""
    import tensorflow as tf
    X = np.arange(24, dtype="float32").reshape(24, 1)

    def first_batch(seed):
        ds = tf.data.Dataset.from_tensor_slices(X)
        ds = ds.shuffle(24, seed=seed) if seed is not None else ds.shuffle(24)
        return list(next(iter(ds.batch(24))).numpy().flatten())

    seed = nh.asset_seed("SP500", 1, 0)
    assert first_batch(seed) == first_batch(seed)
    assert first_batch(None) != first_batch(None)


def test_cache_key_is_namespaced_by_the_seed(monkeypatch):
    """Rows cached by an unseeded run must not be served to a seeded one, and a
    GTRADE_SEED re-roll must not read the previous roll's rows."""
    from core import ar_memory
    monkeypatch.setenv("GTRADE_SEED", "1000")
    base_a = ar_memory.base_key("SP500", {})
    gen_a = ar_memory.genome_key("SP500", "sig", "full")
    monkeypatch.setenv("GTRADE_SEED", "4242")
    assert ar_memory.base_key("SP500", {}) != base_a
    assert ar_memory.genome_key("SP500", "sig", "full") != gen_a
