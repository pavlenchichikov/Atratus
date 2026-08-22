"""Loading of saved champion models.

Moved out of backtest.py: the loader is also needed by predict.py, alert_bot.py
and signal_engine.py, so importing it from the backtest script was wrong.
"""

import json
import os
import threading
import types

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Flatten,
    Input,
    Multiply,
    Permute,
    RepeatVector,
)
from tensorflow.keras.models import Model

from core.architectures import ReduceSumLayer, build_lstm_attention
from core.logger import get_logger

logger = get_logger("model_io")


_SERVE_CUSTOM_OBJECTS = {"ReduceSumLayer": ReduceSumLayer}
_LAMBDA_PATCH_LOCK = threading.Lock()


def _lambda_cast_call(self, inputs, mask=None, training=None):
    """Replacement body for reloaded Lambda layers. Every Lambda this project
    ever saved is the dtype cast `lambda t: tf.cast(t, compute_dt)`; Keras 3
    reloads a marshalled lambda without its globals (`tf` missing) and without
    an output shape, so the original dies at first call. Replaying the cast
    (shape-preserving) restores the saved model's real behavior."""
    return tf.cast(inputs, tf.keras.mixed_precision.global_policy().compute_dtype)


def _lambda_identity_shape(self, input_shape):
    return input_shape


def load_keras_native(path):
    """The straight Keras 3 load of a natively saved .keras champion - the format
    every model since the 2026-06 full retrain is in. Returns the model or None.

    This must be tried FIRST: the legacy V50/V49 rebuild paths reconstruct
    fixed-size architectures (192/96, 128/64) and cannot fit the adaptive-sized
    champions - their name-matched partial weight loads used to leave layers at
    random init and the member quietly predicted ~0.5 noise.

    Lambda handling: custom_objects cannot override the builtin Lambda class, so
    the class is patched (see _lambda_cast_call) only for the duration of the
    load, then the cast is pinned onto the loaded Lambda INSTANCES and the class
    restored - process-wide Lambda behavior is untouched."""
    lam_cls = tf.keras.layers.Lambda
    with _LAMBDA_PATCH_LOCK:
        orig_call = lam_cls.call
        orig_shape = lam_cls.compute_output_shape
        lam_cls.call = _lambda_cast_call
        lam_cls.compute_output_shape = _lambda_identity_shape
        try:
            try:
                model = tf.keras.models.load_model(
                    path, safe_mode=False, custom_objects=_SERVE_CUSTOM_OBJECTS)
            except TypeError:
                # safe_mode is a Keras 3 argument. The GPU env is pinned to TF 2.10
                # (the last native-Windows CUDA build), whose load_model does not
                # accept it - and there the flag is redundant anyway, because Keras
                # 2 has no safe-mode lambda restriction to switch off. Without this
                # fallback every champion fails to load in that env and the legacy
                # rebuild path quietly serves a half-initialised net at ~0.5.
                model = tf.keras.models.load_model(
                    path, custom_objects=_SERVE_CUSTOM_OBJECTS)
        except Exception as e:
            # debug, not warning: this is the FIRST of several attempts, and
            # since the embedded-config rebuild landed the later ones carry
            # most of the book - every legacy champion fails here and loads a
            # moment later. Warning on it printed two lines per asset for 200+
            # assets on every predict run, which is how a real loss would go
            # unnoticed among them. The caller warns, once, only if every path
            # failed; see _warn_lost.
            logger.debug("Native load failed for %s (%s: %s)",
                         path, type(e).__name__, e)
            return None
        finally:
            lam_cls.call = orig_call
            lam_cls.compute_output_shape = orig_shape
    for layer in model.layers:
        if isinstance(layer, lam_cls):
            layer.call = types.MethodType(_lambda_cast_call, layer)
            layer.compute_output_shape = types.MethodType(_lambda_identity_shape, layer)
    return model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OPTUNA_PARAMS_PATH = os.path.join(MODEL_DIR, "optuna_params.json")


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_lstm_legacy(input_shape):
    """V49 architecture: LSTM(128)+Dropout+LSTM(64), Bahdanau-style attention, Lambda-ReduceSum."""
    timesteps = input_shape[0]
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    # V49 attention: Dense(1,tanh), Flatten, Dense(ts,softmax), RepeatVector, Permute, Multiply
    a = Dense(1, activation='tanh')(x)
    a = Flatten()(a)
    a = Dense(timesteps, activation='softmax')(a)
    a = RepeatVector(64)(a)
    a = Permute((2, 1))(a)
    x = Multiply()([x, a])
    x = ReduceSumLayer()(x)
    x = Dense(32, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


# Keys a Keras 2 layer config carries that Keras 3 refuses. `time_major` was
# dropped from the RNN signature; `dtype` used to be a plain string and is now a
# policy object, so the string reaches DTypePolicy and dies on .quantization_mode.
# Dropping both leaves the layer on the default policy, which changes nothing a
# serve-time forward pass can see: the weights are the same shapes either way.
# Only these two. A transformer config also carries MultiHeadAttention's
# query/key/value_shape, which Keras 3 rejects - but stripping them only moves
# the failure to the next incompatible key, and the transformer does not need
# this path at all: transformer_kwargs_from_h5 reads num_heads and ff_dim from
# the WEIGHT SHAPES, which is not a guess. Checked 2026-08-22, it loads with
# real spread (SP500 std 0.071) without any of this.
_KERAS2_ONLY_CONFIG_KEYS = ("time_major", "dtype")


def _scrub_keras2_config(obj):
    if isinstance(obj, dict):
        return {k: _scrub_keras2_config(v) for k, v in obj.items()
                if k not in _KERAS2_ONLY_CONFIG_KEYS}
    if isinstance(obj, list):
        return [_scrub_keras2_config(v) for v in obj]
    return obj


def _keras2_op_layers(cfg):
    """Rewrite Keras 2 `TFOpLambda` nodes as the Keras 3 layer they stand for.

    TFOpLambda is Keras 2's wrapper around a raw TF op used inline in a
    functional graph; Keras 3 has no such class, so a config containing one
    cannot be deserialised at all. Every occurrence in this project's champions
    is the residual join inside a TCN block: measured 2026-08-21 over all 35
    affected files, 105 of 105 are `__operators__.add` with a single node. That
    is exactly `keras.layers.Add`, which carries no weights, so swapping it in
    changes the graph's spelling and not its arithmetic.

    The second input hides in the node's KWARGS (`{"y": [...]}`) rather than
    sitting beside the first, which is the whole reason a plain rename does not
    work - the node has to be rebuilt as an ordinary two-input node.

    Returns None for anything it cannot express, so the caller falls back
    instead of building a model that is subtly not the champion.
    """
    layers = (cfg or {}).get("layers")
    if not isinstance(layers, list):
        return cfg
    out = []
    for layer in layers:
        if layer.get("class_name") != "TFOpLambda":
            out.append(layer)
            continue
        conf = layer.get("config") or {}
        nodes = layer.get("inbound_nodes") or []
        if conf.get("function") != "__operators__.add" or len(nodes) != 1:
            return None
        node = nodes[0]
        if not (isinstance(node, list) and len(node) == 4
                and isinstance(node[3], dict)):
            return None
        other = node[3].get("y")
        if not (isinstance(other, list) and len(other) >= 3):
            return None
        name = layer.get("name") or conf.get("name")
        out.append({
            "class_name": "Add",
            "name": name,                     # kept: the graph refers to it
            "config": {"name": name,
                       "trainable": conf.get("trainable", True)},
            "inbound_nodes": [[[node[0], node[1], node[2], {}],
                               [other[0], other[1], other[2], {}]]],
        })
    rewritten = dict(cfg)
    rewritten["layers"] = out
    return rewritten


def _from_embedded_config(path):
    """Rebuild a legacy HDF5 champion from the architecture IT carries.

    Training runs under Keras 2 in the GPU environment and writes HDF5; serving
    runs under Keras 3 and cannot open it. The other rebuild paths guess the
    architecture from a builder's current defaults, so they only fit a champion
    whose sizes happen to match - and optuna picks its own. Measured on SP500
    2026-08-21: the file holds a 80/40-unit two-layer attention net, while V50
    expects 192 units and V49 expects six layers, so both fail and the member is
    dropped. But the HDF5 also carries `model_config`, the real architecture, so
    nothing has to be guessed at all.

    Returns the model, or None when this is not that kind of file. Weights are
    loaded by name and shape, so a config that did not match would raise rather
    than serve a half-initialised net.
    """
    import shutil

    import h5py

    if _detect_format(path) != "hdf5":
        return None
    h5_path = path.replace(".keras", ".cfg.tmp.h5")
    try:
        shutil.copy2(path, h5_path)
        with h5py.File(h5_path, "r") as fh:
            raw = fh.attrs.get("model_config")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        cfg = json.loads(raw)
        graph = _keras2_op_layers(cfg["config"])
        if graph is None:
            logger.debug("Unconvertible Keras 2 op in %s", path)
            return None
        model = Model.from_config(_scrub_keras2_config(graph))
        model.load_weights(h5_path)
        return model
    except Exception as exc:
        logger.debug("Embedded-config rebuild failed for %s: %s", path, exc)
        return None
    finally:
        if os.path.exists(h5_path):
            try:
                os.remove(h5_path)
            except OSError:
                pass


def _detect_format(path):
    """Detect if file is ZIP (.keras native) or HDF5."""
    with open(path, 'rb') as f:
        header = f.read(4)
    return 'zip' if header[:2] == b'PK' else 'hdf5'


def _load_weights_keras3(model, h5_path):
    """Load weights from Keras 3.x h5 format (layers/<name>/vars/N) into a Keras 2.x model."""
    import h5py
    import numpy as np
    with h5py.File(h5_path, 'r') as f:
        if 'layers' not in f:
            raise ValueError("Not a Keras 3.x h5 file")
        h5_layers = f['layers']
        for layer in model.layers:
            if layer.name in h5_layers and 'vars' in h5_layers[layer.name]:
                vars_grp = h5_layers[layer.name]['vars']
                weights = [np.array(vars_grp[str(i)]) for i in range(len(vars_grp))]
                layer.set_weights(weights)
            # Handle LSTM cell weights (stored as layers/<name>/cell/vars/N)
            elif layer.name in h5_layers and 'cell' in h5_layers[layer.name]:
                cell_grp = h5_layers[layer.name]['cell']
                if 'vars' in cell_grp:
                    vars_grp = cell_grp['vars']
                    weights = [np.array(vars_grp[str(i)]) for i in range(len(vars_grp))]
                    layer.set_weights(weights)


def get_lookback(reg_entry, asset_name):
    """Resolve actual lookback: registry.lookback - optuna_params - profile fallback."""
    if reg_entry and 'lookback' in reg_entry:
        return int(reg_entry['lookback'])
    optuna = load_json(OPTUNA_PARAMS_PATH)
    if asset_name in optuna and 'lookback' in optuna[asset_name]:
        return int(optuna[asset_name]['lookback'])
    return int(reg_entry.get('profile', {}).get('lookback', 10)) if reg_entry else 10


def detect_lookback_from_h5(h5_path):
    """Detect actual lookback from saved weights by finding the attention Dense(timesteps) square kernel."""
    import h5py

    def _find_square_kernel(group):
        """Recursively search for a square kernel tensor (NxN, 5<=N<=100)."""
        for key in group:
            item = group[key]
            if hasattr(item, 'shape'):
                if 'kernel' in key and len(item.shape) == 2 and item.shape[0] == item.shape[1] and 5 <= item.shape[0] <= 100:
                    return int(item.shape[0])
            elif hasattr(item, 'keys'):
                result = _find_square_kernel(item)
                if result is not None:
                    return result
        return None

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'model_weights' in f:
                return _find_square_kernel(f['model_weights'])
            if 'layers' in f:
                return _find_square_kernel(f['layers'])
    except Exception as e:
        logger.debug("Lookback detection failed for %s: %s", h5_path, e)
    return None


def _h5_dataset_shapes(h5_path):
    """{dataset path: shape} for every weight in a legacy HDF5 champion."""
    import h5py
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = tuple(int(d) for d in obj.shape)

    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)
    return out


def transformer_kwargs_from_h5(h5_path):
    """The two build_transformer_encoder arguments the saved weights fix.

    The rebuild paths assume the builder's CURRENT defaults, and a champion
    saved under different ones then fails on a shape mismatch - which is how
    2026-08-21 found every legacy transformer serving as nothing: the file was
    written with num_heads=2, ff_dim=96 against today's 4 and 128. Both are
    readable from the weights themselves: the attention query kernel is
    (n_features, num_heads, key_dim) and the first feed-forward kernel is
    (n_features, ff_dim). Reading them beats guessing, and a file that does not
    carry them returns nothing rather than a default that would load a wrong
    architecture full of freshly initialised weights.
    """
    try:
        shapes = _h5_dataset_shapes(h5_path)
    except Exception as exc:
        logger.debug("Transformer shape probe failed for %s: %s", h5_path, exc)
        return None
    heads = n_feat = None
    for name, shape in shapes.items():
        if "query/kernel" in name and len(shape) == 3:
            n_feat, heads = shape[0], shape[1]
            break
    if heads is None:
        return None
    # The feed-forward block is Dense(ff_dim) on a width-n_feat tensor followed
    # by Dense(n_feat), so the pair of kernels is (n_feat, F) and (F, n_feat).
    # Matching the PAIR rather than a layer called "dense" is what makes this
    # work on a real champion: Keras names layers with a global counter, so the
    # file calls them dense_24 and dense_25, and a name test finds nothing.
    two_d = [sh for sh in shapes.values() if len(sh) == 2]
    for a, b in ((a, b) for a in two_d for b in two_d):
        if a[0] == n_feat and a[1] == b[0] and b[1] == n_feat and a[1] != n_feat:
            return {"num_heads": int(heads), "ff_dim": int(a[1])}
    return None


def _rebuild_from_legacy(path, builder, lookback, n_features, **kwargs):
    """Rebuild `builder`'s architecture and load a legacy HDF5 champion into it.

    load_weights matches by topological order and RAISES on any shape
    disagreement, which is what makes this safe: the alternative failure - a
    skeleton that loads silently and serves half-initialised weights at ~0.5 -
    cannot happen quietly here.
    """
    import shutil
    if _detect_format(path) != "hdf5":
        return None
    # The file's own architecture first; the builder below is the fallback for
    # champions written before model_config was embedded.
    rebuilt = _from_embedded_config(path)
    if rebuilt is not None:
        return rebuilt
    h5_path = path.replace(".keras", ".tmp.h5")
    try:
        shutil.copy2(path, h5_path)
        detected = detect_lookback_from_h5(h5_path)
        shape = (int(detected or lookback), n_features)
        model = builder(shape, **kwargs)
        model.load_weights(h5_path)
        return model
    except Exception as exc:
        logger.debug("Legacy rebuild failed for %s: %s", path, exc)
        return None
    finally:
        if os.path.exists(h5_path):
            try:
                os.remove(h5_path)
            except OSError:
                pass


def _warn_lost(path, model):
    """Warn only when every path failed, which is when a member is truly lost.

    The old warning fired on the first attempt, so it also fired for every
    champion that then loaded fine. A message that appears on success teaches
    the reader to skip it, and the one time it means a trained member has
    silently left the ensemble it is skipped too.
    """
    if model is None and os.path.exists(path):
        logger.warning("Champion exists but no loader could read it: %s", path)
    return model


def load_tcn_model(path, lookback, n_features):
    """A TCN champion, native first and rebuilt from legacy weights after.

    load_keras_native is the whole loader today, so a champion written by the
    training environment (keras 2.10, which saves HDF5 under a .keras name) is
    simply lost on serve. The TCN builder takes no adaptive sizes, so the
    rebuild needs nothing the file does not already fix.
    """
    from core.architectures import build_tcn
    model = load_keras_native(path)
    if model is not None:
        return model
    return _warn_lost(path,
                      _rebuild_from_legacy(path, build_tcn, lookback, n_features))


def load_transformer_model(path, lookback, n_features):
    """A transformer champion, native first and rebuilt from legacy weights after."""
    from core.architectures import build_transformer_encoder
    model = load_keras_native(path)
    if model is not None:
        return model
    kwargs = transformer_kwargs_from_h5(path) if _detect_format(path) == "hdf5" else None
    if kwargs is None:
        return _warn_lost(path, None)
    return _warn_lost(path, _rebuild_from_legacy(
        path, build_transformer_encoder, lookback, n_features, **kwargs))


def load_lstm_model(lstm_path, lookback, n_features):
    """Load LSTM handling V49, V50, HDF5, ZIP, and Keras 3.x formats."""
    import shutil
    import tempfile
    import zipfile
    fmt = _detect_format(lstm_path)

    # For HDF5 files saved as .keras: need .h5 extension for load_weights
    if fmt == 'hdf5':
        h5_path = lstm_path.replace('.keras', '.tmp.h5')
        shutil.copy2(lstm_path, h5_path)
        weights_path = h5_path
    else:
        weights_path = lstm_path
        h5_path = None

    try:
        # Method 0: native Keras 3 load - the correct path for every champion
        # saved since the 2026-06 retrain (adaptive sizes live in the saved
        # config, so no rebuilt skeleton can drift from them).
        if fmt == 'zip':
            model = load_keras_native(lstm_path)
            if model is not None:
                native_lb = getattr(model, 'input_shape', (None, lookback))[1] or lookback
                return model, "DUAL (AI)", int(native_lb)

        # Auto-detect lookback from saved weights (handles frozen champions with outdated optuna params)
        detected_lb = detect_lookback_from_h5(weights_path if fmt == 'hdf5' else lstm_path)
        if detected_lb is not None and detected_lb != lookback:
            logger.debug("Lookback override for %s: %d - %d (from weights)", lstm_path, lookback, detected_lb)
            lookback = detected_lb
        input_shape = (lookback, n_features)

        # Method 0b: the architecture the FILE carries, which no guess can beat.
        # Ahead of V50/V49 on purpose: those two fit only a champion whose sizes
        # match a builder default, and a wrong-but-loadable skeleton is the one
        # failure that is silent.
        rebuilt = _from_embedded_config(lstm_path)
        if rebuilt is not None:
            native_lb = getattr(rebuilt, "input_shape", (None, lookback))[1] or lookback
            return rebuilt, "DUAL (AI)", int(native_lb)

        # Method 1: V50 architecture (192/96+ReduceSumLayer), Keras 2.x load_weights
        try:
            model = build_lstm_attention(input_shape)
            model.load_weights(weights_path)
            return model, "DUAL (AI)", lookback
        except Exception as e:
            logger.debug("V50 load failed: %s", e)

        # Method 2: V49 legacy architecture, Keras 2.x load_weights
        try:
            model = build_lstm_legacy(input_shape)
            model.load_weights(weights_path)
            return model, "DUAL (V49)", lookback
        except Exception as e:
            logger.debug("V49 load failed: %s", e)

        # Method 3: ZIP with Keras 3.x weights - extract h5, load manually
        if fmt == 'zip':
            tmpdir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(lstm_path, 'r') as z:
                    z.extractall(tmpdir)
                inner_h5 = os.path.join(tmpdir, 'model.weights.h5')
                if os.path.exists(inner_h5):
                    # Auto-detect lookback from extracted weights
                    zip_lb = detect_lookback_from_h5(inner_h5)
                    if zip_lb is not None:
                        input_shape = (zip_lb, n_features)
                    zip_lookback = zip_lb if zip_lb is not None else lookback
                    # Try V49 legacy with Keras 3.x weights
                    try:
                        model = build_lstm_legacy(input_shape)
                        _load_weights_keras3(model, inner_h5)
                        return model, "DUAL (AI)", zip_lookback
                    except Exception as e:
                        logger.debug("Keras3 V49 load failed: %s", e)
                    # Try V50 with Keras 3.x weights
                    try:
                        model = build_lstm_attention(input_shape)
                        _load_weights_keras3(model, inner_h5)
                        return model, "DUAL (AI)", zip_lookback
                    except Exception as e:
                        logger.debug("Keras 3.x load failed for %s: %s", lstm_path, e)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        _warn_lost(lstm_path, None)
        return None, "CB ONLY (Err)", lookback
    finally:
        if h5_path and os.path.exists(h5_path):
            os.remove(h5_path)
