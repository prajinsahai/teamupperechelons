"""Every tool of every track, on every sample business, returns strict JSON the model can use.

No LLM: this only exercises the deterministic tool layer (src/tracks/registry.py).
"""

import json

import pytest

from src.data.ingest import bundle_from_paths
from src.data.samples import list_samples, sample_files
from src.tracks.registry import TRACKS

SAMPLES = list_samples()
DEFAULT_ARGS = {"search_policy": {"query": "waiting period"}}


@pytest.fixture(scope="module")
def bundles():
    return {slug: bundle_from_paths(sample_files(slug)) for slug in SAMPLES}


def _no_private_keys(obj):
    if isinstance(obj, dict):
        assert not any(str(k).startswith("_") for k in obj), f"private key leaked: {list(obj)}"
        for v in obj.values():
            _no_private_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            _no_private_keys(v)


@pytest.mark.parametrize("slug", list(SAMPLES))
@pytest.mark.parametrize("track_key", list(TRACKS))
def test_every_tool_returns_strict_json_with_method(bundles, slug, track_key):
    bundle = bundles[slug]
    params = dict(SAMPLES[slug]["programme"])
    track = TRACKS[track_key]
    tools = track.build_tools(bundle, params)
    assert len(tools) >= 4
    for t in tools:
        raw = t.invoke(DEFAULT_ARGS.get(t.name, {}))
        assert isinstance(raw, str)
        # strict: json.loads accepts NaN by default, so forbid it explicitly
        data = json.loads(raw, parse_constant=lambda c: pytest.fail(f"{t.name} emitted {c}"))
        assert isinstance(data, dict), f"{t.name} must return a JSON object"
        if t.name != "describe_loaded_data":
            assert "method" in data or "hits" in data, f"{t.name} lacks a `method` key"
        _no_private_keys(data)
        assert len(raw) < 60_000, f"{t.name} payload too large for a prompt: {len(raw)} chars"


@pytest.mark.parametrize("slug", list(SAMPLES))
def test_metadata_is_flat_and_json_safe(bundles, slug):
    params = dict(SAMPLES[slug]["programme"])
    for key, track in TRACKS.items():
        if track.needs == "documents" and not bundles[slug].documents:
            continue
        meta = track.metadata(bundles[slug], params)
        json.dumps(meta, allow_nan=False)
        assert all(not isinstance(v, (dict, list)) for v in meta.values()), f"{key} metadata must be flat"


def test_expected_loss_keys_do_not_collide(bundles):
    """Actuary owns computed_expected_loss (gross analytic); Capital reports the net simulated mean under its own key."""
    slug = "nexus"
    params = dict(SAMPLES[slug]["programme"])
    a = TRACKS["actuary"].metadata(bundles[slug], params)
    c = TRACKS["capital"].metadata(bundles[slug], params)
    assert "computed_expected_loss" in a and "computed_expected_loss" not in c
    assert "computed_expected_loss_net" in c
