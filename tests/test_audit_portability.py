"""Frozen evidence must replay from a clone without trusting an old checkout."""

import hashlib

import pytest

from benchmarks import audit_step2_spans


@pytest.mark.parametrize("old_path", [
    r"Z:\unavailable\project\benchmarks\results\prior\scores.jsonl",
    "/unavailable/project/benchmarks/results/prior/scores.jsonl",
])
def test_exclusion_checkpoint_relocated_and_hash_verified(tmp_path, old_path):
    checkpoint = tmp_path / "benchmarks/results/prior/scores.jsonl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'{"group_id": "prior-source"}\n')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "arguments": {"exclude_groups_from": [old_path]},
        "exclusion_files": {old_path: digest},
    }

    paths, hashes = audit_step2_spans._checked_exclusion_files(manifest, tmp_path)
    assert paths == [str(checkpoint)]
    assert hashes == {old_path: digest}

    # Moving a checkpoint must not make an altered source-group list acceptable.
    checkpoint.write_bytes(b'{"group_id": "different-source"}\n')
    with pytest.raises(ValueError, match="exclusion checkpoint hashes"):
        audit_step2_spans._checked_exclusion_files(manifest, tmp_path)


def test_exclusion_replay_uses_clone_even_when_old_checkout_exists(tmp_path):
    old_checkpoint = tmp_path / "old/benchmarks/results/prior/scores.jsonl"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_bytes(b'original')
    clone = tmp_path / "clone"
    cloned_checkpoint = clone / "benchmarks/results/prior/scores.jsonl"
    cloned_checkpoint.parent.mkdir(parents=True)
    cloned_checkpoint.write_bytes(b'altered')
    old_path = str(old_checkpoint)
    manifest = {
        "arguments": {"exclude_groups_from": [old_path]},
        "exclusion_files": {old_path: hashlib.sha256(b'original').hexdigest()},
    }
    with pytest.raises(ValueError, match="exclusion checkpoint hashes"):
        audit_step2_spans._checked_exclusion_files(manifest, clone)
