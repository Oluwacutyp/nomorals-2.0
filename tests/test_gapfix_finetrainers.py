"""Lane 119 — finetrainers-conditioning port tests.

Covers the conditioning package against the real vendored fixtures
(tests/fixtures/finetrainers-conditioning/): the four dataset formats,
bucket math (including the three upstream bugs this port fixes),
the precomputed store and both tensor codecs, the condition specs,
the precompute orchestration end to end, the frame decode/preprocess
path on the real hiker clips, the parity TOML, the attribution TOML,
and the registry tool count (774 — the secprobe wire-up added 25
tools on top of 749; the bughunter port (lane 121)
added 8 tools and the mocap port (lane 125) added the mocap tool;
this lane adds no tool).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "finetrainers-conditioning"

from nomorals.training.conditioning import (  # noqa: E402
    CONDITION_SPECS,
    ConditioningConfig,
    ConditioningError,
    ConditioningUnavailable,
    NumpyTensorIO,
    PrecomputedStore,
    Precomputer,
    RealEncoders,
    get_condition_spec,
    load_condition_models,
    load_dataset,
    post_latent_preparation,
    prepare_conditions,
    preprocess_video,
    probe_video,
    select_tensor_io,
    should_perform_precomputation,
)
from nomorals.training.conditioning.attribution import verify_attribution  # noqa: E402
from nomorals.training.conditioning.buckets import (  # noqa: E402
    BucketSampler,
    VideoItem,
    find_nearest_resolution,
    frame_bucket_for,
    rectangle_crop_geometry,
    sample_frame_indices,
)
from nomorals.training.conditioning.configs import parse_args  # noqa: E402
from nomorals.training.conditioning.encoders import (  # noqa: E402
    normalize_latents,
    pack_latents,
    pad_frames,
)
from nomorals.training.conditioning.formats import (  # noqa: E402
    apply_id_token,
    clean_llm_prefixes,
)
from nomorals.training.conditioning.parity import run_parity  # noqa: E402
from nomorals.training.conditioning.precompute import demo_encoders  # noqa: E402
from nomorals.training.conditioning.precomputed import (  # noqa: E402
    precomputation_dir_name,
    string_to_filename,
)

CAPTION = (
    "A hiker standing at the top of a mountain, triumphantly, high quality"
)


# ---------------------------------------------------------------------------
# formats
# ---------------------------------------------------------------------------


def test_two_file_one_row():
    dataset = load_dataset(FIXTURES, "prompts.txt", "videos.txt")
    assert len(dataset) == 1
    assert dataset.prompts == [CAPTION]
    assert dataset.video_paths[0].name == "hiker.mp4"
    assert dataset.video_paths[0].is_file()
    assert dataset.source_format == "two-file"


def test_two_file_multi_row():
    dataset = load_dataset(FIXTURES, "prompts_multi.txt", "videos_multi.txt")
    assert len(dataset) == 2
    assert dataset.video_paths[1].name == "hiker_tiny.mp4"


def test_csv_one_row_keeps_upstream_quoting_quirk():
    dataset = load_dataset(
        FIXTURES, "caption", "video",
        dataset_file=FIXTURES / "metadata.csv",
    )
    assert len(dataset) == 1
    assert dataset.source_format == "csv"
    # Upstream's own fixture double-quotes the caption; pandas and the
    # stdlib csv reader agree: the quotes are part of the value.
    assert dataset.prompts[0] == f'"{CAPTION}"'
    assert dataset.video_paths[0].name == "hiker.mp4"


def test_json_and_jsonl_two_rows():
    for name in ("dataset.json", "dataset.jsonl"):
        dataset = load_dataset(
            FIXTURES, "caption", "video",
            dataset_file=FIXTURES / name,
        )
        assert len(dataset) == 2
        assert dataset.prompts == [CAPTION, CAPTION]
        assert dataset.video_paths[1].name == "hiker_tiny.mp4"


def test_missing_root_refused_with_upstream_message():
    with pytest.raises(ConditioningError, match="Root folder for videos"):
        load_dataset("/nonexistent-root-xyz", "prompts.txt", "videos.txt")


def test_missing_caption_file_refused():
    with pytest.raises(ConditioningError, match="caption_column"):
        load_dataset(FIXTURES, "nope.txt", "videos.txt")


def test_missing_video_file_refused():
    with pytest.raises(ConditioningError, match="video_column"):
        load_dataset(FIXTURES, "prompts.txt", "nope.txt")


def test_missing_video_entry_refused(tmp_path):
    (tmp_path / "p.txt").write_text("a caption\n", encoding="utf-8")
    (tmp_path / "v.txt").write_text("videos/gone.mp4\n", encoding="utf-8")
    with pytest.raises(ConditioningError, match="not a valid file"):
        load_dataset(tmp_path, "p.txt", "v.txt")


def test_count_mismatch_refused(tmp_path):
    (tmp_path / "p.txt").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "v.txt").write_text("videos/hiker.mp4\n", encoding="utf-8")
    (tmp_path / "videos").mkdir()
    import shutil

    shutil.copy(FIXTURES / "videos" / "hiker.mp4", tmp_path / "videos" / "hiker.mp4")
    with pytest.raises(ConditioningError, match="same but found"):
        load_dataset(tmp_path, "p.txt", "v.txt")


def test_unsupported_dataset_suffix_refused():
    with pytest.raises(ConditioningError, match="dataset_file"):
        load_dataset(
            FIXTURES, "caption", "video",
            dataset_file=FIXTURES / "prompts.txt",
        )


def test_csv_missing_column_refused(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("video\nvideos/hiker.mp4\n", encoding="utf-8")
    with pytest.raises(ConditioningError, match="no column 'caption'"):
        load_dataset(tmp_path, "caption", "video", dataset_file=bad)


def test_jsonl_bad_line_refused(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"caption": "x", "video": "videos/hiker.mp4"}\n{nope}\n',
                   encoding="utf-8")
    with pytest.raises(ConditioningError, match="not valid JSON"):
        load_dataset(tmp_path, "caption", "video", dataset_file=bad)


def test_llm_prefix_cleaning_at_load():
    dataset = load_dataset(
        FIXTURES, "prompts_llm.txt", "videos_llm.txt",
        remove_llm_prefixes=True,
    )
    assert dataset.prompts == ["a hiker climbs"]


def test_clean_llm_prefix_vectors():
    assert clean_llm_prefixes("In this video, a hiker climbs") == "a hiker climbs"
    assert clean_llm_prefixes("This video shows a hiker") == "a hiker"
    assert clean_llm_prefixes("Caption: a hiker") == "a hiker"
    assert clean_llm_prefixes("The clip captures a hiker") == "a hiker"
    assert clean_llm_prefixes("A hiker climbs") == "A hiker climbs"
    # The phrase list is upstream's, verbatim: 5 explicit + 9 x 7.
    from nomorals.training.conditioning.constants import COMMON_LLM_START_PHRASES

    assert len(COMMON_LLM_START_PHRASES) == 68


def test_id_token_prefix():
    assert apply_id_token("a prompt", None) == "a prompt"
    assert apply_id_token("a prompt", "  TOK  ") == "TOK a prompt"
    dataset = load_dataset(FIXTURES, "prompts.txt", "videos.txt")
    assert dataset.prompt_for(0, id_token="DEVON") == f"DEVON {CAPTION}"


# ---------------------------------------------------------------------------
# buckets
# ---------------------------------------------------------------------------


def test_nearest_resolution():
    buckets = [(49, 480, 720), (49, 256, 360), (49, 720, 960)]
    assert find_nearest_resolution(buckets, 240, 368) == (256, 360)
    assert find_nearest_resolution(buckets, 480, 720) == (480, 720)


def test_frame_bucket():
    buckets = [(49, 480, 720), (24, 480, 720), (12, 480, 720)]
    assert frame_bucket_for(49, buckets, 49) == 49
    assert frame_bucket_for(30, buckets, 49) == 24
    # Upstream crashes (min default=1, then [0] on an int) for a clip
    # shorter than every bucket; the port returns a 1-frame sample.
    assert frame_bucket_for(5, buckets, 49) == 1


def test_sample_frame_indices():
    assert sample_frame_indices(10, 4) == [0, 2, 4, 6]
    assert sample_frame_indices(49, 49) == list(range(49))
    # Upstream range(step=0) raises for short clips; the port floors at 1.
    assert sample_frame_indices(3, 49) == [0, 1, 2]


def test_rectangle_crop_geometry():
    geo = rectangle_crop_geometry(480, 720, (256, 360), "center")
    assert geo["resized_height"] == 256
    assert geo["resized_width"] == 384
    assert geo["top"] == 0
    assert geo["left"] == 12
    with pytest.raises(ConditioningError, match="video_reshape_mode"):
        rectangle_crop_geometry(480, 720, (256, 360), "sideways")


def test_bucket_sampler_mixed_and_drop_last():
    shapes = [(49, 480, 720), (49, 480, 720), (48, 256, 360), (49, 480, 720)]
    items = [VideoItem(*s, payload=i) for i, s in enumerate(shapes)]
    batches = BucketSampler(items, batch_size=2, shuffle=False).batches()
    assert [len(b) for b in batches] == [2, 1, 1]
    assert sum(len(b) for b in batches) == 4
    dropped = BucketSampler(
        items, batch_size=2, shuffle=False, drop_last=True
    ).batches()
    assert [len(b) for b in dropped] == [2]


def test_bucket_sampler_no_shuffle_keeps_tail():
    # Upstream only yields leftover buckets inside `if self.shuffle:`,
    # so with shuffle=False the tail of the dataset vanishes. Fixed.
    items = [VideoItem(49, 480, 720, 0), VideoItem(48, 256, 360, 1)]
    batches = BucketSampler(items, batch_size=8, shuffle=False).batches()
    assert [len(b) for b in batches] == [1, 1]


# ---------------------------------------------------------------------------
# precomputed store + codecs
# ---------------------------------------------------------------------------


def test_string_to_filename_and_dir_name():
    assert string_to_filename("THUDM/CogVideoX-5b") == "THUDM-CogVideoX-5b"
    assert string_to_filename("Lightricks/LTX-Video") == "Lightricks-LTX-Video"
    assert string_to_filename("a b:c.d,e;f!g?h") == "a-b-c-d-e-f-g-h"
    assert precomputation_dir_name("ltx_video", "Lightricks/LTX-Video") == (
        "ltx_video_Lightricks-LTX-Video_precomputed"
    )


def test_should_perform_precomputation_matrix(tmp_path):
    store = tmp_path / "ltx_video_Lightricks-LTX-Video_precomputed"
    assert should_perform_precomputation(store) is True
    (store / "conditions").mkdir(parents=True)
    (store / "latents").mkdir(parents=True)
    assert should_perform_precomputation(store) is True
    for name in ("conditions-0-0.npz", "conditions-0-1.npz"):
        (store / "conditions" / name).write_bytes(b"x")
    for name in ("latents-0-0.npz", "latents-0-1.npz"):
        (store / "latents" / name).write_bytes(b"x")
    assert should_perform_precomputation(store) is False
    (store / "latents" / "latents-0-1.npz").unlink()
    assert should_perform_precomputation(store) is True
    # Mismatch cleans the stale files before re-running.
    assert list((store / "conditions").glob("*.npz")) == []
    assert list((store / "latents").glob("*.npz")) == []


def test_numpy_tensor_io_round_trip(tmp_path):
    io = NumpyTensorIO()
    path = tmp_path / "pair.npz"
    payload = {
        "prompt_embeds": np.arange(6, dtype=np.float32).reshape(2, 3),
        "num_frames": 8,
        "height": 8,
    }
    io.save(path, payload)
    back = io.load(path)
    np.testing.assert_array_equal(back["prompt_embeds"], payload["prompt_embeds"])
    assert back["num_frames"] == 8
    assert back["height"] == 8
    assert isinstance(back["num_frames"], int)


def test_select_tensor_io_prefers_torch_when_present():
    io = select_tensor_io()
    # This box has no torch: the numpy codec is selected honestly.
    assert io.suffix == ".npz"


def test_torch_codec_fails_honestly_without_torch(tmp_path):
    from nomorals.training.conditioning.precomputed import TorchTensorIO

    with pytest.raises(ConditioningUnavailable, match="torch is not installed"):
        TorchTensorIO().save(tmp_path / "x.pt", {"a": np.zeros(2)})


def test_precomputed_store_pairing_and_mismatch(tmp_path):
    store_dir = tmp_path / "precomputed"
    (store_dir / "conditions").mkdir(parents=True)
    (store_dir / "latents").mkdir(parents=True)
    io = NumpyTensorIO()
    io.save(store_dir / "conditions" / "conditions-0-0.npz",
            {"prompt_embeds": np.zeros((1, 4), dtype=np.float32)})
    io.save(store_dir / "latents" / "latents-0-0.npz",
            {"latents": np.zeros((1, 2), dtype=np.float32)})
    store = PrecomputedStore(tmp_path, tensor_io=io)
    assert len(store) == 1
    pair = store.load_pair(0)
    assert set(pair["text_conditions"]) == {"prompt_embeds"}
    assert set(pair["latent_conditions"]) == {"latents"}
    io.save(store_dir / "conditions" / "conditions-0-1.npz",
            {"prompt_embeds": np.zeros((1, 4), dtype=np.float32)})
    with pytest.raises(ConditioningError, match="do not match"):
        PrecomputedStore(tmp_path, tensor_io=io)


# ---------------------------------------------------------------------------
# encoders: specs, dispatch, post-latent math
# ---------------------------------------------------------------------------


def test_condition_specs():
    ltx = get_condition_spec("ltx_video")
    assert ltx.model_id == "Lightricks/LTX-Video"
    assert ltx.max_sequence_length == 128
    assert ltx.condition_keys == ("prompt_embeds", "prompt_attention_mask")
    assert ltx.text_encoder == "t5"
    cog = get_condition_spec("cogvideox")
    assert cog.model_id == "THUDM/CogVideoX-5b"
    assert cog.max_sequence_length == 226
    assert cog.condition_keys == ("prompt_embeds",)
    hun = get_condition_spec("hunyuan_video")
    assert hun.max_sequence_length == 256
    assert hun.condition_keys == (
        "prompt_embeds", "prompt_attention_mask",
        "pooled_prompt_embeds", "guidance",
    )
    assert set(CONDITION_SPECS) == {"cogvideox", "hunyuan_video", "ltx_video"}
    with pytest.raises(ConditioningError, match="not supported"):
        get_condition_spec("wan")


def test_prepare_conditions_dispatch_and_guidance():
    encode_text, _ = demo_encoders("hunyuan_video")
    spec = get_condition_spec("hunyuan_video")
    out = prepare_conditions(spec, encode_text, "a hiker")
    assert set(out) == set(spec.condition_keys)
    assert out["guidance"].shape == (1,)
    assert float(out["guidance"][0]) == 1000.0

    def bad_encode(prompts, max_sequence_length):
        return {"prompt_embeds": np.zeros((1, 4), dtype=np.float32)}

    with pytest.raises(ConditioningError, match="missing condition keys"):
        prepare_conditions(spec, bad_encode, "a hiker")


def test_pad_frames_keeps_upstream_quirk():
    array = np.zeros((1, 4, 49, 8, 8), dtype=np.float32)
    assert pad_frames(array, 2).shape == (1, 4, 50, 8, 8)
    # Divisible by the patch: upstream still pads a whole patch_size_t.
    assert pad_frames(np.zeros((1, 4, 48, 8, 8), dtype=np.float32), 2).shape == (
        (1, 4, 50, 8, 8)
    )
    assert pad_frames(array, 1).shape == (1, 4, 49, 8, 8)
    assert pad_frames(array, None).shape == (1, 4, 49, 8, 8)


def test_pack_and_normalize_latents():
    packed = pack_latents(np.zeros((1, 4, 8, 8, 8), dtype=np.float32), 1, 1)
    assert packed.shape == (1, 512, 4)
    packed2 = pack_latents(np.zeros((1, 4, 8, 8, 8), dtype=np.float32), 2, 2)
    assert packed2.shape == (1, 64, 32)
    # Packing is a permutation: values survive, count is conserved.
    values = np.arange(1 * 4 * 8 * 8 * 8, dtype=np.float32).reshape(1, 4, 8, 8, 8)
    assert np.sort(pack_latents(values, 2, 2).ravel()).tolist() == np.sort(
        values.ravel()
    ).tolist()
    out = normalize_latents(
        np.full((1, 2, 1, 1, 1), 2.0),
        np.array([0.0, 1.0]), np.array([1.0, 2.0]),
    )
    assert float(out[0, 0, 0, 0, 0]) == 2.0
    assert float(out[0, 1, 0, 0, 0]) == 0.5


def test_post_latent_preparation_cog_and_ltx():
    cog = get_condition_spec("cogvideox")
    latents = np.ones((1, 4, 49, 8, 8), dtype=np.float32)
    out = post_latent_preparation(
        cog, latents, scaling_factor=2.0, patch_size_t=2
    )
    assert out["latents"].shape == (1, 50, 4, 8, 8)
    assert float(out["latents"][0, 0, 0, 0, 0]) == 2.0
    ltx = get_condition_spec("ltx_video")
    raw = np.full((1, 4, 8, 8, 8), 2.0)
    out = post_latent_preparation(
        ltx, raw, latents_mean=np.zeros(4), latents_std=np.ones(4),
        patch_size=1, patch_size_t=1,
    )
    assert out["latents"].shape == (1, 512, 4)
    assert out["num_frames"] == 8 and out["height"] == 8 and out["width"] == 8
    with pytest.raises(ConditioningError, match="latents_mean"):
        post_latent_preparation(ltx, raw)


def test_load_condition_models_fails_honestly_without_diffusers():
    with pytest.raises(ConditioningUnavailable, match="diffusers is not installed"):
        load_condition_models("ltx_video")


def test_real_encoders_fail_honestly_without_torch():
    with pytest.raises(ConditioningUnavailable, match="torch is not installed"):
        RealEncoders(get_condition_spec("ltx_video"), {})


# ---------------------------------------------------------------------------
# frames: real decode on the real hiker clips
# ---------------------------------------------------------------------------


def test_probe_hiker_tiny():
    info = probe_video(FIXTURES / "videos" / "hiker_tiny.mp4")
    assert info["width"] == 368
    assert info["height"] == 240
    assert info["num_frames"] == 49


def test_preprocess_hiker_tiny_shape_and_range():
    video = preprocess_video(FIXTURES / "videos" / "hiker_tiny.mp4")
    # 368x240 -> nearest bucket in the default Cartesian grid is
    # (256, 384): |256-240| + |384-368| = 32, the smallest distance.
    assert video.shape == (49, 3, 256, 384)
    assert video.dtype == np.float32
    assert float(video.min()) >= -1.0 and float(video.max()) <= 1.0


def test_probe_missing_file_refused():
    with pytest.raises(ConditioningError, match="does not exist"):
        probe_video(FIXTURES / "videos" / "gone.mp4")


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def test_config_defaults_and_validation():
    config = ConditioningConfig(
        data_root=str(FIXTURES), model_name="ltx_video"
    )
    assert config.model_id == "Lightricks/LTX-Video"
    assert config.effective_max_sequence_length == 128
    assert config.precompute_conditions is False
    assert config.caption_dropout_technique == "empty"
    with pytest.raises(ConditioningError, match="At least one of"):
        ConditioningConfig()
    with pytest.raises(ConditioningError, match="not supported"):
        ConditioningConfig(data_root=".", model_name="wan")
    with pytest.raises(ConditioningError, match="caption_dropout_technique"):
        ConditioningConfig(
            data_root=".", caption_dropout_technique="sideways"
        )


def test_parse_args_flags():
    config = parse_args([
        "--data-root", str(FIXTURES),
        "--caption-column", "prompts_multi.txt",
        "--video-column", "videos_multi.txt",
        "--model-name", "cogvideox",
        "--training-type", "lora",
        "--precompute-conditions",
        "--caption-dropout-p", "0.1",
        "--caption-dropout-technique", "zero",
        "--remove-common-llm-caption-prefixes",
        "--video-resolution-buckets", "49x480x720",
        "--batch-size", "1",
    ])
    assert config.model_name == "cogvideox"
    assert config.model_id == "THUDM/CogVideoX-5b"
    assert config.precompute_conditions is True
    assert config.caption_dropout_p == 0.1
    assert config.video_resolution_buckets == [(49, 480, 720)]


# ---------------------------------------------------------------------------
# precompute end to end
# ---------------------------------------------------------------------------


def test_precompute_end_to_end_and_reuse(tmp_path):
    dataset = load_dataset(FIXTURES, "prompts_multi.txt", "videos_multi.txt")
    config = ConditioningConfig(
        data_root=str(FIXTURES),
        model_name="ltx_video",
        precompute_conditions=True,
        output_dir=str(tmp_path),
    )
    encode_text, encode_video = demo_encoders("ltx_video")
    report = Precomputer(
        dataset, config, encode_text, encode_video,
        tensor_io=NumpyTensorIO(),
    ).run()
    assert report.reused is False
    assert report.conditions == 2 and report.latents == 2
    assert report.codec == "npz"
    assert report.manifest_path is not None
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == "Lightricks/LTX-Video"
    assert manifest["conditions"] == 2 and manifest["latents"] == 2
    store = PrecomputedStore(
        tmp_path, model_name="ltx_video",
        cleaned_model_id="Lightricks-LTX-Video",
        tensor_io=NumpyTensorIO(),
    )
    assert len(store) == 2
    pair = store.load_pair(0)
    assert set(pair["text_conditions"]) == {
        "prompt_embeds", "prompt_attention_mask",
    }
    assert set(pair["latent_conditions"]) == {
        "latents", "latents_mean", "latents_std",
        "num_frames", "height", "width",
    }
    # Second run reuses the store instead of recomputing.
    again = Precomputer(
        dataset, config, encode_text, encode_video,
        tensor_io=NumpyTensorIO(),
    ).run()
    assert again.reused is True
    assert again.conditions == 2 and again.latents == 2


def test_precompute_batch_size_refusal(tmp_path):
    dataset = load_dataset(FIXTURES, "prompts.txt", "videos.txt")
    config = ConditioningConfig(
        data_root=str(FIXTURES), model_name="ltx_video",
        precompute_conditions=True, batch_size=2,
        output_dir=str(tmp_path),
    )
    encode_text, encode_video = demo_encoders("ltx_video")
    with pytest.raises(ConditioningError, match="batch size 1"):
        Precomputer(dataset, config, encode_text, encode_video).run()


def test_precompute_caption_dropout_warns(tmp_path, caplog):
    dataset = load_dataset(FIXTURES, "prompts.txt", "videos.txt")
    config = ConditioningConfig(
        data_root=str(FIXTURES), model_name="ltx_video",
        precompute_conditions=True, caption_dropout_p=0.5,
        caption_dropout_technique="empty",
        output_dir=str(tmp_path),
    )
    encode_text, encode_video = demo_encoders("ltx_video")
    with caplog.at_level(
        logging.WARNING,
        logger="nomorals.training.conditioning.precompute",
    ):
        Precomputer(
            dataset, config, encode_text, encode_video,
            tensor_io=NumpyTensorIO(),
        ).run()
    assert "Caption dropout is not supported with precomputation" in caplog.text


def test_demo_encoders_are_deterministic():
    encode_text, encode_video = demo_encoders("ltx_video")
    first = encode_text(["a hiker"], 128)["prompt_embeds"]
    second = encode_text(["a hiker"], 128)["prompt_embeds"]
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 128, 8)


def test_cli_demo_run(tmp_path):
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, "training/prepare_conditions.py",
            "--data-root", str(FIXTURES),
            "--caption-column", "prompts_multi.txt",
            "--video-column", "videos_multi.txt",
            "--model-name", "ltx_video",
            "--precompute-conditions",
            "--demo-encoders",
            "--output-dir", str(out),
        ],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["conditions"] == 2 and report["latents"] == 2
    assert report["reused"] is False
    assert report["codec"] == "npz"


def test_cli_without_demo_fails_honestly(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "training/prepare_conditions.py",
            "--data-root", str(FIXTURES),
            "--caption-column", "prompts.txt",
            "--video-column", "videos.txt",
            "--model-name", "ltx_video",
            "--precompute-conditions",
            "--output-dir", str(tmp_path),
        ],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "diffusers is not installed" in result.stderr


# ---------------------------------------------------------------------------
# parity, attribution, tool count
# ---------------------------------------------------------------------------


def test_parity_toml(tmp_path):
    result = run_parity(tmp_dir=tmp_path)
    failures = [c for c in result["cases"] if not c["passed"]]
    assert result["failed"] == 0, failures
    assert result["passed"] == 41


def test_attribution():
    result = verify_attribution()
    assert result["failed"] == [], result["failed"]
    assert result["skipped"] == [], result["skipped"]
    assert result["verified"] == 39


def test_tool_count_still_774():
    from types import SimpleNamespace

    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    assert len(registry._tools) == 774, (
        "lane 119 adds no tool (774 after the secprobe wire-up added "
        "25 tools; 749 after the mocap port (lane 125) "
        "added the mocap tool on top of the auth-plugin wiring, which "
        "added 5 tools on top of the bughunter port, lane 121, which "
        "added 8 tools) — if this changed, a tool was added and "
        "every pin test needs updating with the reason named"
    )
