import argparse
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from data_prep.nmfs_csl import extract_poses as tool
from data_prep.wlasl.extract_mmpose import build_pose_pkl


class PoseExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.lists = self.base / "lists"
        self.lists.mkdir()
        (self.lists / "trainlist01.txt").write_text("0/a.mp4 0\n1/b.mp4 1\n")
        (self.lists / "testlist01.txt").write_text("0/c.mp4\n1/d.mp4\n")
        self.root = self.base / "frames"
        for clip in ("0/a", "1/b", "0/c", "1/d"):
            folder = self.root / clip
            folder.mkdir(parents=True)
            for i in range(1, 4):
                Image.new("RGB", (16, 12)).save(folder / f"image_{i:05d}.jpg")

    @staticmethod
    def inferencer(frame):
        # Different people in the same image: select the higher mean confidence.
        yield {"predictions": [[
            {"keypoints": np.ones((133, 2)).tolist(), "keypoint_scores": [0.2] * 133},
            {"keypoints": (np.ones((133, 2)) * 5).tolist(), "keypoint_scores": [0.9] * 133},
        ]]}

    def data(self):
        frames = tool.checked_frames(self.root, {"clip": "0/a"})
        return frames, tool.pose_data(frames, self.inferencer, "manifest", "protocol")

    def test_original_labels_and_test_path_labels(self):
        rows = tool.read_splits(self.lists, classes=2)
        self.assertEqual([r["clip"] for r in rows], ["0/a", "0/c", "1/b", "1/d"])
        self.assertEqual([r["label"] for r in rows], [0, 0, 1, 1])

    def test_overlap_duplicates_wrong_labels_and_missing_classes_rejected(self):
        for text in ("0/a.mp4\n1/b.mp4\n", "0/c.mp4\n0/c.mp4\n1/d.mp4\n", "0/c.mp4\n"):
            (self.lists / "testlist01.txt").write_text(text)
            with self.assertRaises(ValueError):
                tool.read_splits(self.lists, classes=2)
        (self.lists / "trainlist01.txt").write_text("0/a.mp4 1\n1/b.mp4 1\n")
        with self.assertRaises(ValueError):
            tool.read_splits(self.lists, classes=2)

    def test_balanced_disjoint_deterministic_shards(self):
        rows = [{"clip": str(i), "frames": n} for i, n in enumerate([50, 40, 30, 20, 10, 5])]
        result = tool.balance(rows, 3)
        self.assertEqual(result, tool.balance(list(reversed(rows)), 3))
        self.assertEqual(len(set(r["clip"] for r in result)), len(rows))
        totals = [sum(r["frames"] for r in result if r["shard"] == i) for i in range(3)]
        self.assertLessEqual(max(totals) - min(totals), 5)

    def test_frame_names_and_counts_must_match(self):
        self.assertEqual(len(tool.checked_frames(self.root, {"clip": "0/a", "frames": 3})), 3)
        for row in ({"clip": "missing"}, {"clip": "0/a", "frames": 4}):
            with self.assertRaises((ValueError, FileNotFoundError)):
                tool.checked_frames(self.root, row)
        (self.root / "0/a/image_00002.jpg").unlink()
        with self.assertRaises(ValueError):
            tool.checked_frames(self.root, {"clip": "0/a"})

    def test_pose_values_identical_to_wlasl_only_sentinel_differs(self):
        frames, data = self.data()
        original = build_pose_pkl(frames, self.inferencer)
        np.testing.assert_array_equal(data["keypoints"], original["keypoints"])
        self.assertEqual(data["img_list"], original["img_list"][:-1])
        self.assertEqual(data["keypoints"].dtype, np.float32)
        np.testing.assert_array_equal(data["keypoints"][:, :, :2], 5)
        self.assertTrue(np.all(data["keypoints"][:, :, 2] == np.float32(0.9)))

    def test_no_detection_preserves_legacy_zero_fallback(self):
        def empty(frame):
            yield {"predictions": [[]]}
        frames, _ = self.data()
        data = tool.pose_data(frames, empty, "manifest", "protocol")
        self.assertEqual(data["keypoints"].shape, (3, 133, 3))
        self.assertEqual(float(data["keypoints"].sum()), 0)

    def test_validation_rejects_broken_values_alignment_and_provenance(self):
        frames, data = self.data()
        names = [p.name for p in frames]
        for key, value in (
            ("keypoints", np.zeros((2, 133, 3), np.float32)),
            ("keypoints", data["keypoints"].astype(np.float64)),
            ("keypoints", np.full((3, 133, 3), np.nan, np.float32)),
            ("img_list", list(reversed(names))),
            ("extraction_id", "different"), ("protocol_id", "different"),
        ):
            with self.assertRaises(ValueError):
                tool.validate_pose(dict(data, **{key: value}), names, "manifest", "protocol")

    def test_atomic_failed_replace_keeps_old_file_and_removes_temp(self):
        path = self.base / "pose.pkl"
        tool.atomic_bytes(path, b"old")
        with patch.object(tool.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                tool.atomic_bytes(path, b"new")
        self.assertEqual(path.read_bytes(), b"old")
        self.assertFalse(list(self.base.glob("*.tmp")))
        tool.atomic_bytes(path, b"new")
        self.assertEqual(path.read_bytes(), b"new")

    def test_manifest_tamper_and_changed_assets_rejected(self):
        asset = self.base / "weights"
        asset.write_bytes(b"a")
        manifest = {"assets": {str(asset): tool.digest(asset)}}
        manifest["id"] = tool.object_hash(manifest)
        path = self.base / "manifest.json"
        tool.write_json(path, manifest)
        self.assertEqual(tool.load_plan(path), manifest)
        asset.write_bytes(b"b")
        with self.assertRaises(ValueError):
            tool.load_plan(path)
        manifest["assets"] = {}
        tool.write_json(path, manifest)
        with self.assertRaises(ValueError):
            tool.load_plan(path)

    def test_same_shard_lock_fails_and_releases(self):
        path = self.base / "lock"
        with tool.lock(path):
            with self.assertRaises(RuntimeError):
                with tool.lock(path):
                    pass
        with tool.lock(path):
            pass

    def manifest(self):
        return {"id": "manifest", "frames_root": str(self.root),
                "lists_root": str(self.lists), "out_root": str(self.base / "ready"),
                "shards": 3, "items": [
                    {"clip": "0/a", "frames": 3, "shard": 0},
                    {"clip": "0/c", "frames": 3, "shard": 1}]}

    def run_worker(self, manifest, inferencer=None):
        args = argparse.Namespace(manifest=self.base / "unused", shard=0, limit=1)
        with patch.object(tool, "load_plan", return_value=manifest), \
             patch.object(tool, "init_model", return_value=(inferencer or self.inferencer, "protocol")), \
             patch("torch.cuda.get_device_name", return_value="test GPU"), \
             contextlib.redirect_stdout(io.StringIO()):
            tool.run(args)

    def test_worker_resume_no_reinference_and_layout(self):
        manifest = self.manifest()
        self.run_worker(manifest)
        root = Path(manifest["out_root"])
        self.assertEqual((root / "jpg_video").resolve(), self.root)
        self.assertEqual((root / "traintestlist").resolve(), self.lists)
        output = root / "Joint_2D_mmpose/0/a.pkl"
        before = output.read_bytes()
        def forbidden(frame):
            raise AssertionError("Resume must not infer completed clip")
        self.run_worker(manifest, forbidden)
        self.assertEqual(output.read_bytes(), before)
        summary = tool.read_json(root / "summary-0-pilot.json")
        self.assertEqual(summary["skipped"], 1)
        self.assertFalse((root / "Joint_2D_mmpose/0/c.pkl").exists())

    def test_failure_log_then_retry(self):
        manifest = self.manifest()
        def failed(frame):
            raise RuntimeError("GPU error")
        with self.assertRaises(SystemExit):
            self.run_worker(manifest, failed)
        root = Path(manifest["out_root"])
        failure = json.loads((root / "failures-0.jsonl").read_text())
        self.assertEqual(failure["clip"], "0/a")
        self.assertFalse((root / "Joint_2D_mmpose/0/a.pkl").exists())
        self.run_worker(manifest)
        self.assertEqual(tool.read_json(root / "summary-0-pilot.json")["failed"], 0)

    def test_invalid_existing_pose_not_silently_skipped_or_overwritten(self):
        manifest = self.manifest()
        self.run_worker(manifest)
        path = Path(manifest["out_root"]) / "Joint_2D_mmpose/0/a.pkl"
        path.write_bytes(b"broken")
        with self.assertRaises(SystemExit):
            self.run_worker(manifest)
        self.assertEqual(path.read_bytes(), b"broken")

    def test_actual_nmfs_loader_reads_equal_length_output(self):
        from feeder.single_dataset.NMFs_CSL import NMFs_CSL
        manifest = self.manifest()
        self.run_worker(manifest)
        loader = NMFs_CSL(manifest["out_root"], use_cache=False)
        data, _ = loader.extractvideoinfo(0, "0/a.mp4")
        self.assertEqual(len(data["img_list"]), data["keypoints"].shape[0])
        self.assertEqual(data["keypoints"].shape, (3, 133, 3))

    def test_resolved_models_versions_and_weights_recorded(self):
        import torch
        pose = torch.nn.Linear(2, 2)
        detector = torch.nn.Linear(2, 2)
        detector.cfg = SimpleNamespace(pretty_text="resolved detector")
        inner = SimpleNamespace(model=pose, detector=SimpleNamespace(model=detector),
                                cfg=SimpleNamespace(pretty_text="resolved pose"), det_cat_ids=(0,))
        versions = {"torch": "1.8.1+cu111", "numpy": "1.22.4", "mmcv": "2.1.0",
                    "mmengine": "0.10.7", "mmdet": "3.2.0", "mmpose": "1.3.2", "cv2": "5.0.0"}
        def package(name):
            return SimpleNamespace(__version__=versions[name])
        with patch.object(tool.importlib, "import_module", side_effect=package):
            first = tool.model_metadata(SimpleNamespace(inferencer=inner), {"id": "manifest"})
            second = tool.model_metadata(SimpleNamespace(inferencer=inner), {"id": "manifest"})
            self.assertEqual(first, second)
            self.assertEqual(first["resolved_detector_config"], "resolved detector")
            with torch.no_grad():
                detector.weight.add_(1)
            third = tool.model_metadata(SimpleNamespace(inferencer=inner), {"id": "manifest"})
            self.assertNotEqual(first["detector_tensor_sha256"], third["detector_tensor_sha256"])
            versions["mmpose"] = "different"
            with self.assertRaises(ValueError):
                tool.model_metadata(SimpleNamespace(inferencer=inner), {"id": "manifest"})

    def test_only_transient_top_level_visualizer_name_is_normalized(self):
        first = ("model = dict(name='visualizer-123.1')\n"
                 "visualizer = dict(\n    name='visualizer-1794002750.102702',\n"
                 "    type='DetLocalVisualizer',\n    radius=3)\n")
        second = first.replace("1794002750.102702", "1794002848.606203")
        normalized = tool.stable_detector_config(first)
        self.assertEqual(normalized, tool.stable_detector_config(second))
        self.assertIn("model = dict(name='visualizer-123.1')", normalized)
        self.assertNotEqual(normalized, tool.stable_detector_config(second.replace("radius=3", "radius=4")))
        self.assertNotEqual(normalized, tool.stable_detector_config(second.replace("DetLocalVisualizer", "Other")))
        self.assertEqual(tool.stable_detector_config(normalized), normalized)

    def test_overlay_writes_only_preview_images(self):
        manifest = self.manifest()
        before = tool.digest(self.root / "0/a/image_00001.jpg")
        args = argparse.Namespace(manifest="unused", clip="0/a")
        with patch.object(tool, "load_plan", return_value=manifest), \
             patch.object(tool, "init_model", return_value=(self.inferencer, "protocol")), \
             contextlib.redirect_stdout(io.StringIO()):
            tool.overlay(args)
        images = list((Path(manifest["out_root"]) / "overlays/0/a").glob("*.jpg"))
        self.assertEqual(len(images), 3)
        self.assertEqual(tool.digest(self.root / "0/a/image_00001.jpg"), before)

    def test_audit_reports_missing_then_accepts_complete_outputs(self):
        manifest = self.manifest()
        out = Path(manifest["out_root"])
        metadata = {"manifest_id": "manifest"}
        tool.write_json(out / "extraction_protocol.json", metadata)
        protocol_id = tool.object_hash(metadata)
        args = argparse.Namespace(manifest="unused")
        with patch.object(tool, "load_plan", return_value=manifest), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                tool.audit(args)
            self.assertEqual(len(tool.read_json(out / "pose_audit.json")["issues"]), 2)
            for row in manifest["items"]:
                frames = tool.checked_frames(self.root, row)
                data = tool.pose_data(frames, self.inferencer, "manifest", protocol_id)
                tool.atomic_bytes(out / "Joint_2D_mmpose" / (row["clip"] + ".pkl"),
                                  pickle.dumps(data))
            tool.audit(args)
        self.assertTrue(tool.read_json(out / "pose_audit.json")["passed"])


if __name__ == "__main__":
    unittest.main()
