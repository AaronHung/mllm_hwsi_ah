from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from Ours.rcc_k16_protocol import PROMPTS, append_record
from Ours.run_rcc_pilot20_k16 import _write_artifacts, load_features, run


class FeatureFixtureMixin:
    slide_id = "slide-valid"

    def write_features(
        self,
        root: Path,
        *,
        slide_id: str | None = None,
        wsi: torch.Tensor | None = None,
        region: torch.Tensor | None = None,
        patch: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        patch_indices: torch.Tensor | None = None,
        cell_indices: torch.Tensor | None = None,
    ) -> tuple[Path, Path]:
        slide_id = slide_id or self.slide_id
        feature_root = root / "features"
        cell_root = root / "cells"
        for directory in (
            feature_root / "wsi",
            feature_root / "region_4k",
            feature_root / "patches_filtered",
            cell_root / slide_id,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        wsi = torch.ones(192) if wsi is None else wsi
        region = torch.ones(3, 192) if region is None else region
        patch = torch.ones(3, 16, 512) if patch is None else patch
        cell = torch.ones(3, 16, 384) if cell is None else cell
        patch_indices = (
            torch.arange(48).reshape(3, 16)
            if patch_indices is None
            else patch_indices
        )
        cell_indices = patch_indices.clone() if cell_indices is None else cell_indices

        torch.save(wsi, feature_root / "wsi" / f"{slide_id}.pt")
        torch.save(region, feature_root / "region_4k" / f"{slide_id}.pt")
        torch.save(
            {
                "selected_features": patch,
                "selected_indices": patch_indices,
            },
            feature_root / "patches_filtered" / f"{slide_id}.pt",
        )
        torch.save(
            {
                "encoded_cell_features": cell,
                "selected_patch_cell_level_indices": cell_indices,
            },
            cell_root / slide_id / "encoded_cell_features.pt",
        )
        return feature_root, cell_root


class LoadFeaturesTests(FeatureFixtureMixin, unittest.TestCase):
    def assert_load_error(
        self,
        root: Path,
        level: str,
        *,
        slide_id: str | None = None,
    ) -> str:
        slide_id = slide_id or self.slide_id
        with self.assertRaises(Exception) as context:
            load_features(slide_id, root / "features", root / "cells")
        message = str(context.exception)
        self.assertIn(slide_id, message)
        self.assertIn(level, message.lower())
        return message

    def test_valid_r3_fixture_returns_exact_batched_shapes_and_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_root, cell_root = self.write_features(Path(directory))

            features = load_features(self.slide_id, feature_root, cell_root)

        self.assertEqual(set(features), {"wsi", "region", "patch", "cell"})
        self.assertEqual(features["wsi"].shape, (1, 192))
        self.assertEqual(features["region"].shape, (1, 3, 192))
        self.assertEqual(features["patch"].shape, (1, 3, 16, 512))
        self.assertEqual(features["cell"].shape, (1, 3, 16, 384))
        self.assertTrue(all(value.dtype == torch.float32 for value in features.values()))

    def test_missing_file_names_slide_and_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(root)
            (root / "features" / "wsi" / f"{self.slide_id}.pt").unlink()

            self.assert_load_error(root, "wsi")

    def test_missing_payload_keys_name_slide_and_level(self) -> None:
        cases = (
            (
                "patch",
                "patches_filtered",
                {"selected_indices": torch.arange(48).reshape(3, 16)},
            ),
            (
                "cell",
                "cell",
                {
                    "selected_patch_cell_level_indices": torch.arange(48).reshape(
                        3, 16
                    )
                },
            ),
        )
        for level, location, payload in cases:
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                feature_root, cell_root = self.write_features(root)
                if location == "cell":
                    path = cell_root / self.slide_id / "encoded_cell_features.pt"
                else:
                    path = feature_root / location / f"{self.slide_id}.pt"
                torch.save(payload, path)

                self.assert_load_error(root, level)

    def test_wrong_wsi_dimension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(root, wsi=torch.ones(191))
            self.assert_load_error(root, "wsi")

    def test_zero_regions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(
                root,
                region=torch.ones(0, 192),
                patch=torch.ones(0, 16, 512),
                cell=torch.ones(0, 16, 384),
                patch_indices=torch.empty(0, 16, dtype=torch.long),
                cell_indices=torch.empty(0, 16, dtype=torch.long),
            )
            self.assert_load_error(root, "region")

    def test_region_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(root, region=torch.ones(2, 192))
            self.assert_load_error(root, "patch")

    def test_patch_k_other_than_16_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(
                root,
                patch=torch.ones(3, 15, 512),
                patch_indices=torch.arange(45).reshape(3, 15),
                cell_indices=torch.arange(45).reshape(3, 15),
            )
            self.assert_load_error(root, "patch")

    def test_patch_dimension_other_than_512_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(root, patch=torch.ones(3, 16, 511))
            self.assert_load_error(root, "patch")

    def test_cell_dimension_other_than_384_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(root, cell=torch.ones(3, 16, 383))
            self.assert_load_error(root, "cell")

    def test_index_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_features(
                root,
                patch_indices=torch.arange(45).reshape(3, 15),
                cell_indices=torch.arange(45).reshape(3, 15),
            )
            self.assert_load_error(root, "patch/cell indices")

    def test_patch_and_cell_indices_must_be_equal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch_indices = torch.arange(48).reshape(3, 16)
            cell_indices = patch_indices.clone()
            cell_indices[0, 0] = 99
            self.write_features(
                root,
                patch_indices=patch_indices,
                cell_indices=cell_indices,
            )
            self.assert_load_error(root, "patch/cell indices")

    def test_fractional_indices_are_rejected_before_lossy_cast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = torch.arange(48).reshape(3, 16).float()
            self.write_features(
                root,
                patch_indices=base + 0.1,
                cell_indices=base + 0.9,
            )
            self.assert_load_error(root, "patch/cell indices")

    def test_boolean_indices_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indices = torch.zeros(3, 16, dtype=torch.bool)
            self.write_features(
                root,
                patch_indices=indices,
                cell_indices=indices.clone(),
            )
            self.assert_load_error(root, "patch/cell indices")

    def test_patch_and_cell_index_dtypes_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = torch.arange(48).reshape(3, 16)
            self.write_features(
                root,
                patch_indices=values.to(torch.int32),
                cell_indices=values.to(torch.int64),
            )
            self.assert_load_error(root, "patch/cell indices")

    def test_nan_and_inf_at_each_feature_level_are_rejected(self) -> None:
        base = {
            "wsi": lambda: torch.ones(192),
            "region": lambda: torch.ones(3, 192),
            "patch": lambda: torch.ones(3, 16, 512),
            "cell": lambda: torch.ones(3, 16, 384),
        }
        for level in base:
            for nonfinite in (float("nan"), float("inf")):
                with (
                    self.subTest(level=level, nonfinite=nonfinite),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    value = base[level]()
                    value.reshape(-1)[0] = nonfinite
                    self.write_features(root, **{level: value})
                    self.assert_load_error(root, level)


class DeterministicFakeModel:
    def __init__(
        self,
        failures_remaining: int = 0,
        answer: str = "clear cell renal cell carcinoma",
    ) -> None:
        self.failures_remaining = failures_remaining
        self.answer = answer
        self.prompts: list[str] = []

    def generate_from_features(
        self,
        *,
        wsi: torch.Tensor,
        region: torch.Tensor,
        patch: torch.Tensor,
        cell: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
    ) -> str:
        self.prompts.append(prompt)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("intentional generation failure")
        return self.answer


class RunTests(FeatureFixtureMixin, unittest.TestCase):
    def make_experiment(self, root: Path) -> argparse.Namespace:
        selection_csv = root / "selection.csv"
        with selection_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class", "filename"])
            writer.writeheader()
            for number in range(20):
                true_class = "KIRC" if number < 10 else "KIRP"
                slide_id = f"slide-{number:02d}"
                writer.writerow(
                    {"class": true_class, "filename": f"{slide_id}.svs"}
                )
                self.write_features(
                    root,
                    slide_id=slide_id,
                    wsi=torch.ones(192),
                    region=torch.ones(1, 192),
                    patch=torch.ones(1, 16, 512),
                    cell=torch.ones(1, 16, 384),
                    patch_indices=torch.arange(16).reshape(1, 16),
                    cell_indices=torch.arange(16).reshape(1, 16),
                )
        return argparse.Namespace(
            model_dir=root / "model",
            feature_root=root / "features",
            cell_root=root / "cells",
            selection_csv=selection_csv,
            output_dir=root / "output",
            device="cpu",
            limit=None,
            preflight_only=False,
        )

    @staticmethod
    def successful_jsonl_records(path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("status") == "ok"
        ]

    def test_append_after_torn_tail_preserves_records_for_artifact_rebuild(
        self,
    ) -> None:
        prompt_by_id = dict(PROMPTS)
        protocol_version = "rcc-k16-three-prompt-v1"
        expected_metadata = {
            ("slide-a", "P0_open"): {
                "protocol_version": protocol_version,
                "prompt": prompt_by_id["P0_open"],
                "true_class": "KIRC",
            },
            ("slide-b", "P1_clear_first"): {
                "protocol_version": protocol_version,
                "prompt": prompt_by_id["P1_clear_first"],
                "true_class": "KIRP",
            },
            ("slide-c", "P2_papillary_first"): {
                "protocol_version": protocol_version,
                "prompt": prompt_by_id["P2_papillary_first"],
                "true_class": "KIRC",
            },
        }

        def success_record(
            slide_id: str, prompt_id: str, true_class: str, raw_answer: str
        ) -> dict[str, object]:
            parsed_class = "KIRP" if "papillary" in raw_answer else "KIRC"
            return {
                "status": "ok",
                "protocol_version": protocol_version,
                "slide_id": slide_id,
                "true_class": true_class,
                "prompt_id": prompt_id,
                "prompt": prompt_by_id[prompt_id],
                "raw_answer": raw_answer,
                "parsed_class": parsed_class,
                "correct": parsed_class == true_class,
                "generation_time_sec": 0.01,
                "started_at": "2026-08-10T00:00:00+00:00",
            }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            jsonl_path = output_dir / "predictions.jsonl"
            first = success_record(
                "slide-a", "P0_open", "KIRC", "clear cell renal cell carcinoma"
            )
            append_record(jsonl_path, first)
            complete_prefix = jsonl_path.read_bytes()
            with jsonl_path.open("ab") as handle:
                handle.write(b'{"status":"ok","slide_id":"torn"\n\n')

            second = success_record(
                "slide-b", "P1_clear_first", "KIRP", "papillary renal cell carcinoma"
            )
            third = success_record(
                "slide-c", "P2_papillary_first", "KIRC", "clear cell renal cell carcinoma"
            )
            append_record(jsonl_path, second)
            append_record(jsonl_path, third)
            successful = _write_artifacts(
                output_dir,
                jsonl_path,
                expected_metadata,
            )
            with (output_dir / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            final_bytes = jsonl_path.read_bytes()

        self.assertTrue(final_bytes.startswith(complete_prefix))
        self.assertEqual(len(successful), 3)
        self.assertEqual(len(csv_rows), 3)
        self.assertEqual(summary["total_successful_records"], 3)

    def test_preflight_only_returns_zero_without_evaluating_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            args.preflight_only = True

            def forbidden_loader(*_args: object, **_kwargs: object) -> object:
                raise AssertionError("preflight must not load the model")

            with redirect_stdout(io.StringIO()) as output:
                result = run(args, model_loader=forbidden_loader)

        self.assertEqual(result, 0)
        self.assertIn("Total slides: 20", output.getvalue())
        self.assertIn("KIRC=10", output.getvalue())
        self.assertIn("KIRP=10", output.getvalue())

    def test_limit_one_creates_three_successful_unique_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            args.limit = 1

            result = run(
                args,
                model_loader=lambda *_args, **_kwargs: (
                    DeterministicFakeModel(),
                    {},
                ),
            )

            with (args.output_dir / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(result, 0)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["prompt_id"] for row in rows},
            {"P0_open", "P1_clear_first", "P2_papillary_first"},
        )
        self.assertEqual(len({(row["slide_id"], row["prompt_id"]) for row in rows}), 3)

    def test_raw_answer_preserves_model_whitespace_in_jsonl_and_csv(self) -> None:
        raw_answer = " \n clear cell renal cell carcinoma \t"
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            args.limit = 1

            result = run(
                args,
                model_loader=lambda *_args, **_kwargs: (
                    DeterministicFakeModel(answer=raw_answer),
                    {},
                ),
            )
            jsonl_records = self.successful_jsonl_records(
                args.output_dir / "predictions.jsonl"
            )
            with (args.output_dir / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(result, 0)
        self.assertEqual(len(jsonl_records), 3)
        self.assertEqual(len(csv_rows), 3)
        self.assertTrue(
            all(record["raw_answer"] == raw_answer for record in jsonl_records)
        )
        self.assertTrue(all(row["raw_answer"] == raw_answer for row in csv_rows))
        self.assertTrue(
            all(record["parsed_class"] == "KIRC" for record in jsonl_records)
        )
        self.assertTrue(all(row["parsed_class"] == "KIRC" for row in csv_rows))

    def test_limit_one_all_errors_returns_nonzero_and_reports_three_missing_keys(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            args.limit = 1
            always_fails = DeterministicFakeModel(failures_remaining=3)

            with redirect_stdout(io.StringIO()) as output:
                result = run(
                    args,
                    model_loader=lambda *_args, **_kwargs: (always_fails, {}),
                )
            records = [
                json.loads(line)
                for line in (args.output_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertNotEqual(result, 0)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["status"] == "error" for record in records))
        self.assertTrue(
            all(
                record["protocol_version"] == "rcc-k16-three-prompt-v1"
                for record in records
            )
        )
        for prompt_id, _prompt in PROMPTS:
            self.assertIn(
                f"Missing successful key: slide-00 / {prompt_id}",
                output.getvalue(),
            )

    def test_repeating_limit_one_does_not_duplicate_successful_jsonl_records(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            args.limit = 1
            loader = lambda *_args, **_kwargs: (DeterministicFakeModel(), {})

            self.assertEqual(run(args, model_loader=loader), 0)
            self.assertEqual(run(args, model_loader=loader), 0)
            successful = self.successful_jsonl_records(
                args.output_dir / "predictions.jsonl"
            )

        self.assertEqual(len(successful), 3)
        self.assertEqual(
            len({(record["slide_id"], record["prompt_id"]) for record in successful}),
            3,
        )

    def test_generation_error_is_retryable_on_later_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            args.limit = 1

            first_result = run(
                args,
                model_loader=lambda *_args, **_kwargs: (
                    DeterministicFakeModel(failures_remaining=1),
                    {},
                ),
            )
            first_records = [
                json.loads(line)
                for line in (args.output_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            second_result = run(
                args,
                model_loader=lambda *_args, **_kwargs: (
                    DeterministicFakeModel(),
                    {},
                ),
            )
            successful = self.successful_jsonl_records(
                args.output_dir / "predictions.jsonl"
            )

        self.assertNotEqual(first_result, 0)
        self.assertEqual(second_result, 0)
        self.assertEqual(sum(record["status"] == "error" for record in first_records), 1)
        self.assertEqual(len(successful), 3)
        self.assertEqual(
            len({(record["slide_id"], record["prompt_id"]) for record in successful}),
            3,
        )

    def test_normal_run_creates_sixty_rows_and_all_prompt_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))

            result = run(
                args,
                model_loader=lambda *_args, **_kwargs: (
                    DeterministicFakeModel(),
                    {},
                ),
            )
            with (args.output_dir / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(
                (args.output_dir / "summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result, 0)
        self.assertEqual(len(rows), 60)
        self.assertEqual(
            set(summary["per_prompt"]),
            {"P0_open", "P1_clear_first", "P2_papillary_first"},
        )

    def test_full_run_scopes_artifacts_and_preserves_foreign_audit_record(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            jsonl_path = args.output_dir / "predictions.jsonl"
            append_record(
                jsonl_path,
                {
                    "status": "ok",
                    "slide_id": "foreign-slide",
                    "true_class": "KIRC",
                    "prompt_id": "P0_open",
                    "prompt": "foreign prompt",
                    "raw_answer": "clear cell renal cell carcinoma",
                    "parsed_class": "KIRC",
                    "correct": True,
                    "generation_time_sec": 0.01,
                    "started_at": "2026-08-10T00:00:00+00:00",
                },
            )

            result = run(
                args,
                model_loader=lambda *_args, **_kwargs: (
                    DeterministicFakeModel(),
                    {},
                ),
            )
            with (args.output_dir / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(
                (args.output_dir / "summary.json").read_text(encoding="utf-8")
            )
            audit_records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result, 0)
        self.assertEqual(len(rows), 60)
        self.assertNotIn("foreign-slide", {row["slide_id"] for row in rows})
        self.assertEqual(summary["total_successful_records"], 60)
        self.assertEqual(summary["unique_slides"], 20)
        self.assertIn("foreign-slide", {record["slide_id"] for record in audit_records})

    def test_stale_same_key_successes_are_retried_and_excluded_from_artifacts(
        self,
    ) -> None:
        protocol_version = "rcc-k16-three-prompt-v1"
        prompt_by_id = dict(PROMPTS)
        stale_generation_times = {str(900 + index) for index in range(6)}

        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            args.limit = 2
            jsonl_path = args.output_dir / "predictions.jsonl"
            keys = [
                (slide_id, prompt_id)
                for slide_id in ("slide-00", "slide-01")
                for prompt_id, _prompt in PROMPTS
            ]
            records: list[dict[str, object]] = []
            for index, (slide_id, prompt_id) in enumerate(keys):
                record: dict[str, object] = {
                    "status": "ok",
                    "protocol_version": protocol_version,
                    "slide_id": slide_id,
                    "true_class": "KIRC",
                    "prompt_id": prompt_id,
                    "prompt": prompt_by_id[prompt_id],
                    "raw_answer": "clear cell renal cell carcinoma",
                    "parsed_class": "KIRC",
                    "correct": True,
                    "generation_time_sec": 900 + index,
                    "started_at": "2026-08-10T00:00:00+00:00",
                }
                records.append(record)

            records[0].pop("protocol_version")
            records[1]["protocol_version"] = "rcc-k16-obsolete"
            records[2]["prompt"] = "stale prompt"
            records[3]["true_class"] = "KIRP"
            records[4]["parsed_class"] = "KIRP"
            records[5]["correct"] = False
            for record in records:
                append_record(jsonl_path, record)

            model = DeterministicFakeModel()
            result = run(
                args,
                model_loader=lambda *_args, **_kwargs: (model, {}),
            )
            with (args.output_dir / "predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            summary = json.loads(
                (args.output_dir / "summary.json").read_text(encoding="utf-8")
            )
            audit_records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result, 0)
        self.assertEqual(len(model.prompts), 6)
        self.assertEqual(len(csv_rows), 6)
        self.assertTrue(
            stale_generation_times.isdisjoint(
                {row["generation_time_sec"] for row in csv_rows}
            )
        )
        self.assertTrue(
            all(row["prompt"] == prompt_by_id[row["prompt_id"]] for row in csv_rows)
        )
        self.assertTrue(all(row["true_class"] == "KIRC" for row in csv_rows))
        self.assertTrue(all(row["parsed_class"] == "KIRC" for row in csv_rows))
        self.assertTrue(all(row["correct"] == "True" for row in csv_rows))
        self.assertEqual(summary["total_successful_records"], 6)
        self.assertEqual(len(audit_records), 12)
        self.assertTrue(all(record in audit_records for record in records))

    def test_incomplete_normal_run_returns_nonzero_and_reports_missing_keys(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_experiment(Path(directory))
            always_fails = DeterministicFakeModel(failures_remaining=60)

            with redirect_stdout(io.StringIO()) as output:
                result = run(
                    args,
                    model_loader=lambda *_args, **_kwargs: (always_fails, {}),
                )
            records = [
                json.loads(line)
                for line in (args.output_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertNotEqual(result, 0)
        self.assertEqual(len(records), 60)
        self.assertTrue(all(record["status"] == "error" for record in records))
        self.assertIn("Missing successful key: slide-00 / P0_open", output.getvalue())


if __name__ == "__main__":
    unittest.main()
