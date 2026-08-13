import csv
import json
import tempfile
import unittest
from pathlib import Path

from Ours.rcc_k16_protocol import (
    PROMPTS,
    append_record,
    build_summary,
    load_selection,
    parse_answer,
    read_success_records,
    slide_id_from_filename,
)


class RCCK16ProtocolTests(unittest.TestCase):
    def write_selection(self, directory: Path, rows: list[dict[str, str]]) -> Path:
        path = directory / "selection.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class", "filename"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_selection_rejects_wrong_kirc_kirp_balance(self):
        rows = [
            {"class": "KIRC", "filename": f"clear-{index:02d}.svs"}
            for index in range(19)
        ] + [{"class": "KIRP", "filename": "papillary-00.svs"}]
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_selection(Path(temporary), rows)
            with self.assertRaisesRegex(
                ValueError, r"expected 10 KIRC and 10 KIRP.*19.*1"
            ):
                load_selection(path)

    def test_selection_rejects_unsupported_class(self):
        rows = [
            {"class": "KIRC", "filename": f"clear-{index:02d}.svs"}
            for index in range(10)
        ] + [
            {"class": "KIRP", "filename": f"papillary-{index:02d}.svs"}
            for index in range(9)
        ] + [{"class": "KICH", "filename": "unsupported.svs"}]
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_selection(Path(temporary), rows)
            with self.assertRaisesRegex(ValueError, "unsupported class"):
                load_selection(path)

    def test_selection_rejects_duplicate_normalized_slide_id(self):
        rows = [
            {"class": "KIRC", "filename": "same.svs"},
            {"class": "KIRC", "filename": "same.extra.svs"},
        ] + [
            {"class": "KIRC", "filename": f"clear-{index:02d}.svs"}
            for index in range(8)
        ] + [
            {"class": "KIRP", "filename": f"papillary-{index:02d}.svs"}
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_selection(Path(temporary), rows)
            with self.assertRaisesRegex(ValueError, "duplicate normalized slide ID"):
                load_selection(path)

    def test_selection_returns_sorted_normalized_rows_with_required_keys(self):
        rows = [
            {"class": "KIRP", "filename": f"z-papillary-{index:02d}.x.svs"}
            for index in range(10)
        ] + [
            {"class": "KIRC", "filename": f"a-clear-{index:02d}.svs"}
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            selection = load_selection(self.write_selection(Path(temporary), rows))

        self.assertEqual(len(selection), 20)
        self.assertEqual(selection[0], {
            "slide_id": "a-clear-00", "true_class": "KIRC", "filename": "a-clear-00.svs"
        })
        self.assertEqual(selection[-1], {
            "slide_id": "z-papillary-09", "true_class": "KIRP", "filename": "z-papillary-09.x.svs"
        })
        self.assertEqual(
            [row["slide_id"] for row in selection],
            sorted(row["slide_id"] for row in selection),
        )
        self.assertTrue(
            all(set(row) == {"slide_id", "true_class", "filename"} for row in selection)
        )

    def test_jsonl_recovers_ok_record_and_ignores_error_and_torn_final_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runs" / "records.jsonl"
            append_record(path, {
                "slide_id": "slide-a", "prompt_id": "P0_open", "status": "ok", "answer": "ok"
            })
            append_record(path, {
                "slide_id": "slide-b", "prompt_id": "P0_open", "status": "error", "error": "timeout"
            })
            with path.open("ab") as handle:
                handle.write(b'{"slide_id":"partial"')

            records = read_success_records(path)

        self.assertEqual(records, {
            ("slide-a", "P0_open"): {
                "slide_id": "slide-a", "prompt_id": "P0_open", "status": "ok", "answer": "ok"
            }
        })

    def test_jsonl_rejects_malformed_line_before_later_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_text(
                json.dumps({"slide_id": "first", "prompt_id": "P0_open", "status": "ok"})
                + "\n{bad json}\n"
                + json.dumps({"slide_id": "last", "prompt_id": "P0_open", "status": "ok"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"line 2"):
                read_success_records(path)

    def test_append_rejects_malformed_nonfinal_line_without_mutating_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            original = (
                json.dumps(
                    {"slide_id": "first", "prompt_id": "P0_open", "status": "ok"}
                ).encode("utf-8")
                + b"\n{bad json}\n"
                + json.dumps(
                    {"slide_id": "last", "prompt_id": "P0_open", "status": "ok"}
                ).encode("utf-8")
                + b"\n"
            )
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, r"line 2"):
                append_record(
                    path,
                    {
                        "slide_id": "new",
                        "prompt_id": "P0_open",
                        "status": "ok",
                    },
                )

            self.assertEqual(path.read_bytes(), original)

    def test_append_adds_missing_lf_before_two_new_records(self):
        first = {
            "slide_id": "slide-a",
            "prompt_id": "P0_open",
            "status": "ok",
            "answer": "first",
        }
        second = {
            "slide_id": "slide-b",
            "prompt_id": "P1_clear_first",
            "status": "ok",
            "answer": "second",
        }
        third = {
            "slide_id": "slide-c",
            "prompt_id": "P2_papillary_first",
            "status": "ok",
            "answer": "third",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_bytes(
                json.dumps(
                    first, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            )

            append_record(path, second)
            append_record(path, third)
            records = read_success_records(path)

        self.assertEqual(
            records,
            {
                ("slide-a", "P0_open"): first,
                ("slide-b", "P1_clear_first"): second,
                ("slide-c", "P2_papillary_first"): third,
            },
        )

    def test_truncated_utf8_tail_is_ignored_then_repaired_before_two_appends(self):
        first = {
            "slide_id": "slide-a",
            "prompt_id": "P0_open",
            "status": "ok",
            "answer": "first",
        }
        second = {
            "slide_id": "slide-b",
            "prompt_id": "P1_clear_first",
            "status": "ok",
            "answer": "second",
        }
        third = {
            "slide_id": "slide-c",
            "prompt_id": "P2_papillary_first",
            "status": "ok",
            "answer": "third",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            append_record(path, first)
            complete_prefix = path.read_bytes()
            with path.open("ab") as handle:
                handle.write(b'{"status":"ok","answer":"' + "漢".encode("utf-8")[:2])

            recovered_before_append = read_success_records(path)
            append_record(path, second)
            append_record(path, third)
            records = read_success_records(path)
            final_bytes = path.read_bytes()

        self.assertEqual(recovered_before_append, {("slide-a", "P0_open"): first})
        self.assertTrue(final_bytes.startswith(complete_prefix))
        self.assertEqual(
            records,
            {
                ("slide-a", "P0_open"): first,
                ("slide-b", "P1_clear_first"): second,
                ("slide-c", "P2_papillary_first"): third,
            },
        )

    def test_jsonl_uses_latest_ok_record_for_duplicate_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            append_record(path, {
                "slide_id": "slide-a", "prompt_id": "P1_clear_first", "status": "ok", "answer": "old"
            })
            append_record(path, {
                "slide_id": "slide-a", "prompt_id": "P1_clear_first", "status": "ok", "answer": "new"
            })
            records = read_success_records(path)

        self.assertEqual(records[("slide-a", "P1_clear_first")]["answer"], "new")

    def test_append_record_round_trips_non_ascii_as_one_jsonl_line(self):
        record = {
            "slide_id": "slide-漢字", "prompt_id": "P2_papillary_first", "status": "ok", "answer": "診斷"
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "records.jsonl"
            append_record(path, record)
            lines = path.read_text(encoding="utf-8").splitlines()
            records = read_success_records(path)

        self.assertEqual(lines, [json.dumps(record, ensure_ascii=False, separators=(",", ":"))])
        self.assertEqual(records, {("slide-漢字", "P2_papillary_first"): record})

    def test_missing_jsonl_returns_no_success_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = read_success_records(Path(temporary) / "missing.jsonl")
        self.assertEqual(records, {})

    def test_summary_reports_independent_prompt_metrics_and_closed_set_stability(self):
        records = [
            {"slide_id": "slide-a", "true_class": "KIRC", "prompt_id": "P0_open", "parsed_class": "OTHER", "status": "ok"},
            {"slide_id": "slide-a", "true_class": "KIRC", "prompt_id": "P1_clear_first", "parsed_class": "KIRC", "status": "ok"},
            {"slide_id": "slide-a", "true_class": "KIRC", "prompt_id": "P2_papillary_first", "parsed_class": "KIRC", "status": "ok"},
            {"slide_id": "slide-b", "true_class": "KIRP", "prompt_id": "P0_open", "parsed_class": "AMBIGUOUS", "status": "ok"},
            {"slide_id": "slide-b", "true_class": "KIRP", "prompt_id": "P1_clear_first", "parsed_class": "KIRC", "status": "ok"},
            {"slide_id": "slide-b", "true_class": "KIRP", "prompt_id": "P2_papillary_first", "parsed_class": "KIRP", "status": "ok"},
        ]

        summary = build_summary(records)

        self.assertEqual(summary["total_successful_records"], 6)
        self.assertEqual(summary["unique_slides"], 2)
        expected_metrics = {
            "P0_open": {
                "n": 2, "accuracy": 0.0, "kirc_recall": 0.0, "kirp_recall": 0.0,
                "balanced_accuracy": 0.0, "other_rate": 0.5, "ambiguous_rate": 0.5,
                "prediction_counts": {"KIRC": 0, "KIRP": 0, "OTHER": 1, "AMBIGUOUS": 1},
                "confusion_counts": {
                    "KIRC": {"KIRC": 0, "KIRP": 0, "OTHER": 1, "AMBIGUOUS": 0},
                    "KIRP": {"KIRC": 0, "KIRP": 0, "OTHER": 0, "AMBIGUOUS": 1},
                },
            },
            "P1_clear_first": {
                "n": 2, "accuracy": 0.5, "kirc_recall": 1.0, "kirp_recall": 0.0,
                "balanced_accuracy": 0.5, "other_rate": 0.0, "ambiguous_rate": 0.0,
                "prediction_counts": {"KIRC": 2, "KIRP": 0, "OTHER": 0, "AMBIGUOUS": 0},
                "confusion_counts": {
                    "KIRC": {"KIRC": 1, "KIRP": 0, "OTHER": 0, "AMBIGUOUS": 0},
                    "KIRP": {"KIRC": 1, "KIRP": 0, "OTHER": 0, "AMBIGUOUS": 0},
                },
            },
            "P2_papillary_first": {
                "n": 2, "accuracy": 1.0, "kirc_recall": 1.0, "kirp_recall": 1.0,
                "balanced_accuracy": 1.0, "other_rate": 0.0, "ambiguous_rate": 0.0,
                "prediction_counts": {"KIRC": 1, "KIRP": 1, "OTHER": 0, "AMBIGUOUS": 0},
                "confusion_counts": {
                    "KIRC": {"KIRC": 1, "KIRP": 0, "OTHER": 0, "AMBIGUOUS": 0},
                    "KIRP": {"KIRC": 0, "KIRP": 1, "OTHER": 0, "AMBIGUOUS": 0},
                },
            },
        }
        self.assertEqual(summary["per_prompt"], expected_metrics)
        self.assertEqual(summary["closed_set_stability"], {
            "prompt_ids": ["P1_clear_first", "P2_papillary_first"],
            "n_paired": 2,
            "agreement_rate": 0.5,
            "prompt_sensitive_rate": 0.5,
            "stable_correct_rate": 0.5,
            "stable_wrong_rate": 0.0,
        })
        self.assertNotIn("P0_open", summary["closed_set_stability"]["prompt_ids"])

    def test_prompt_ids_are_the_required_order(self):
        self.assertEqual(
            [prompt_id for prompt_id, _ in PROMPTS],
            ["P0_open", "P1_clear_first", "P2_papillary_first"],
        )

    def test_clear_first_prompt_orders_target_diagnoses(self):
        prompt = PROMPTS[1][1].lower()
        self.assertLess(
            prompt.index("clear cell renal cell carcinoma"),
            prompt.index("papillary renal cell carcinoma"),
        )

    def test_papillary_first_prompt_orders_target_diagnoses(self):
        prompt = PROMPTS[2][1].lower()
        self.assertLess(
            prompt.index("papillary renal cell carcinoma"),
            prompt.index("clear cell renal cell carcinoma"),
        )

    def test_papillary_answers_parse_as_kirp(self):
        for answer in ("Papillary renal cell carcinoma", "KIRP"):
            with self.subTest(answer=answer):
                self.assertEqual(parse_answer(answer), "KIRP")

    def test_clear_cell_and_conventional_answers_parse_as_kirc(self):
        for answer in (
            "Clear-cell renal cell carcinoma",
            "Conventional renal cell carcinoma",
            "KIRC",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(parse_answer(answer), "KIRC")

    def test_non_target_or_empty_answers_parse_as_other(self):
        for answer in ("Chromophobe renal cell carcinoma", "Renal cell carcinoma", ""):
            with self.subTest(answer=answer):
                self.assertEqual(parse_answer(answer), "OTHER")

    def test_mixed_target_signals_parse_as_ambiguous(self):
        self.assertEqual(
            parse_answer("Clear-cell and papillary renal cell carcinoma"),
            "AMBIGUOUS",
        )

    def test_gdc_filename_normalizes_to_slide_id(self):
        self.assertEqual(
            slide_id_from_filename(
                "TCGA-A3-3374-01Z-00-DX1.560b8f9f-5195-43e0-be52-73aede19e077.svs"
            ),
            "TCGA-A3-3374-01Z-00-DX1",
        )


if __name__ == "__main__":
    unittest.main()
