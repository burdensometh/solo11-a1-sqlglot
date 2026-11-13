import unittest

import sqlglot
from sqlglot.errors import ParseError, TokenError


class TestBatchParse(unittest.TestCase):
    def test_all_valid(self):
        report = sqlglot.parse_batch(["SELECT 1", "SELECT 2", "SELECT 3"])

        self.assertTrue(report.ok)
        self.assertEqual(report.total, 3)
        self.assertEqual(report.succeeded, 3)
        self.assertEqual(report.failed, 0)
        self.assertEqual(report.failures, [])
        self.assertEqual(
            [result[0].sql() for result in report.results], ["SELECT 1", "SELECT 2", "SELECT 3"]
        )

    def test_failure_does_not_stop_batch(self):
        sqls = ["SELECT 1", "SELECT foo FROM WHERE x", "SELECT 3"]
        report = sqlglot.parse_batch(sqls)

        self.assertFalse(report.ok)
        self.assertEqual(report.total, 3)
        self.assertEqual(report.succeeded, 2)
        self.assertEqual(report.failed, 1)
        self.assertEqual(len(report.results), 3)
        self.assertIsNone(report.results[1])
        self.assertEqual(report.results[0][0].sql(), "SELECT 1")
        self.assertEqual(report.results[2][0].sql(), "SELECT 3")

        failure = report.failures[0]
        self.assertEqual(failure.index, 1)
        self.assertEqual(failure.sql, sqls[1])
        self.assertIsInstance(failure.error, ParseError)
        self.assertEqual(failure.line, 1)
        self.assertEqual(failure.col, 21)
        self.assertEqual(failure.context, "SELECT foo FROM WHERE x")
        self.assertIn("Expected table name", failure.description)

    def test_multiple_failures(self):
        report = sqlglot.parse_batch(
            ["SELECT foo FROM WHERE x", "SELECT 1", "SELECT a FROM WHERE y", "SELECT 2"]
        )

        self.assertEqual(report.failed, 2)
        self.assertEqual([failure.index for failure in report.failures], [0, 2])
        self.assertEqual(report.succeeded, 2)

    def test_token_error_is_captured(self):
        report = sqlglot.parse_batch(["SELECT 'abc", "SELECT 1"])

        self.assertEqual(report.failed, 1)
        failure = report.failures[0]
        self.assertIsInstance(failure.error, TokenError)
        self.assertIsNone(failure.line)
        self.assertIsNone(failure.col)
        self.assertEqual(failure.context, "")
        self.assertTrue(failure.description)
        self.assertEqual(report.results[1][0].sql(), "SELECT 1")

    def test_error_message_context_defaults_to_20(self):
        sql = "SELECT " + "a" * 40 + " FROM WHERE x"
        report = sqlglot.parse_batch([sql])

        self.assertEqual(report.failures[0].context, "a" * 14 + " FROM WHERE x")

    def test_error_message_context_is_configurable(self):
        sql = "SELECT " + "a" * 40 + " FROM WHERE x"
        report = sqlglot.parse_batch([sql], error_message_context=5)

        self.assertEqual(report.failures[0].context, "FROM WHERE x")

    def test_multi_statement_script_collects_errors(self):
        report = sqlglot.parse_batch(["SELECT 1; SELECT foo FROM WHERE x"])

        self.assertEqual(report.failed, 1)
        self.assertIsInstance(report.failures[0].error, ParseError)
        self.assertGreaterEqual(len(report.failures[0].error.errors), 1)

    def test_dialect_and_generator_input(self):
        report = sqlglot.parse_batch(iter(["SELECT 1", "SELECT 2"]), dialect="duckdb")

        self.assertTrue(report.ok)
        self.assertEqual(report.total, 2)

    def test_report_text(self):
        report = sqlglot.parse_batch(["SELECT 1", "SELECT foo FROM WHERE x"])
        text = report.report()

        self.assertIn("2 total, 1 succeeded, 1 failed", text)
        self.assertIn("script #1 (line 1, col 21)", text)
        self.assertIn("error:", text)
        self.assertIn("context: SELECT foo FROM WHERE x", text)
        self.assertIn("sql: SELECT foo FROM WHERE x", text)
        self.assertEqual(str(report), "Batch parse report: 2 total, 1 succeeded, 1 failed")

    def test_report_text_truncates_long_sql(self):
        report = sqlglot.parse_batch(["SELECT " + "a" * 500 + " FROM WHERE x"])
        text = report.report()

        sql_line = next(line for line in text.splitlines() if line.strip().startswith("sql:"))
        self.assertTrue(sql_line.endswith("..."))
        self.assertLessEqual(len(sql_line), len("    sql: ") + 203)


if __name__ == "__main__":
    unittest.main()
