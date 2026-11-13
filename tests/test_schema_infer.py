import unittest

from sqlglot import exp, parse_one
from sqlglot.schema_infer import infer_schema, infer_type


class TestSchemaInfer(unittest.TestCase):
    def assert_inferred(self, sql: str, expected: dict[str, dict[str, str]]) -> None:
        self.assertEqual(infer_schema(sql).tables, expected)

    def test_literal_types(self) -> None:
        self.assertEqual(infer_type(parse_one("'foo'")), exp.DType.VARCHAR)
        self.assertEqual(infer_type(parse_one("1")), exp.DType.INT)
        self.assertEqual(infer_type(parse_one("-1")), exp.DType.INT)
        self.assertEqual(infer_type(parse_one("1.5")), exp.DType.DOUBLE)
        self.assertEqual(infer_type(parse_one("TRUE")), exp.DType.BOOLEAN)
        self.assertEqual(infer_type(parse_one("NULL")), exp.DType.NULL)

    def test_cast(self) -> None:
        self.assertEqual(infer_type(parse_one("CAST(x AS DATE)")), exp.DType.DATE)
        self.assertEqual(infer_type(parse_one("CAST(x AS DECIMAL(10, 2))")), exp.DType.DECIMAL)
        self.assertEqual(infer_type(parse_one("TRY_CAST(x AS BIGINT)")), exp.DType.BIGINT)
        self.assertEqual(
            infer_type(parse_one("CAST(1 + 2.5 AS VARCHAR)")), exp.DType.VARCHAR
        )

    def test_arithmetic_promotion(self) -> None:
        self.assertEqual(infer_type(parse_one("1 + 2")), exp.DType.INT)
        self.assertEqual(infer_type(parse_one("1 + 2.5")), exp.DType.DOUBLE)
        self.assertEqual(infer_type(parse_one("1.5 * 2")), exp.DType.DOUBLE)
        self.assertEqual(infer_type(parse_one("1 - CAST(x AS BIGINT)")), exp.DType.BIGINT)
        self.assertEqual(
            infer_type(parse_one("CAST(x AS DECIMAL) + 1")), exp.DType.DECIMAL
        )
        self.assertEqual(
            infer_type(parse_one("CAST(x AS DECIMAL) + 1.5")), exp.DType.DOUBLE
        )
        self.assertEqual(infer_type(parse_one("5 % 2")), exp.DType.INT)

    def test_division(self) -> None:
        self.assertEqual(infer_type(parse_one("1 / 2")), exp.DType.DOUBLE)
        self.assertEqual(infer_type(parse_one("1 / 2.5")), exp.DType.DOUBLE)

    def test_arithmetic_with_unknown_operand(self) -> None:
        self.assertEqual(infer_type(parse_one("x + 1")), exp.DType.UNKNOWN)
        self.assertEqual(infer_type(parse_one("1 + 'a'")), exp.DType.UNKNOWN)

    def test_function_return_types(self) -> None:
        self.assertEqual(infer_type(parse_one("COUNT(*)")), exp.DType.BIGINT)
        self.assertEqual(infer_type(parse_one("AVG(x)")), exp.DType.DOUBLE)
        self.assertEqual(infer_type(parse_one("CONCAT(a, b)")), exp.DType.VARCHAR)
        self.assertEqual(infer_type(parse_one("LENGTH(a)")), exp.DType.INT)
        self.assertEqual(infer_type(parse_one("NOW()")), exp.DType.TIMESTAMP)
        self.assertEqual(infer_type(parse_one("CURRENT_DATE")), exp.DType.DATE)

    def test_unknown_functions_are_counted(self) -> None:
        result = infer_schema(
            "CREATE VIEW v AS SELECT MY_UDF(a) AS x, MY_UDF(b) AS y, other_fn(1) AS z"
        )
        self.assertEqual(
            result.tables, {"v": {"x": "UNKNOWN", "y": "UNKNOWN", "z": "UNKNOWN"}}
        )
        self.assertEqual(result.unknown_functions, {"MY_UDF": 2, "OTHER_FN": 1})

    def test_nested_expressions(self) -> None:
        self.assertEqual(infer_type(parse_one("CAST(SUM(x) AS DECIMAL)")), exp.DType.DECIMAL)
        self.assertEqual(infer_type(parse_one("1 + CAST('2' AS INT)")), exp.DType.INT)
        self.assertEqual(infer_type(parse_one("(1 + 2) * 3")), exp.DType.INT)
        self.assertEqual(infer_type(parse_one("CONCAT(a) + 1")), exp.DType.UNKNOWN)

    def test_create_table_with_declared_types(self) -> None:
        self.assert_inferred(
            "CREATE TABLE db.t (a INT, b TEXT, c DECIMAL(10, 2))",
            {"db.t": {"a": "INT", "b": "TEXT", "c": "DECIMAL"}},
        )

    def test_create_table_as_select(self) -> None:
        self.assert_inferred(
            "CREATE TABLE t AS SELECT 1 AS id, 'a' AS name, CAST(x AS DATE) AS d",
            {"t": {"id": "INT", "name": "VARCHAR", "d": "DATE"}},
        )

    def test_create_table_with_column_list_as_select(self) -> None:
        self.assert_inferred(
            "CREATE TABLE t (id, name) AS SELECT 1, 'a'",
            {"t": {"id": "INT", "name": "VARCHAR"}},
        )

    def test_create_view(self) -> None:
        self.assert_inferred(
            "CREATE VIEW v AS SELECT COUNT(*) AS cnt, AVG(x) AS avg_x FROM t",
            {"v": {"cnt": "BIGINT", "avg_x": "DOUBLE"}},
        )

    def test_create_table_without_types_or_query(self) -> None:
        self.assert_inferred(
            "CREATE TABLE t (a, b)",
            {"t": {"a": "UNKNOWN", "b": "UNKNOWN"}},
        )

    def test_star_projections_are_skipped(self) -> None:
        self.assert_inferred(
            "CREATE VIEW v AS SELECT *, 1 AS one FROM t",
            {"v": {"one": "INT"}},
        )

    def test_bare_query(self) -> None:
        self.assert_inferred("SELECT 1 AS one, 2.5 AS two", {"": {"one": "INT", "two": "DOUBLE"}})

    def test_multiple_statements(self) -> None:
        self.assert_inferred(
            "CREATE TABLE a (x INT); CREATE VIEW b AS SELECT 1.5 AS y",
            {"a": {"x": "INT"}, "b": {"y": "DOUBLE"}},
        )

    def test_unnamed_projection_falls_back_to_sql(self) -> None:
        self.assert_inferred("CREATE VIEW v AS SELECT COUNT(*)", {"v": {"COUNT(*)": "BIGINT"}})

    def test_inferred_schema_feeds_mapping_schema(self) -> None:
        from sqlglot.schema import MappingSchema

        result = infer_schema("CREATE VIEW v AS SELECT 1 AS id, 'a' AS name")
        schema = MappingSchema(result.tables)
        self.assertEqual(schema.column_names("v"), ["id", "name"])
        self.assertEqual(schema.get_column_type("v", "id"), exp.DataType.build("INT"))


if __name__ == "__main__":
    unittest.main()
