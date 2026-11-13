"""
Lightweight, catalog-free schema inference for scraped table definitions.

Unlike `sqlglot.optimizer.annotate_types`, which resolves column references against a
catalog, this module infers *basic* types purely from the AST:

- literals are typed by their literal kind (string -> VARCHAR, integer -> INT,
  decimal -> DOUBLE, boolean -> BOOLEAN)
- CAST / TRY_CAST take the target type
- arithmetic expressions follow numeric promotion rules
- function calls are resolved through the built-in FUNCTION_RETURN_TYPES table;
  misses stay UNKNOWN and are counted, so the table can be extended where it matters
"""

from __future__ import annotations

import typing as t
from collections import Counter

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType

__all__ = ["FUNCTION_RETURN_TYPES", "SchemaInference", "infer_schema", "infer_type"]


# Built-in function return types, keyed by uppercased SQL function name. Only functions
# whose return type doesn't depend on their arguments are listed; anything else (e.g.
# MIN/MAX, COALESCE, UDFs) intentionally stays UNKNOWN and is counted as a miss.
FUNCTION_RETURN_TYPES: dict[str, exp.DType] = {
    # Aggregates
    "COUNT": exp.DType.BIGINT,
    "COUNTIF": exp.DType.BIGINT,
    "SUM": exp.DType.DOUBLE,
    "AVG": exp.DType.DOUBLE,
    "STDDEV": exp.DType.DOUBLE,
    "STDDEV_POP": exp.DType.DOUBLE,
    "STDDEV_SAMP": exp.DType.DOUBLE,
    "VARIANCE": exp.DType.DOUBLE,
    "VAR_POP": exp.DType.DOUBLE,
    "VAR_SAMP": exp.DType.DOUBLE,
    # Ranking / windowing
    "ROW_NUMBER": exp.DType.BIGINT,
    "RANK": exp.DType.BIGINT,
    "DENSE_RANK": exp.DType.BIGINT,
    "NTILE": exp.DType.INT,
    "PERCENT_RANK": exp.DType.DOUBLE,
    "CUME_DIST": exp.DType.DOUBLE,
    # String -> string
    "CONCAT": exp.DType.VARCHAR,
    "CONCAT_WS": exp.DType.VARCHAR,
    "LOWER": exp.DType.VARCHAR,
    "UPPER": exp.DType.VARCHAR,
    "TRIM": exp.DType.VARCHAR,
    "LTRIM": exp.DType.VARCHAR,
    "RTRIM": exp.DType.VARCHAR,
    "BTRIM": exp.DType.VARCHAR,
    "SUBSTRING": exp.DType.VARCHAR,
    "SUBSTR": exp.DType.VARCHAR,
    "LEFT": exp.DType.VARCHAR,
    "RIGHT": exp.DType.VARCHAR,
    "REPLACE": exp.DType.VARCHAR,
    "REVERSE": exp.DType.VARCHAR,
    "REPEAT": exp.DType.VARCHAR,
    "LPAD": exp.DType.VARCHAR,
    "RPAD": exp.DType.VARCHAR,
    "INITCAP": exp.DType.VARCHAR,
    "REGEXP_REPLACE": exp.DType.VARCHAR,
    "SPLIT_PART": exp.DType.VARCHAR,
    "MD5": exp.DType.VARCHAR,
    "SHA1": exp.DType.VARCHAR,
    "SHA2": exp.DType.VARCHAR,
    "UUID": exp.DType.VARCHAR,
    "CURRENT_USER": exp.DType.VARCHAR,
    "USER": exp.DType.VARCHAR,
    # String -> number
    "LENGTH": exp.DType.INT,
    "CHAR_LENGTH": exp.DType.INT,
    "CHARACTER_LENGTH": exp.DType.INT,
    "ASCII": exp.DType.INT,
    "INSTR": exp.DType.INT,
    "LOCATE": exp.DType.INT,
    "STRPOS": exp.DType.INT,
    # Math
    "SQRT": exp.DType.DOUBLE,
    "POWER": exp.DType.DOUBLE,
    "POW": exp.DType.DOUBLE,
    "EXP": exp.DType.DOUBLE,
    "LN": exp.DType.DOUBLE,
    "LOG": exp.DType.DOUBLE,
    "LOG2": exp.DType.DOUBLE,
    "LOG10": exp.DType.DOUBLE,
    "SIN": exp.DType.DOUBLE,
    "COS": exp.DType.DOUBLE,
    "TAN": exp.DType.DOUBLE,
    "COT": exp.DType.DOUBLE,
    "ASIN": exp.DType.DOUBLE,
    "ACOS": exp.DType.DOUBLE,
    "ATAN": exp.DType.DOUBLE,
    "ATAN2": exp.DType.DOUBLE,
    "SINH": exp.DType.DOUBLE,
    "COSH": exp.DType.DOUBLE,
    "TANH": exp.DType.DOUBLE,
    "RADIANS": exp.DType.DOUBLE,
    "DEGREES": exp.DType.DOUBLE,
    "PI": exp.DType.DOUBLE,
    "RAND": exp.DType.DOUBLE,
    "RANDOM": exp.DType.DOUBLE,
    "SIGN": exp.DType.INT,
    # Date / time
    "CURRENT_DATE": exp.DType.DATE,
    "TO_DATE": exp.DType.DATE,
    "LAST_DAY": exp.DType.DATE,
    "CURRENT_TIMESTAMP": exp.DType.TIMESTAMP,
    "NOW": exp.DType.TIMESTAMP,
    "TO_TIMESTAMP": exp.DType.TIMESTAMP,
    "DATE_TRUNC": exp.DType.TIMESTAMP,
    "UNIX_TIMESTAMP": exp.DType.BIGINT,
    "YEAR": exp.DType.INT,
    "MONTH": exp.DType.INT,
    "DAY": exp.DType.INT,
    "HOUR": exp.DType.INT,
    "MINUTE": exp.DType.INT,
    "SECOND": exp.DType.INT,
    "DAYOFWEEK": exp.DType.INT,
    "DAYOFYEAR": exp.DType.INT,
    "WEEKOFYEAR": exp.DType.INT,
    "DATEDIFF": exp.DType.INT,
    # Predicates
    "STARTS_WITH": exp.DType.BOOLEAN,
    "ENDS_WITH": exp.DType.BOOLEAN,
}

# Numeric promotion ladder, from lowest to highest precedence.
_NUMERIC_PROMOTION = (
    exp.DType.TINYINT,
    exp.DType.SMALLINT,
    exp.DType.INT,
    exp.DType.BIGINT,
    exp.DType.DECIMAL,
    exp.DType.FLOAT,
    exp.DType.DOUBLE,
)
_NUMERIC_PRECEDENCE = {dtype: precedence for precedence, dtype in enumerate(_NUMERIC_PROMOTION)}

_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)


class SchemaInference(t.NamedTuple):
    """
    Attributes:
        tables: Table name -> column name -> basic type name, e.g.
            {"events": {"id": "INT", "score": "DOUBLE"}}. Compatible with
            `sqlglot.schema.MappingSchema`. Bare queries are stored under "".
        unknown_functions: Function names missing from FUNCTION_RETURN_TYPES, mapped to
            the number of times they were encountered during inference.
    """

    tables: dict[str, dict[str, str]]
    unknown_functions: dict[str, int]


def infer_type(expression: exp.Expression) -> exp.DType:
    """
    Infers the basic type of an expression from its AST.

    Example:
        >>> infer_type(exp.maybe_parse("1 + 2.5"))
        <DType.DOUBLE: 'DOUBLE'>

    Args:
        expression: The expression to infer the type of.

    Returns:
        The inferred type, or UNKNOWN if it can't be inferred from the AST alone.
    """
    return _infer_type(expression, None)


def infer_schema(sql: str | exp.Expression, dialect: DialectType = None) -> SchemaInference:
    """
    Infers table schemas from CREATE TABLE/VIEW statements (including CTAS) or bare queries.

    Example:
        >>> result = infer_schema("CREATE VIEW v AS SELECT 1 AS a, CAST(x AS DATE) AS b, f(x) AS c")
        >>> result.tables, result.unknown_functions
        ({'v': {'a': 'INT', 'b': 'DATE', 'c': 'UNKNOWN'}}, {'F': 1})

    Args:
        sql: One or more SQL statements, or an already-parsed expression.
        dialect: The dialect to parse `sql` with, if it's a string.

    Returns:
        The inferred schema together with the unknown-function counts.
    """
    statements = (
        [sql] if isinstance(sql, exp.Expression) else Dialect.get_or_raise(dialect).parse(sql)
    )

    tables: dict[str, dict[str, exp.DType]] = {}
    unknown_functions: Counter[str] = Counter()

    for statement in statements:
        if isinstance(statement, exp.Create):
            name, columns = _infer_create(statement, unknown_functions)
            if name is not None:
                tables[name] = columns
        elif isinstance(statement, exp.Query):
            tables[""] = _infer_query_columns(statement, unknown_functions)

    return SchemaInference(
        tables={
            name: {column: dtype.value for column, dtype in columns.items()}
            for name, columns in tables.items()
        },
        unknown_functions=dict(unknown_functions),
    )


def _promote_numeric(left: exp.DType, right: exp.DType) -> exp.DType:
    left_precedence = _NUMERIC_PRECEDENCE.get(left)
    right_precedence = _NUMERIC_PRECEDENCE.get(right)

    if left_precedence is None or right_precedence is None:
        return exp.DType.UNKNOWN

    return left if left_precedence >= right_precedence else right


def _infer_type(expression: exp.Expression, unknown_functions: Counter[str] | None) -> exp.DType:
    while isinstance(expression, (exp.Alias, exp.Paren)):
        expression = expression.this

    if isinstance(expression, exp.Literal):
        if expression.is_string:
            return exp.DType.VARCHAR
        return exp.DType.INT if expression.is_int else exp.DType.DOUBLE

    if isinstance(expression, exp.Boolean):
        return exp.DType.BOOLEAN

    if isinstance(expression, exp.Null):
        return exp.DType.NULL

    if isinstance(expression, exp.Cast):
        to = expression.to
        return to.this if isinstance(to, exp.DataType) else exp.DType.UNKNOWN

    if isinstance(expression, exp.Neg):
        return _infer_type(expression.this, unknown_functions)

    if isinstance(expression, _ARITHMETIC):
        left = _infer_type(expression.left, unknown_functions)
        right = _infer_type(expression.right, unknown_functions)

        if (
            isinstance(expression, exp.Div)
            and left in exp.DataType.INTEGER_TYPES
            and right in exp.DataType.INTEGER_TYPES
        ):
            # Integer division still widens to a fractional type in most engines
            return exp.DType.DOUBLE

        return _promote_numeric(left, right)

    if isinstance(expression, exp.Func):
        name = (
            expression.name
            if isinstance(expression, (exp.Anonymous, exp.AnonymousAggFunc))
            else expression.sql_name()
        ).upper()
        return_type = FUNCTION_RETURN_TYPES.get(name)

        if return_type is None:
            if unknown_functions is not None:
                unknown_functions[name] += 1
            return exp.DType.UNKNOWN

        return return_type

    return exp.DType.UNKNOWN


def _infer_query_columns(
    query: exp.Query, unknown_functions: Counter[str]
) -> dict[str, exp.DType]:
    columns: dict[str, exp.DType] = {}

    for projection in query.selects:
        if isinstance(projection, exp.Star) or (
            isinstance(projection, exp.Column) and projection.is_star
        ):
            # Star projections can't be enumerated without a catalog
            continue
        columns[projection.output_name or projection.sql()] = _infer_type(
            projection, unknown_functions
        )

    return columns


def _infer_create(
    create: exp.Create, unknown_functions: Counter[str]
) -> tuple[str | None, dict[str, exp.DType]]:
    this = create.this

    if isinstance(this, exp.Schema):
        table = this.this
        definitions = this.expressions
    else:
        table = this
        definitions = []

    if create.args.get("kind") not in ("TABLE", "VIEW") or not isinstance(table, exp.Table):
        return None, {}

    name = ".".join(part for part in (table.catalog, table.db, table.name) if part)

    column_defs = [d for d in definitions if isinstance(d, exp.ColumnDef)]
    if column_defs:
        columns = {}
        for column_def in column_defs:
            kind = column_def.args.get("kind")
            columns[column_def.name] = (
                kind.this if isinstance(kind, exp.DataType) else exp.DType.UNKNOWN
            )
        return name, columns

    query = create.expression
    inferred = (
        _infer_query_columns(query, unknown_functions) if isinstance(query, exp.Query) else {}
    )

    column_names = [definition.name for definition in definitions]
    if not column_names:
        return name, inferred

    # CTAS with an explicit column list: match names to projections positionally
    types = list(inferred.values())
    columns = {
        column_name: types[i] if i < len(types) else exp.DType.UNKNOWN
        for i, column_name in enumerate(column_names)
    }
    for extra_name, extra_type in list(inferred.items())[len(column_names) :]:
        columns.setdefault(extra_name, extra_type)

    return name, columns
