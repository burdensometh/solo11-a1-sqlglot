from __future__ import annotations

import typing as t
from dataclasses import dataclass

from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import (
    ERROR_POSITION_CONTEXT_DEFAULT,
    ErrorLevel,
    ParseError,
    SqlglotError,
)
from sqlglot.expressions import Expr

if t.TYPE_CHECKING:
    from typing_extensions import Unpack

    from sqlglot._typing import ParserNoDialectArgs
    from sqlglot.dialects.dialect import DialectType


@dataclass
class BatchParseFailure:
    """A single SQL script that failed to parse during a batch parse."""

    index: int
    """The 0-based index of the script in the input collection."""

    sql: str
    """The source SQL of the failed script."""

    error: SqlglotError
    """The error that was raised while parsing the script."""

    @property
    def line(self) -> int | None:
        """The 1-based line number of the offending token, if available."""
        return self.error.line if isinstance(self.error, ParseError) else None

    @property
    def col(self) -> int | None:
        """The column number of the offending token, if available."""
        return self.error.col if isinstance(self.error, ParseError) else None

    @property
    def description(self) -> str:
        """A single-line description of the error."""
        if isinstance(self.error, ParseError) and self.error.description:
            return self.error.description
        return str(self.error)

    @property
    def context(self) -> str:
        """The offending source snippet, with up to 20 characters of context on each side."""
        return self.error.context() if isinstance(self.error, ParseError) else ""


@dataclass
class BatchParseReport:
    """The aggregated result of batch-parsing a collection of SQL scripts."""

    results: list[list[Expr | None] | None]
    """One slot per input script: the parsed statements, or None if the script failed."""

    failures: list[BatchParseFailure]
    """One entry per failed script, in input order."""

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return self.total - len(self.failures)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def ok(self) -> bool:
        return not self.failures

    def report(self) -> str:
        """Returns a human-readable summary of every failure in the batch."""
        lines = [str(self)]
        for i, failure in enumerate(self.failures, start=1):
            header = f"[{i}] script #{failure.index}"
            if failure.line is not None:
                header += f" (line {failure.line}, col {failure.col})"
            lines.append(header)
            lines.append(f"    error: {failure.description}")
            if failure.context:
                lines.append(f"    context: {' '.join(failure.context.split())}")
            sql = " ".join(failure.sql.split())
            if len(sql) > 200:
                sql = f"{sql[:200]}..."
            lines.append(f"    sql: {sql}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"Batch parse report: {self.total} total, "
            f"{self.succeeded} succeeded, {self.failed} failed"
        )


def parse_batch(
    sqls: t.Iterable[str],
    read: DialectType = None,
    dialect: DialectType = None,
    **opts: Unpack[ParserNoDialectArgs],
) -> BatchParseReport:
    """
    Parses a collection of SQL scripts, capturing errors per script so that one
    failure does not prevent the remaining scripts from being parsed.

    Args:
        sqls: the SQL code strings to parse.
        read: the SQL dialect to apply during parsing (eg. "spark", "hive", "presto", "mysql").
            See `sqlglot.dialects.Dialects` for options.
        dialect: the SQL dialect (alias for read).
        **opts: other `sqlglot.parser.Parser` options. By default, `error_level` is
            `ErrorLevel.RAISE` so that all errors in a script are collected, and
            `error_message_context` is 20 so that errors carry a short source snippet.

    Returns:
        A `BatchParseReport` with one result slot per input script and one
        `BatchParseFailure` per script that failed to parse.
    """
    dialect_ = Dialect.get_or_raise(read or dialect)
    opts.setdefault("error_level", ErrorLevel.RAISE)
    opts.setdefault("error_message_context", ERROR_POSITION_CONTEXT_DEFAULT)

    results: list[list[Expr | None] | None] = []
    failures: list[BatchParseFailure] = []

    for index, sql in enumerate(sqls):
        try:
            results.append(dialect_.parse(sql, **opts))
        except SqlglotError as e:
            results.append(None)
            failures.append(BatchParseFailure(index=index, sql=sql, error=e))

    return BatchParseReport(results=results, failures=failures)
