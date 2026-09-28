"""calculate(expression) - a safe arithmetic evaluator.

The LLM never runs code. It sends a *string*; we parse it into an AST and only evaluate
node types on an allow-list (numbers, + - * / // % **, a few math functions).
Anything else - names, attribute access, imports, calls to other functions - is rejected.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from app.tools.base import Permission, Tool, ToolError

MAX_EXPRESSION_LENGTH = 500
MAX_EXPONENT = 1000  # stops 9**9**9 from freezing the process

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
    "exp": math.exp, "min": min, "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            # Estimate the size of the result (in decimal digits) before computing it.
            if abs(right) > MAX_EXPONENT or (abs(left) > 1 and abs(right) * math.log10(abs(left)) > MAX_EXPONENT):
                raise ToolError(f"Result of '**' would be too large (over {MAX_EXPONENT} digits).", "invalid_expression")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
    raise ToolError(f"Unsupported element in expression: {ast.dump(node)[:80]}", "invalid_expression")


def calculate(expression: str) -> dict[str, Any]:
    expression = expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not expression:
        raise ToolError("Expression is empty.", "invalid_expression")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Expression too long (max {MAX_EXPRESSION_LENGTH} chars).", "invalid_expression")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Invalid syntax: {exc.msg}", "invalid_expression") from exc

    try:
        result = _eval(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero.", "math_error") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"Math error: {exc}", "math_error") from exc

    if isinstance(result, float):
        if math.isnan(result) or math.isinf(result):
            raise ToolError("Result is not a finite number.", "math_error")
        if result.is_integer() and abs(result) < 1e15:
            result = int(result)
        else:
            result = round(result, 10)
    return {"expression": expression, "value": result}


CALCULATOR_TOOL = Tool(
    name="calculate",
    description=(
        "Evaluate an arithmetic expression exactly. Use this for any calculation instead of doing "
        "math in your head. Supports + - * / // % ** and parentheses, the constants pi and e, and "
        "the functions sqrt, abs, round, floor, ceil, sin, cos, tan, log, log10, exp, min, max."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to evaluate, e.g. '25 * 17' or 'sqrt(2) * (3 + 4)'.",
                "minLength": 1,
                "maxLength": MAX_EXPRESSION_LENGTH,
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    function=calculate,
    title="Calculator",
    permission=Permission.READ,
)
