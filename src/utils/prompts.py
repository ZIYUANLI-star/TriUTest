SYSTEM_LABEL = "You are a Python testing assistant. Output only executable Python test code."

GEN_TEMPLATE = """[System]
{system}

[User]
{prompt}
-- TASK: Generate comprehensive pytest-compatible test cases.
-- REQUIREMENTS:
1) Typical, edge, and error conditions.
2) Tests must be fully runnable without external deps.
3) Use pytest style; avoid flaky timing.

[Assistant]
"""

FIX_TEMPLATE = """[System]
{system}

[User]
The previous tests failed with the following runtime error logs:
{error_log}

Code Under Test (CUT):
{code}

-- TASK: Modify/fix tests so they pass when CUT is correct; if CUT is wrong, write failing test that precisely reveals the defect. Keep tests deterministic.

[Assistant]
"""

DIAG_TEMPLATE = """[System]
{system}

[User]
We ran tests and saw errors:
{error_log}

CUT:
{code}

-- TASK: Diagnose whether the issue is in CUT or tests. Output a JSON with fields:
{{
  "root_cause": "CUT|TEST|BOTH|UNKNOWN",
  "explanation": "...",
  "suggested_fix": "..."
}}

[Assistant]
"""
