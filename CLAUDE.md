## Environment & Dependencies

- Tooling: Python 3.11+, standard `pip`, and `venv`
- Virtual Env: Always active `.venv/` in the project root
- Dependencies: Listed in `requirements.txt` (or split dev requirements)

## Core Commands

### Adding a dependency

Install, then pin the resolved version in `requirements.txt` by hand:

```bash
pip install PACKAGE
pip show PACKAGE | grep -i version   # add to requirements.txt as PACKAGE==X.Y.Z
```

Do not use `pip freeze` to update `requirements.txt` — it writes the full
transitive tree and obscures the project's direct dependencies. Pin direct
dependencies only.

## Development & Quality Checks

- Run Tests: `pytest`
- Run Single Test: `pytest tests/test_file.py::test_function`

## Project Structure Conventions

- `llm/` - Production application source code
- `tests/` - Test suites matching the structure of `src/`
- `.venv/` - Local virtual environment (git-ignored)
- `requirements.txt` - Deployment dependencies

## Code Style & Anti-Patterns

- ALWAYS check if `.venv` is active before executing scripts.
- NEVER run bare `pip install <package>` without appending to `requirements.txt`.
- Prefer explicit type hints on all function signatures.
- Write tests for any new modules or functions added.
- Keep functions focused and small (< 50 lines if possible).
- Match the existing project style before introducing a new pattern.
- Remove imports, variables, or functions made unused by your own changes.
- Add type hints to public functions and return values.
- All exception paths should be handled explicitly.
- Ask before adding runtime dependencies.

## Working Style

- State assumptions before editing.
- If the task is ambiguous, ask one clarifying question.
- Prefer the smallest change that solves the problem.
- Touch only files needed for the requested change.
- Before finishing, verify with the narrowest relevant test.

## Communication

- Be concise.
- Call out trade-offs when multiple reasonable approaches exist.
- Push back if a safer or smaller approach would meet the goal.
