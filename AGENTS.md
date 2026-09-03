# Working Guidelines

## Scope and simplicity

- If a direct instruction in the active conversation conflicts with this file,
  stop and ask the user to confirm which instruction should govern before
  proceeding.
- Make the smallest change that solves the requested problem.
- Do not add features, abstractions, duplicate implementations, or operational
  services that were not requested.
- Prefer established project patterns and working dependencies over replacements.
- Ask before making a change when its necessity or intended behavior is unclear.

## Code quality

- Add honest type annotations to new code and docstrings where they clarify an
  interface or behavior.
- Do not suppress typing errors with `# type: ignore`; fix the type contract or
  leave a visible, documented limitation.
- For asynchronous applications, keep network and other blocking work out of
  the event loop. Ask before introducing a synchronous bridge where it matters.
- After generating or changing concurrent code, check variable scope and the
  values captured or passed into asynchronous tasks.
- Use the standard-library `tomllib` to read TOML when available.
- In Typer CLIs, accept date and datetime inputs as strings and parse them in
  the command function.
- Validate external configuration structurally and by type. Require native
  booleans rather than truthy strings, and reject `null` for required fields
  unless the schema explicitly permits it.
- Before adding a datastore aggregation, verify the source data type and ensure
  all writers and readers use the same key and value contract.
- Use DuckDB when the project needs an analytical database: local, query-heavy
  research data, joins, aggregations, or Parquet/CSV analysis belong there by
  default unless another storage system is explicitly required.
- Do not create daemons or mark Python files executable unless explicitly asked.
- Add command-line argument parsing to standalone scripts only when requested.
- Put durable usage examples and demonstration scripts in `temp/`; reserve
  `debug/` for disposable work created during implementation.

## Numerical and tabular work

- Prefer vectorized NumPy operations—broadcasting, ufuncs, boolean masks, and
  reductions—for numerical transformations.
- For a genuinely hot numerical loop that cannot be vectorized clearly, use
  Numba where it is compatible and worthwhile; preserve a clear, tested
  fallback when needed.
- In Pandas, prefer columnar arithmetic, boolean masks, `groupby` reductions,
  joins, and built-in vectorized methods. Do not use row-wise loops,
  `.apply`, `.loc`, or `.iloc` unless a vectorized expression is impractical.
  Treat them as a final escape hatch and briefly document why they are needed.

## Test-driven development

- For a new behavior or a bug fix, first add or update a focused test that
  demonstrates the required behavior, then implement the smallest change that
  makes it pass.
- Keep tests parallel to the source layout: a module at `package/path/foo.py`
  should normally have its tests at `tests/path/test_foo.py`.
- Refactor only after the relevant tests pass; keep the test focused on
  observable behavior rather than implementation details.

## Validation

- Run the narrowest relevant checks before declaring a change complete.
- Use the project's configured dependency and test tooling when it exists; do
  not assume a package manager or test runner that the repository does not use.
- Before a commit, run the relevant test suite and do not commit with known
  failing tests unless the user explicitly directs otherwise.

## Running the project

- Use `uv` for the project environment, dependency management, and command
  execution. Do not create or invoke a virtual environment directly.
- Run an individual script with `uv run python <script>.py --help` before
  relying on its command-line interface.
- This repository does not yet have a declared dependency set or test suite.
  When either is added, declare the dependencies in a project configuration,
  use `pytest`, and run tests with `uv run pytest` (or a focused path such as
  `uv run pytest tests/test_runner.py`).
- Keep the README's setup, run, test, formatting, and type-check commands
  current whenever the project tooling changes.

## Interpretation defaults

- Treat requests for temporary or one-off code as a request to use `/tmp` or
  Python's `tempfile` module, rather than adding durable project files.
- When the user asks for barebones code, provide only the minimum executable
  solution; do not add unrequested logging, error handling, or structure.
