---
name: ruff-check
description: Run ruff lint and format checks on gwanbo-ocr source, tests, and scripts
license: MIT
compatibility: opencode
---

## What I do
- Run `ruff check src tests scripts` for linting
- Run `ruff format --check src tests scripts` for format validation
- Report all violations with file paths and line numbers

## When to use me
Use after every code change. Verify lint and formatting pass before committing.

## Project conventions
- line-length=100, target-version=py312
- Rules: E, F, I, UP, B
- Ignores: B008 (Typer), E501 (long literals)
- Test files: E402 per-file-ignore
