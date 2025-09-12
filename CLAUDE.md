# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Monte Carlo simulator that predicts the composition of the Quorum of the Twelve Apostles of the Church of Jesus Christ of Latter-day Saints over time using actuary tables and data about current General Authorities.

## Development Commands

- **Install dependencies**: `uv add <package> --bounds=exact` or `uv sync`
- **Run linting**: `uv run ruff check --fix`
- **Format code**: `uv run ruff format`
- **Type checking**: `uv run pyrefly check`
- **Development mode**: `uv add --dev <package> --bounds=exact` to add dev dependencies

## Architecture

- **Package management**: Uses `uv` for dependency management and building
- **Core module**: `src/apostle_predictor/` contains the main package
- **Actuary data**: `actuary_table.py` contains 2022 Social Security Administration Period Life Table data with death probabilities and life expectancy by age and gender
- **Data format**: Uses pandas DataFrames for handling actuary table data

## Key Components

- `actuary_table.py`: Contains SSA mortality data (ACTUARY_DATAFRAME) with columns for age, death probability, number of lives, and life expectancy for both male and female populations
- Data sourced from SSA 2025 Trustees Report for 2022 period life tables
- `leader_models.py`: Contains models and scraping tools to create a biography of leaders in the church (including birthdate, age, number of conference talks, assignments, etc...) relevant to predicting future ages/death, and future leadership positions.

## Project Structure

```
src/apostle_predictor/
├── __init__.py                     # Package initialization
├── actuary_table.py               # SSA mortality data tables
├── simulation.py                  # Monte Carlo simulation engine
├── data_converters.py            # Convert Pydantic models to Leader objects
└── models/                       # Data models for web scraping
    ├── __init__.py
    ├── leader_models.py          # Core Leader model and scraping logic
    ├── biography_models.py       # Pydantic models for individual biography pages
    ├── organization_models.py    # Pydantic models for leadership collection pages
    └── seventies_models.py       # Pydantic models for General Authority Seventies API
```

## Web Scraping Architecture

The system scrapes leadership data from churchofjesuschrist.org using a structured approach:

1. **Collection Pages**: Leadership overview pages (First Presidency, Quorum of Twelve, etc.) contain `__NEXT_DATA__` JSON with canonical URLs
2. **Individual Biographies**: Each leader's biography page contains structured JSON data with birth dates, calling history, and seniority information
3. **API Endpoints**: General Authority Seventies data comes from a dedicated API endpoint
4. **Caching**: Uses `auto_pydantic_cache` decorator for HTTP request caching to reduce server load whenever you can write a function returning a result using pydantic.

## Key Data Models

- **Leader**: Core model representing a church leader with birth date, callings, and biographical data
- **Calling**: Represents a church position with type, status, start date, and seniority
- **CallingType**: Enum for position types (Prophet, Apostle, Counselor, etc.)
- **SimulationResult**: Contains outcomes from Monte Carlo runs including events and statistics

## Monte Carlo Simulation

The simulation engine (`simulation.py`) models:
- **Mortality**: Uses SSA actuarial tables to predict death probabilities by age
- **Succession Rules**: Senior apostle (lowest seniority number) becomes Prophet
- **New Callings**: General Authority Seventies are called to fill vacant apostle positions
- **Time Progression**: Daily iteration with probabilistic death events

## Important Notes

- **Never use `pip` or `python` directly**: Always use `uv run python` or `uv run <command>`
- **HTTP Redirects**: Church website returns 302 redirects; use `follow_redirects=True` in httpx requests
- **Data Validation**: All web scraping uses Pydantic models for type-safe JSON parsing
- **Caching Strategy**: Individual biography pages are cached to minimize repeated requests during development
- **Use Claude Agents**: For complex multi-step tasks, code reviews, linting, and documentation synchronization, leverage specialized Claude Code agents to maintain code quality and consistency

## Git Workflow

**Commit frequently when functionality is working** - This project benefits from incremental commits to track progress and maintain working states.

### Conventional Commit Format

Use conventional commit messages with the following format:
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Common commit types:**
- `feat:` - New features or functionality
- `fix:` - Bug fixes
- `refactor:` - Code refactoring without changing behavior
- `test:` - Adding or updating tests
- `docs:` - Documentation changes
- `chore:` - Maintenance tasks, dependency updates

**Examples:**
- `feat(simulation): add Monte Carlo engine for apostolic succession`
- `fix(scraper): handle missing birth dates in leader data`
- `refactor(models): simplify Pydantic-based scraping architecture`
- `test(scraper): add comprehensive web scraping tests`
- `docs: update CLAUDE.md with git workflow guidelines`
- `chore: update dependencies to latest versions`

**When to commit:**
- After implementing a complete feature or fix
- When tests are passing and functionality is verified
- Before starting major refactoring or architectural changes
- After successfully resolving bugs or issues

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.