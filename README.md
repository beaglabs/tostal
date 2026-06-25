# Tostal Sci-data Platform

Cloud-hosted geoscience data platform with per-customer Icechunk stores, Jupyter notebook frontend, and Murmurative task models.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your credentials
make docker-up
```

## Architecture

- **gateway/** — NextJS OAuth gateway (BetterAuth + Stripe)
- **api/** — FastAPI backend (OpenAPI 3.1, SQLAlchemy, Icechunk)
- **frontend/** — Custom Jupyter notebook UI (React + ipywidgets)
- **workflows/** — Temporal durable workflows
- **storage/** — Terraform IaC + provisioning scripts
- **docs/** — Architecture and design docs

## Development

```bash
make dev-api       # Start FastAPI on :8000
make dev-gateway   # Start NextJS on :3000
make test          # Run test suite
make lint          # Run linters
make migrate       # Run Alembic migrations
```

See `docs/tostal-implementation-plan.md` for the full architecture plan.