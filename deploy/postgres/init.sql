-- Runs once, when the Postgres volume is first created.
-- The agent owns the "voiceops" database (created by POSTGRES_DB, schema via Alembic).
-- The mock enterprise systems get their own database, like a separate vendor system would.
CREATE DATABASE mocks OWNER voiceops;
