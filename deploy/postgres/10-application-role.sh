#!/bin/sh
# Runs once, when the database volume is first created.
#
# The platform's tables enforce tenant isolation with row-level security, and a superuser
# is exempt from every such policy. POSTGRES_USER is a superuser, so the platform is given
# a role of its own: it owns the database and can create its schema, but it is not a
# superuser and cannot bypass the policies it installs.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE ROLE "$APP_DB_USER" LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '$APP_DB_PASSWORD';
ALTER DATABASE "$POSTGRES_DB" OWNER TO "$APP_DB_USER";
ALTER SCHEMA public OWNER TO "$APP_DB_USER";
SQL
