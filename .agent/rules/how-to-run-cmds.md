---
trigger: always_on
---

If you want to run command or test, you have to do it in the docker containers.
They are called rechtmaschine-app and rechtmaschine-postgres.

# Smart Rules

## Container Execution
- **App Shell**: `docker exec -it rechtmaschine-app /bin/bash`
- **Database Shell**: `docker exec -it rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db`
- **Logs**: `docker-compose logs -f app`

## Development Workflow
- **Hot Reload**: Enabled and working. Changes to python files should reflect immediately.
- **Restarting**: If needed, use `docker compose restart app`.
- **Testing**:
  - Tests must be run inside the container.
  - If `pytest` is missing: `docker exec -it rechtmaschine-app pip install pytest`
  - Run tests: `docker exec -it rechtmaschine-app pytest` (Note: Ensure `tests/` directory is available/mounted)

## Project Structure
- Code is mounted at `/app`.
- Database is persisted in `postgres_data`.
