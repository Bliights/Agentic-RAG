# -------------- Install dependencies --------------
install:
	@echo Install the dependencies with uv and pre commit...
	uv sync
	uv run pre-commit install	

update:
	@echo Updating all dependencies of the environment...
	uv lock --upgrade
	uv sync
	uv run pre-commit autoupdate

# ------------------ Formating ---------------------
lint:
	@echo Check with ruff...
	uv run ruff check .

format:
	@echo Format with Ruff...
	uv run ruff format .

fix:
	@echo Fix with Ruff...
	uv run ruff check --fix .

# ------------------- Pre-commit -------------------
pre-commit:
	@echo Run pre-commit...
	uv run pre-commit run --all-files

# ------------------ Scripts ---------------------
launch-db:
	@echo Starting the Qdrant database...
	docker-compose up

process-visual:
	@echo Starting visual databse processing...
	uv run python -m src.scripts.retriever.visual.processing

process-textual:
	@echo Starting textual databse processing...
	uv run python -m src.scripts.retriever.textual.processing

train-scorer:
	@echo Starting scorer training...
	uv run python -m src.scripts.scorer.training