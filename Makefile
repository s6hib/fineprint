# Keys come from .env when it exists, otherwise from the environment.
RUN = uv run $(if $(wildcard .env),--env-file .env)

.PHONY: eval answers grade

# Search recall, the citation check on the latest answers, then the judge scored on the
# labeled run.
eval:
	$(RUN) evals.py
	$(RUN) citations.py
	$(RUN) judge.py

# Run the agent on all 50 gold questions again. Overwrites answers.jsonl.
answers:
	$(RUN) answers.py

# Have the judge mark the latest answers pass or fail. `make grade CLI=--cli` uses the
# bigger judge through the Claude Code CLI.
grade:
	$(RUN) judge.py --grade $(CLI)
