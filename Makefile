# Keys come from .env when it exists, otherwise from the environment.
RUN = uv run $(if $(wildcard .env),--env-file .env)

.PHONY: eval answers

# Every eval over the saved answers: search recall, citations, then the judge.
eval:
	$(RUN) evals.py
	$(RUN) citations.py
	$(RUN) judge.py

# Run the agent on all 50 gold questions again. Overwrites answers.jsonl.
answers:
	$(RUN) answers.py
