.PHONY: install playground

install:
	uv sync

playground:
	agents-cli playground --port 8084
