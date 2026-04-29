# Run targets from this directory (clone root) so Gemini CLI picks up `.gemini/settings.json`.
# The codelab calls this folder "simple-travel-agent"; in this repo it is `adk-gemini-cli-workshop`.
#
# Gemini CLI needs Node 20+ (Node 18 crashes in string-width with "Invalid regular expression flags").
# With nvm: `nvm install` && `nvm use` (see `.nvmrc`).
#
# Vertex + ModelNotFoundError (404): preview IDs in your CLI defaults may not exist for your project/region.
# `test-step1` pins `-m gemini-2.5-flash` (same as the workshop agent). If the location is still wrong, run:
#   make test-step1-vertex-us
# or: export GOOGLE_CLOUD_LOCATION=us-central1  (some preview models: `global` — see Cloud docs)
# Alternative: use Google AI Studio (GOOGLE_API_KEY) and turn off Vertex for the CLI.
.PHONY: test-step1 test-step1-vertex-us web-4steps _check_node

GEMINI_PING := Reply with exactly one word: ok

_check_node:
	@node -e 'const v=+process.version.slice(1).split(".")[0]; if(v<20){console.error("Gemini CLI requires Node 20+. You have Node "+process.version+". Try: nvm use"); process.exit(1);}'

# One-shot Gemini API check: success means auth works; failures often show 403 PERMISSION_DENIED.
# `--skip-trust`: non-interactive runs (make/CI) cannot confirm trust; this repo is the workshop root you opened on purpose.
# `-m gemini-2.5-flash`: avoids CLI default preview models that often return 404 on Vertex.
test-step1: _check_node
	gemini --skip-trust -m gemini-2.5-flash -p '$(GEMINI_PING)'

# Forces us-central1 when your shell uses a region whose catalog does not include the pinned model.
test-step1-vertex-us: _check_node
	GOOGLE_CLOUD_LOCATION=us-central1 gemini --skip-trust -m gemini-2.5-flash -p '$(GEMINI_PING)'

# ADK Web over bundled step agents (step01–step04). Pick an agent in the UI. Stays running until Ctrl-C.
# For your own code, use: uv run adk web .  and select mysolution/
web-4steps:
	uv run adk web steps/
