### Changelog – WebAI to API

#### v0.8.0 – Unreleased

##### Changed

- **Upgraded `gemini-webapi` to the latest master (>= 2.0).** This introduces dynamic
  model discovery per account via `list_models()` / `resolve_model()`.
- **Hard switch to discovered model names** (`gemini-pro`, `gemini-flash`,
  `gemini-flash-lite`). All `gemini-3-*` / `gemini-2-*` hardcoded names were removed.
  Legacy/variant model names sent by clients are now resolved at request time and fall
  back to the account default when unknown.
- **Extended thinking is now a flag** (`extended_thinking`), not a separate model.
  Enable it by appending `thinking` to a model name or by sending `extended_thinking: true`
  on OpenAI `/chat/completions` requests.
- The admin API now returns the account's dynamically discovered models instead of a
  static list, so the dashboard dropdown stays in sync with what the account can use.

##### Added

- Centralized `app/services/model_resolver.py` for model resolution and extended-thinking
  detection (replaces the per-endpoint alias tables).

---

#### v0.4.0 – 2025-06-27

##### Added

- Displayed a user message explaining how to use the `gpt4free` server.

##### Fixed

- Resolved execution issue on Windows 11.
- Improved error handling with appropriate user-facing messages.

##### Changed

- Updated internal libraries and dependencies.

---

#### v0.3.0 – 2025-06-25

##### Added

- Improved server startup information display, including available services and API endpoints.
- Added a new method using the [gpt4free v0.5.5.5](https://github.com/xtekky/gpt4free) library, which also functions as a fallback.
- Introduced support for switching between models using keyboard shortcuts (keys `1` and `2`) in the terminal.
- WebAI-to-API now uses your browser and cookies **only for Gemini**, resulting in faster performance.
- `gpt4free` integration provides access to multiple providers (ChatGPT, Gemini, Claude, DeepSeek, etc.), ensuring continuous availability of various models.

##### Changed

- Updated internal libraries.
- Upgraded to [Gemini API v1.14.0](https://github.com/HanaokaYuzu/Gemini-API).

##### Fixed

- Ensured compatibility with Windows (tested on Windows 11).
