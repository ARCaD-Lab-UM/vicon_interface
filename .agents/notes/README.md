# Notes

Dated records for humans and coding agents, read on demand and never loaded automatically. The code,
the launch file and `param/*.yaml` win over anything written here; a note can be stale.

Only the files whitelisted in this folder's `.gitignore` are tracked (`README.md`,
`ros2-jazzy-migration.md`). Everything else that lives here is a local working note. Add a file to the
whitelist to publish it.

## Conventions

- Write little, and only with the user's agreement. Not every finding deserves a note. Propose the text
  in the conversation and add it here after the user says yes, never on your own.
- A note holds a fact that is settled, verified, worth reading again, and not already stated by the
  code, the YAML or `README.md`. Raw measurements, hypotheses, dead ends and the story of how something
  was found stay in the conversation. Keep a note to a few lines and delete what stops being true.
- Every note starts with a header block: `Date`, `Status` (`current`, or `unverified since <date>`),
  `Evidence` (what was checked and how), then a short `Summary`.
- Notes are append-only. Correct a conclusion by adding a dated update under it, not by rewriting.
- Tracked notes mention only what concerns this package. Local notes may describe the machine they live on.
- File names: one `.md` extension, lowercase, hyphens. The root `.gitignore` ignores any directory named
  `build`, `lib` or `dist` and any file ending in `.log`, so avoid those names.
- English only.

## Tracked notes

| File | What |
|---|---|
| `ros2-jazzy-migration.md` | What changed when the package moved from Ubuntu 22.04 / Humble to 24.04 / Jazzy, and what a fresh 24.04 machine needs |
