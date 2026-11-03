---
type: conventions
title: Conventions
description: How this wiki is written, sourced, checked and refreshed. It is a reading aid, not the source of truth.
sources:
  - id: okf-spec
    resource: https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md
    title: Open Knowledge Format v0.2 specification
  - id: okf-parser
    resource: https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/src/reference_agent/bundle/document.py
    title: OKF reference parser
  - id: okf-viewer
    resource: https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/src/reference_agent/viewer/generator.py
    title: OKF reference viewer
  - id: checks
    resource: ../../../tests/test_agent_wiki.py
    title: The wiki checks
generated:
  by: "claude-code/2.1.317"
  at: "2026-11-03T18:50:00Z"
verified: []
status: draft
stale_after: "2026-12-03T00:00:00Z"
---

# Conventions

> **This wiki is not the source of truth.** It is an agent's reading of the code, the tests, the README and the pull requests. If it disagrees with them, they win and the wiki is fixed.

## Rules

1. Every claim names its source: a code anchor, a file, a commit, a pull request, or a command and the commit it ran on. No source, no claim. A chat is not a source.
2. The wiki decides nothing. When records disagree with each other or with the code, the mismatch goes into [open items](open-items.md) and is fixed in a separate change.
3. Wiki work changes only `docs/agent/wiki/` and its checks in `tests/test_agent_wiki.py`.
4. No secrets, keys, tokens, private server addresses or account names on any page.
5. A test count or a benchmark number names the commit and the setup it came from. A count from one machine is not a count for CI.
6. An item is CLOSED only when a record clearly closes it. If unsure, it is UNCLEAR, and UNCLEAR counts as open.
7. `verified` is filled only by a maintainer, with `human:` entries. An agent never verifies its own pages.
8. Refresh after every merged pull request, or once `stale_after` has passed (see [Refreshing](#refreshing)).

## Format

The wiki is an OKF v0.2 bundle.[^okf-spec]

- `index.md` only lists the pages, each with the `description` from its frontmatter (§8). [log.md](log.md) lists the changes grouped by date, newest first (§9). Neither has frontmatter, except `okf_version` in the index.
- Every other page has `type`, `title`, `description`, `sources`, `generated`, `verified`, `status` and `stale_after`.
- Each `sources` entry has a `resource`: a path relative to the page, a URL, or a description with spaces in it, such as a range of commits. A path must exist.
- A claim cites its source with a footnote keyed to the source's `id`, such as `[^readme]`, defined at the end of the page with the source's title.
- Timestamps are quoted ISO 8601 with an explicit offset, such as `"2026-11-03T18:40:00Z"`. Unquoted, PyYAML turns them into datetime objects; the OKF reference parser keeps them as text.[^okf-parser]
- Links are relative. GitHub resolves a leading `/` from the repository root, not from this bundle, and the OKF reference viewer draws graph edges only for relative links.[^okf-viewer]
- Code anchors are written `path::name` from the repository root, for example `app/pdf_engine.py::build_index`, or `path::Class.method`. Never line numbers: they go stale without anyone noticing.
- Dates are UTC. Git author dates in this repository carry +08:00, so a commit can show the next day. Pull request dates are GitHub's `merged_at` or `closed_at`.
- Actors follow OKF §7: `claude-code/<version>` for an agent, `human:<github-login>` for a maintainer.
- `status` is `draft` until a maintainer has read the page.

## Checks

`tests/test_agent_wiki.py` runs with the rest of the test suite, so CI checks every pull request.[^checks] It fails when:

- a page has no frontmatter that parses, lacks one of the keys above, or has a timestamp that is not a quoted ISO 8601 string with an offset;
- `generated.by` is not an actor, or a `verified` entry has an actor that does not start with `human:`;
- the index misses a file, or an entry does not contain the page's `description`;
- a log heading is not an ISO date, or the dates are not newest first;
- a link starts with `/`, or a relative link or a `sources` path points at nothing;
- a footnote has no definition, or its id is not one of the page's sources;
- a code anchor names a file or a name that does not exist;
- a page cites a line number instead of an anchor, or holds text that looks like a key, a token or a private server address.

The links, anchors, line numbers and secrets in `AGENTS.md` are checked too. It never compares `stale_after` with the clock, so the suite cannot start failing on a date. To list the pages that are stale now, run `python -m tests.test_agent_wiki` from the repository root.

## Refreshing

After a pull request merges:

1. Add its row to the [change index](change-index.md).
2. Update [current state](current-state.md): the latest changes, the checks and what is waiting.
3. Move items in [open items](open-items.md): close what the change closes, naming it, and add what it leaves open.
4. Add a trap to [gotchas](gotchas.md) if the change found one, with an anchor.
5. Update [architecture](architecture.md) if modules, stored files or locks changed.
6. On every page you checked, update the commit it was checked against, `generated.at` and `stale_after`. Add an entry to the [log](log.md).

[^okf-spec]: Open Knowledge Format v0.2 specification
[^okf-parser]: OKF reference parser
[^okf-viewer]: OKF reference viewer
[^checks]: The wiki checks
