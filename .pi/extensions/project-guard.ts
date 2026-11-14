/**
 * Project guard for LlamaServer agent sessions.
 *
 * The small model's failure modes in this repo are mechanical: it re-runs the
 * same command, tries a local Xcode build (there is no local build env), or
 * curls the GitHub API (needs auth, returns 404). A block carries the
 * replacement command, so the model can correct course in the same turn.
 *
 * Loaded explicitly by scripts/pi4b.sh:
 *   pi --no-extensions -e .pi/extensions/project-guard.ts ...
 */

import { type ExtensionAPI, isToolCallEventType } from "@earendil-works/pi-coding-agent";

let lastBash = "";

export default function projectGuard(pi: ExtensionAPI) {
  pi.on("tool_call", (event) => {
    if (!isToolCallEventType("bash", event)) return;

    const cmd = event.input.command.trim().replace(/\s+/g, " ");
    const repeated = cmd === lastBash;
    lastBash = cmd;

    if (repeated) {
      return {
        block: true,
        terminate: true,
        reason:
          "Identical command already ran. Read the previous output, change approach, " +
          "or stop and report. Do not retry it.",
      };
    }

    if (/\bxcodebuild\b/.test(cmd)) {
      return {
        block: true,
        reason:
          "No local Xcode build in this repo. Push, then run " +
          "`bash scripts/ci-wait.sh <run-id>`; use `bash scripts/ci-logs.sh <run-id>` for the failed log.",
      };
    }

    if (/curl\b[^\n]*api\.github\.com/.test(cmd)) {
      return {
        block: true,
        reason:
          "The GitHub API needs auth; curl returns 404. Use " +
          "`bash scripts/ci-logs.sh <run-id>` (gh-based) or `bash scripts/ci-wait.sh <run-id>`.",
      };
    }
  });
}
