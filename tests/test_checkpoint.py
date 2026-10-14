"""Offline fault-injection tests. No models, corpus downloads, or API keys needed."""

import asyncio
import copy
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from contextlib import ExitStack, closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ideascientist.harness import runner, tools
from ideascientist.harness.checkpoint import Checkpoint, CheckpointError, Scope
from ideascientist.harness.client import AgenticLLMClient


class Crash(BaseException):
    """Simulate process death without being converted to a tool error."""


def message(*calls, content="done"):
    result = {"role": "assistant", "content": content,
              "reasoning_content": "reasoning must survive recovery",
              "_opaque_provider_state": [{"encrypted": "test-continuation"}]}
    if calls:
        result["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)]
    return result


def edit(path, command="create", **kwargs):
    return "edit_doc", dict(path=path, command=command, **kwargs)


def spawn(role):
    return "spawn_subagent", dict(role=role, task=role + " task")


class ScriptedAPI:
    def __init__(self, agentic_reader=False):
        self.requests = {}
        self.counts = Counter()
        self.reads = Counter()
        self.lock = threading.Lock()
        self.agentic_reader = agentic_reader
        self.script = {
            "orch:0": message(content="premature stop"),
            "orch:1": message(edit("plan.md", content="original plan\n"),
                              spawn("gap_finder"), spawn("innovator")),
            "orch:2": message(edit("gaps.md", command="view"), spawn("report_writer")),
            "orch:3": message(),
            "sub:gap_finder:gap_finder#1:0": message(content="premature stop"),
            "sub:gap_finder:gap_finder#1:1": message(
                ("keyword_search", {"queries": ["physics"], "k": 1}),
                ("read_paper", {"paper_id": 7, "goals": ["first goal"]})),
            "sub:gap_finder:gap_finder#1:2": message(
                ("read_paper", {"paper_id": 7, "goals": ["second goal"]}),
                edit("gaps.md", command="insert", insert_line=0,
                     content="A sufficiently detailed research gap.")),
            "sub:gap_finder:gap_finder#1:3": message(),
            "sub:innovator:innovator#2:0": message(edit(
                "candidates.md", content="A sufficiently detailed candidate idea.")),
            "sub:innovator:innovator#2:1": message(),
            "sub:report_writer:report_writer#3:0": message(edit(
                "report.json", content='{"idea": "a grounded physics research idea"}')),
            "sub:report_writer:report_writer#3:1": message(),
        }

    def chat(self, client, messages, tools=None, max_retries=6):
        key = client._step_name
        if key.startswith("orchestrator/"):
            reply = message(content="digest for " + messages[-1]["content"])
        elif key.startswith("sub:paper_reader:"):
            # Each reader invocation has its own stable conversation ID.
            if key.endswith(":0"):
                reply = message(edit("papers/7.md", command="insert", insert_line=0,
                                     content="agentic reader digest"))
            else:
                reply = message()
        else:
            reply = self.script[key]
        with self.lock:
            self.counts[key] += 1
            self.requests.setdefault(key, []).append(copy.deepcopy(messages))
        client._call_counter += 1
        client._account_usage({"prompt_tokens": 11, "completion_tokens": 7})
        client._ensure_log()
        record = {"step": key, "messages": messages, "response": reply}
        if client._shared_log is not None:
            client._shared_log.write(record)
        else:
            client._log_writer.write(record)
        return {"choices": [{"message": copy.deepcopy(reply)}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_pipeline(self, path, api, *, resume=False, **kwargs):
        return asyncio.run(self.run_pipeline_async(path, api, resume=resume, **kwargs))

    async def run_pipeline_async(self, path, api, *, resume=False, **kwargs):
        def reader_task(pid, goals, prior):
            api.reads["reader_task"] += 1
            return json.dumps([pid, goals, prior])

        def search(*args, **kwargs):
            api.reads["search"] += 1
            return [{"id": 7}]

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(AgenticLLMClient, "chat",
                lambda client, *a, **kw: api.chat(client, *a, **kw)))
            stack.enter_context(patch.object(runner, "build_reader_task",
                side_effect=reader_task))
            stack.enter_context(patch.object(tools, "keyword_search", side_effect=search))
            return await runner.run_ideation(
                None if resume else "Important physics problem", path,
                model=None if resume else "test-model", resume=resume, **kwargs)

    def artifacts(self, path):
        return {p.relative_to(path / "outputs").as_posix(): p.read_text(encoding="utf-8")
                for p in (path / "outputs").rglob("*") if p.is_file()}

    def assert_same_run(self, expected, actual, reference_api, api, baseline, recovered):
        self.assertEqual(self.artifacts(baseline), self.artifacts(recovered))
        self.assertEqual({k: v for k, v in expected.items() if k != "run_dir"},
                         {k: v for k, v in actual.items() if k != "run_dir"})
        self.assertEqual(reference_api.requests, api.requests)
        self.assertEqual(reference_api.counts, api.counts)
        self.assertEqual(reference_api.reads, api.reads)

    def test_interrupt_after_each_completed_operation(self):
        baseline = self.root / "baseline"
        reference_api = ScriptedAPI()
        expected = self.run_pipeline(baseline, reference_api)
        # Discover the real journal boundaries, then fail after each one. Every
        # run still uses the production loops, tools, SQLite, and file writes.
        with closing(sqlite3.connect(baseline / "checkpoint" / "state.sqlite3")) as db:
            keys = [r[0] for r in db.execute("SELECT key FROM calls ORDER BY rowid")]
        original_call = Scope.call
        for index, target in enumerate(keys):
            with self.subTest(target=target):
                recovered = self.root / f"recovery_{index}"
                api = ScriptedAPI()
                fired = []

                def interrupt(scope, key, *args, **kwargs):
                    result = original_call(scope, key, *args, **kwargs)
                    if scope.name + "/" + key == target and not fired:
                        fired.append(True)
                        raise Crash(target)
                    return result

                with patch.object(Scope, "call", interrupt), self.assertRaises(Crash):
                    self.run_pipeline(recovered, api)
                old_logs = {p: p.read_bytes() for p in (recovered / "logs").rglob("*.json")}
                actual = self.run_pipeline(recovered, api, resume=True)
                self.assert_same_run(expected, actual, reference_api, api, baseline, recovered)
                for path, content in old_logs.items():
                    self.assertEqual(path.read_bytes(), content)
        self.assertEqual(expected["papers_read"], 2)
        self.assertEqual(expected["total_tokens"], sum(reference_api.counts.values()) * 18)

    def test_agentic_reader_resume(self):
        with patch.dict(os.environ, {"IDEASCIENTIST_READER_SINGLE_TURN": "0"}):
            baseline, recovered = self.root / "baseline", self.root / "recovered"
            reference_api, api = ScriptedAPI(True), ScriptedAPI(True)
            expected = self.run_pipeline(baseline, reference_api)
            original_call = Scope.call

            def interrupt(scope, key, *args, **kwargs):
                result = original_call(scope, key, *args, **kwargs)
                if "/reader/" in scope.name and key == "edit":
                    raise Crash()
                return result

            with patch.object(Scope, "call", interrupt), self.assertRaises(Crash):
                self.run_pipeline(recovered, api)
            actual = self.run_pipeline(recovered, api, resume=True)
            self.assert_same_run(expected, actual, reference_api, api, baseline, recovered)

    def test_unreceived_model_response_is_retried(self):
        api = ScriptedAPI()
        original = api.chat

        def disconnect(client, *args, **kwargs):
            if client._step_name == "sub:gap_finder:gap_finder#1:2":
                raise ConnectionError("lost response")
            return original(client, *args, **kwargs)

        with patch.object(api, "chat", disconnect), self.assertRaises(CheckpointError):
            self.run_pipeline(self.root, api)
        previous = api.counts.copy()
        self.run_pipeline(self.root, api, resume=True)
        for key, count in previous.items():
            self.assertEqual(api.counts[key], count)
        self.assertEqual(api.counts["sub:gap_finder:gap_finder#1:2"], 1)

    def test_parallel_readers_have_independent_checkpoints(self):
        baseline, recovered = self.root / "baseline", self.root / "recovered"
        reference_api, api = ScriptedAPI(), ScriptedAPI()
        for endpoint in (reference_api, api):
            endpoint.script["sub:innovator:innovator#2:0"] = message(
                ("read_paper", {"paper_id": 8, "goals": ["independent goal"]}),
                edit("candidates.md", content="A sufficiently detailed candidate idea."))
        expected = self.run_pipeline(baseline, reference_api)
        original_call = Scope.call

        def interrupt(scope, key, *args, **kwargs):
            result = original_call(scope, key, *args, **kwargs)
            if scope.name == "orchestrator/innovator#2/tool:0:0" and key == "chat":
                raise Crash()
            return result

        with patch.object(Scope, "call", interrupt), self.assertRaises(Crash):
            self.run_pipeline(recovered, api)
        actual = self.run_pipeline(recovered, api, resume=True)
        self.assert_same_run(expected, actual, reference_api, api, baseline, recovered)
        self.assertEqual(actual["papers_read"], 3)

    def test_completed_run_is_noop_and_settings_are_checked(self):
        api = ScriptedAPI()
        expected = self.run_pipeline(self.root, api)
        counts = api.counts.copy()
        with patch.dict(os.environ, {"IDEASCIENTIST_API_KEY": "rotated-test-secret"}):
            self.assertEqual(expected, self.run_pipeline(self.root, api, resume=True))
        self.assertEqual(api.counts, counts)
        with self.assertRaisesRegex(CheckpointError, "configuration"):
            self.run_pipeline(self.root, api, resume=True, max_turns=3)
        with patch.dict(os.environ, {"IDEASCIENTIST_TEXT_TOOLS": "1"}):
            with self.assertRaisesRegex(CheckpointError, "configuration"):
                self.run_pipeline(self.root, api, resume=True)
        with self.assertRaisesRegex(CheckpointError, "already has"):
            self.run_pipeline(self.root, api)
        (self.root / "outputs" / "plan.md").write_text("user edit", encoding="utf-8")
        with self.assertRaisesRegex(CheckpointError, "Artifact changed"):
            self.run_pipeline(self.root, api, resume=True)
        self.assertEqual((self.root / "outputs" / "plan.md").read_text(), "user edit")

    def test_old_runs_and_concurrent_attempts_are_rejected(self):
        with self.assertRaisesRegex(CheckpointError, "no checkpoint"):
            with Checkpoint(self.root, resume=True):
                pass
        with Checkpoint(self.root) as cp:
            cp.prepare({"test": True})
            with self.assertRaisesRegex(CheckpointError, "Another process"):
                with Checkpoint(self.root, resume=True):
                    pass

    def test_real_process_death_around_file_materialization(self):
        # Exit without Python cleanup, both before and after os.replace, to
        # exercise SQLite recovery and release of the operating-system lock.
        for after_replace in (False, True):
            with self.subTest(after_replace=after_replace):
                path = self.root / str(after_replace)
                script = '''
import os, sys
from unittest.mock import patch
from ideascientist.harness.checkpoint import Checkpoint
with Checkpoint(sys.argv[1]) as cp:
    cp.prepare({"test": True})
    replace = os.replace
    def die(*args):
        if sys.argv[2] == "True":
            replace(*args)
        os._exit(17)
    with patch("os.replace", die):
        cp.scope("test").write("append", "plan.md", "exactly once\\n")
'''
                result = subprocess.run([sys.executable, "-c", script, str(path), str(after_replace)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 17, result.stderr)
                with Checkpoint(path, resume=True) as cp:
                    cp.prepare({"test": True})
                    cp.scope("test").write("append", "plan.md", "exactly once\n")
                self.assertEqual((path / "outputs" / "plan.md").read_text(), "exactly once\n")

    def test_replay_detects_changed_inputs(self):
        with Checkpoint(self.root) as cp:
            scope = cp.scope("test")
            scope.call("op", {"input": 1}, lambda: "saved")
            with self.assertRaisesRegex(CheckpointError, "Replay inputs changed"):
                scope.call("op", {"input": 2}, lambda: "different")

    def test_cancel_waits_for_inflight_worker_to_checkpoint(self):
        api = ScriptedAPI()
        entered, release = threading.Event(), threading.Event()
        original = api.chat

        def wait_for_cancel(client, *args, **kwargs):
            if client._step_name == "sub:gap_finder:gap_finder#1:1":
                entered.set()
                if not release.wait(10):
                    raise AssertionError("cancellation test did not release worker")
            return original(client, *args, **kwargs)

        async def cancel():
            task = asyncio.create_task(self.run_pipeline_async(self.root, api))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 10))
                task.cancel()
                await asyncio.sleep(0)  # deliver cancellation while HTTP is in flight
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()

        with patch.object(api, "chat", wait_for_cancel):
            asyncio.run(cancel())
        completed = api.counts.copy()
        self.run_pipeline(self.root, api, resume=True)
        for key, count in completed.items():
            self.assertEqual(api.counts[key], count)

    def test_crlf_artifacts_and_credential_exclusion(self):
        from ideascientist.harness.checkpoint import runtime_config
        with patch.dict(os.environ, {"IDEASCIENTIST_API_KEY": "do-not-save-me",
                                     "OPENAI_API_KEY": "do-not-save-me"}):
            self.assertNotIn("do-not-save-me", json.dumps(runtime_config()))
        with Checkpoint(self.root) as cp:
            cp.prepare({"test": True})
            cp.scope("test").write("op", "plan.md", "physics\r\n物理\r\n")
        with Checkpoint(self.root, resume=True) as cp:
            cp.prepare({"test": True})


if __name__ == "__main__":
    unittest.main()
