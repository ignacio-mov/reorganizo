import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

process.env.DATABASE_PATH = ":memory:";
const { config } = await import("../src/config.js");
const { checkModel, LlmError } = await import("../src/agent/llm.js");
const { singleCall } = await import("../src/agent/litellm.js");

const originalFetch = globalThis.fetch;
const originalProvider = config.LLM_PROVIDER;
const originalBaseUrl = config.LITELLM_BASE_URL;
const originalApiKey = config.LITELLM_API_KEY;

describe("LiteLLM adapter", () => {
  beforeEach(() => {
    config.LLM_PROVIDER = "litellm";
    config.LITELLM_BASE_URL = "http://localhost:4000/v1/";
    config.LITELLM_API_KEY = "test-litellm-key";
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    config.LLM_PROVIDER = originalProvider;
    config.LITELLM_BASE_URL = originalBaseUrl;
    config.LITELLM_API_KEY = originalApiKey;
  });

  it("sends Chat Completions requests and normalizes tool calls", async () => {
    globalThis.fetch = async (input, init) => {
      assert.equal(input, "http://localhost:4000/v1/chat/completions");
      assert.equal(new Headers(init?.headers).get("authorization"), "Bearer test-litellm-key");
      const request = JSON.parse(String(init?.body)) as {
        model: string;
        messages: unknown[];
        tools: unknown[];
        tool_choice: string;
      };
      assert.equal(request.model, "calendar-assistant");
      assert.equal(request.tool_choice, "auto");
      assert.equal(request.tools.length, 1);
      assert.equal(request.messages.length, 1);
      return Response.json({
        model: "calendar-assistant",
        choices: [
          {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "create_event", arguments: '{"title":"Study"}' },
                },
              ],
            },
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 5, cost: 0.0002 },
      });
    };

    const reply = await singleCall(
      [{ role: "user", content: "Schedule study time" }],
      [{ type: "function", function: { name: "create_event" } }],
      "calendar-assistant",
    );

    assert.equal(reply.provider, "litellm");
    assert.equal(reply.finishReason, "tool_calls");
    assert.equal(reply.message.tool_calls?.[0]?.function.name, "create_event");
    assert.deepEqual(reply.usage, { prompt: 12, completion: 5, cost: 0.0002 });
  });

  it("returns a useful authentication hint for rejected keys", async () => {
    globalThis.fetch = async () =>
      Response.json({ error: { message: "invalid key" } }, { status: 401 });

    await assert.rejects(
      singleCall([{ role: "user", content: "hello" }], [], "calendar-assistant"),
      (error: unknown) =>
        error instanceof LlmError &&
        error.status === 401 &&
        error.userHint.includes("LiteLLM"),
    );
  });

  it("does not pass diagnostics when the model ignores tools", async () => {
    globalThis.fetch = async () =>
      Response.json({
        choices: [{ finish_reason: "stop", message: { role: "assistant", content: "ok" } }],
      });

    const result = await checkModel("calendar-assistant");
    assert.equal(result.ok, false);
    assert.match(result.error ?? "", /herramientas/i);
  });
});
