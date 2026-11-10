import {
  isRetryableStatus,
  LlmError,
  normalizeToolCalls,
  type ModelReply,
} from "./llm.js";

interface ApiError {
  code?: number | string;
  message?: string;
}

interface ApiResponse {
  choices?: { message?: Record<string, unknown>; finish_reason?: string }[];
  error?: ApiError | null;
  provider?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number; cost?: number };
}

/** Some compatible APIs return content as an array of parts instead of a string. */
function normalizeContent(content: unknown): string | null {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;
  const text = content
    .map((part) => (typeof part === "string" ? part : (part as { text?: string })?.text ?? ""))
    .join("")
    .trim();
  return text || null;
}

export async function parseChatCompletionsResponse(
  res: Response,
  provider: string,
  model: string,
  hintFor: (status: number | undefined, model: string, message: string) => string,
): Promise<ModelReply> {
  const retryAfterHeader = Number(res.headers.get("Retry-After"));
  const retryAfter = Number.isFinite(retryAfterHeader) && retryAfterHeader > 0 ? retryAfterHeader : undefined;
  const bodyText = await res.text();
  let body: ApiResponse;
  try {
    body = JSON.parse(bodyText) as ApiResponse;
  } catch {
    throw new LlmError(
      `non-JSON response (${res.status}): ${bodyText.slice(0, 200)}`,
      `${provider} devolvió una respuesta que no entiendo.`,
      isRetryableStatus(res.status),
      res.status,
      retryAfter,
    );
  }

  if (!res.ok || body.error) {
    const status = res.ok ? Number(body.error?.code) || 500 : res.status;
    const message = body.error?.message ?? bodyText.slice(0, 300);
    throw new LlmError(
      `${provider} ${status}: ${message}`,
      hintFor(status, model, message),
      isRetryableStatus(status),
      status,
      retryAfter,
    );
  }

  const choice = body.choices?.[0];
  const raw = choice?.message;
  if (!raw) {
    throw new LlmError(
      `empty response: ${bodyText.slice(0, 200)}`,
      "El modelo devolvió una respuesta vacía.",
      true,
    );
  }

  const toolCalls = normalizeToolCalls(
    Array.isArray(raw.tool_calls)
      ? raw.tool_calls.map((call: { id?: string; function?: { name?: string; arguments?: unknown } }) => ({
          id: call?.id,
          name: call?.function?.name,
          arguments: call?.function?.arguments,
        }))
      : [],
  );
  return {
    message: {
      role: "assistant",
      content: normalizeContent(raw.content),
      ...(toolCalls ? { tool_calls: toolCalls } : {}),
    },
    finishReason: choice?.finish_reason ?? null,
    provider: body.provider ?? provider.toLowerCase(),
    usage: body.usage
      ? {
          prompt: body.usage.prompt_tokens ?? 0,
          completion: body.usage.completion_tokens ?? 0,
          cost: body.usage.cost,
        }
      : undefined,
  };
}
