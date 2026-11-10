import { config } from "../config.js";
import { LlmError, withRetries, type ChatMessage, type ModelReply } from "./llm.js";
import { parseChatCompletionsResponse } from "./chatCompletions.js";

const REQUEST_TIMEOUT_MS = 90_000;

function hintFor(status: number | undefined, model: string, message: string): string {
  switch (status) {
    case 401:
      return "La API key de LiteLLM no es válida o fue revocada. Revisala en la configuración web.";
    case 403:
      return "LiteLLM rechazó el pedido (permisos del proxy o del modelo).";
    case 404:
      return `El endpoint o el alias de modelo "${model}" no existe en LiteLLM. Revisá la URL y el modelo.`;
    case 429:
      return "LiteLLM está limitando los pedidos. Probá de nuevo en un minuto.";
    case 408:
      return "El modelo tardó demasiado en responder.";
    default:
      if (status && status >= 500) return "LiteLLM o el proveedor del modelo están caídos ahora mismo.";
      return message;
  }
}

export async function singleCall(messages: ChatMessage[], tools: unknown, model: string): Promise<ModelReply> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const endpoint = `${config.LITELLM_BASE_URL.replace(/\/+$/, "")}/chat/completions`;

  let res: Response;
  try {
    res = await fetch(endpoint, {
      method: "POST",
      signal: controller.signal,
      headers: {
        ...(config.LITELLM_API_KEY
          ? { Authorization: ["Bearer", config.LITELLM_API_KEY].join(" ") }
          : {}),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages,
        tools,
        tool_choice: "auto",
        temperature: 0.2,
        max_tokens: 4_000,
      }),
    });
  } catch (err) {
    const aborted = (err as Error).name === "AbortError";
    throw new LlmError(
      aborted ? `timeout after ${REQUEST_TIMEOUT_MS}ms` : `network error: ${(err as Error).message}`,
      aborted
        ? "El modelo no respondió a tiempo."
        : "No pude conectarme a LiteLLM (revisá la URL y la conexión de red).",
      true,
    );
  } finally {
    clearTimeout(timer);
  }

  return parseChatCompletionsResponse(res, "LiteLLM", model, hintFor);
}

export function callModel(messages: ChatMessage[], tools: unknown, model: string): Promise<ModelReply> {
  return withRetries(model, () => singleCall(messages, tools, model));
}

export function routingSummary(): string {
  return "LiteLLM · OpenAI-compatible Chat Completions";
}
