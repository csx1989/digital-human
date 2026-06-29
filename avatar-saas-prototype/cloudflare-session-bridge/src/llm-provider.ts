const LLM_TIMEOUT_MS = 15_000;

export interface LlmMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface LlmResponse {
  text: string;
  provider: string;
  model: string;
  latencyMs: number;
}

interface ChatCompletionResponse {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type?: string; text?: string }>;
    };
  }>;
}

export class OpenAiCompatibleProvider {
  #baseUrl: string;
  #apiKey?: string;
  #model: string;
  #providerName: string;

  constructor(options?: {
    apiKey?: string;
    baseUrl?: string;
    model?: string;
    providerName?: string;
  }) {
    this.#baseUrl = (options?.baseUrl ?? "").replace(/\/$/, "");
    this.#apiKey = options?.apiKey;
    this.#model = options?.model ?? "unknown";
    this.#providerName = options?.providerName ?? "openai-compatible";
  }

  get providerName(): string {
    return this.#providerName;
  }

  get model(): string {
    return this.#model;
  }

  async chat(messages: LlmMessage[]): Promise<LlmResponse> {
    const startedAt = Date.now();

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.#apiKey) {
      headers.Authorization = `Bearer ${this.#apiKey}`;
    }

    const response = await fetch(`${this.#baseUrl}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: this.#model,
        stream: false,
        messages,
      }),
      signal: AbortSignal.timeout(LLM_TIMEOUT_MS),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `LLM ${this.#providerName} ${response.status}: ${body.substring(0, 500)}`,
      );
    }

    const completion = (await response.json()) as ChatCompletionResponse;
    const text = extractCompletionText(completion);

    return {
      text,
      provider: this.#providerName,
      model: this.#model,
      latencyMs: Date.now() - startedAt,
    };
  }
}

function extractCompletionText(
  completion: ChatCompletionResponse,
): string {
  const choice = completion.choices?.[0];
  const content = choice?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => (part.type === "text" || part.text ? part.text ?? "" : ""))
      .join("");
  }
  return "";
}
