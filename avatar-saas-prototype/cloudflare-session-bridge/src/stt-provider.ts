const SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text";
const STT_TIMEOUT_MS = 25_000;

export interface TranscriptionResult {
  text: string;
  languageCode?: string;
  durationMs: number;
  provider: string;
}

export interface SttProvider {
  transcribePcm16(
    audio: ArrayBuffer,
    options?: { sampleRate?: number; languageCode?: string },
  ): Promise<TranscriptionResult>;
}

export class SarvamSttProvider implements SttProvider {
  #apiKey: string;
  #model: string;
  #language: string;

  constructor(options: {
    apiKey: string;
    model?: string;
    language?: string;
  }) {
    this.#apiKey = options.apiKey;
    this.#model = options.model ?? "saarika:v2.5";
    this.#language = options.language ?? "unknown";
  }

  async transcribePcm16(
    audio: ArrayBuffer,
    options?: { sampleRate?: number; languageCode?: string },
  ): Promise<TranscriptionResult> {
    const startedAt = Date.now();
    const language = options?.languageCode ?? this.#language;

    const form = new FormData();
    form.append(
      "file",
      new File([audio], "audio.pcm", { type: "application/octet-stream" }),
    );
    form.append("model", this.#model);
    form.append("language_code", language);
    form.append("input_audio_codec", "pcm_s16le");

    const response = await fetch(SARVAM_STT_URL, {
      method: "POST",
      headers: { "api-subscription-key": this.#apiKey },
      body: form,
      signal: AbortSignal.timeout(STT_TIMEOUT_MS),
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      throw new Error(
        `Sarvam STT ${response.status}: ${errText.substring(0, 500)}`,
      );
    }

    const result = (await response.json()) as {
      transcript?: string;
      text?: string;
      request_id?: string;
      language_code?: string | null;
      language_probability?: number | null;
    };

    const text = result.transcript ?? result.text ?? "";

    return {
      text,
      languageCode: result.language_code ?? undefined,
      durationMs: Date.now() - startedAt,
      provider: `sarvam/${this.#model}`,
    };
  }
}
