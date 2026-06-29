import type { Ai } from "@cloudflare/workers-types";
import { signModalRequest } from "./modal-request-signing";

type DeepgramSpeaker =
  | "angus"
  | "asteria"
  | "arcas"
  | "orion"
  | "orpheus"
  | "athena"
  | "luna"
  | "zeus"
  | "perseus"
  | "helios"
  | "hera"
  | "stella";

const DEFAULT_SPEAKER: DeepgramSpeaker = "asteria";
const TTS_TIMEOUT_MS = 30_000;

export interface TtsResult {
  audio: ArrayBuffer;
  mimeType: "audio/wav";
  sampleRate: 16000;
  provider: string;
  latencyMs: number;
  directEnqueued?: boolean;
  directEnqueueDurationMs?: number;
}

export interface OmniVoiceTtsEvent {
  event:
    | "omnivoice_tts_requested"
    | "omnivoice_tts_completed"
    | "omnivoice_tts_failed"
    | "tts_fallback_to_sarvam"
    | "same_gpu_parallel_omnivoice_tts_requested"
    | "same_gpu_parallel_omnivoice_tts_completed"
    | "same_gpu_parallel_omnivoice_tts_failed";
  properties: Record<string, unknown>;
}

export class BridgeTtsProvider {
  #ai: Ai;
  #sarvamApiKey?: string;
  #speaker: DeepgramSpeaker;
  #languageCode: string;
  #omnivoiceTtsUrl?: string;
  #enableOmniVoiceTts: boolean;
  #enableOmniVoiceSameGpuParallel: boolean;
  #parallelLifecycleUrl?: string;
  #sameGpuParallelTimeoutMs: number;
  #gpuTarget?: string;
  #sessionId?: string;
  #modalRequestSigningSecret?: string;
  #omnivoiceEvents: OmniVoiceTtsEvent[] = [];

  constructor(options: {
    ai: Ai;
    sarvamApiKey?: string;
    speaker?: DeepgramSpeaker;
    languageCode?: string;
    omnivoiceTtsUrl?: string;
    enableOmniVoiceTts?: boolean;
    enableOmniVoiceSameGpuParallel?: boolean;
    parallelLifecycleUrl?: string;
    sameGpuParallelTimeoutMs?: number;
    gpuTarget?: string;
    sessionId?: string;
    modalRequestSigningSecret?: string;
  }) {
    this.#ai = options.ai;
    this.#sarvamApiKey = options.sarvamApiKey;
    this.#speaker = options.speaker ?? DEFAULT_SPEAKER;
    this.#languageCode = options.languageCode ?? "en-IN";
    this.#omnivoiceTtsUrl = options.omnivoiceTtsUrl;
    this.#enableOmniVoiceTts = options.enableOmniVoiceTts ?? false;
    this.#enableOmniVoiceSameGpuParallel = options.enableOmniVoiceSameGpuParallel ?? false;
    this.#parallelLifecycleUrl = options.parallelLifecycleUrl;
    this.#sameGpuParallelTimeoutMs = options.sameGpuParallelTimeoutMs ?? 12_000;
    this.#gpuTarget = options.gpuTarget;
    this.#sessionId = options.sessionId;
    this.#modalRequestSigningSecret = options.modalRequestSigningSecret;
  }

  async #signedHeaders(url: string, bodyJson: string): Promise<Record<string, string>> {
    const result = await signModalRequest(this.#modalRequestSigningSecret, "POST", url, bodyJson);
    return result.headers;
  }

  get omnivoiceEvents(): OmniVoiceTtsEvent[] {
    return this.#omnivoiceEvents;
  }

  get providerChain(): string {
    if (this.#enableOmniVoiceSameGpuParallel && this.#parallelLifecycleUrl)
      return "omnivoice_same_gpu_parallel_sarvam_melotts";
    if (this.#enableOmniVoiceTts && this.#omnivoiceTtsUrl)
      return "omnivoice_sarvam_melotts";
    return "sarvam_melotts";
  }

  async synthesize(text: string, options?: { turnId?: number; fmp4AvMode?: boolean }): Promise<TtsResult | null> {
    const normalized = normalizeTTSInput(text);
    if (!isSpeakableText(normalized)) {
      console.log("[TTS] Skipped non-speakable text");
      return null;
    }

    const chain = this.providerChain;
    const turnId = options?.turnId;
    const fmp4AvMode = options?.fmp4AvMode ?? false;
    console.log(`[TTS] synthesize() textLength=${normalized.length} providerChain=${chain} languageCode=${this.#languageCode} turnId=${turnId} fmp4AvMode=${fmp4AvMode}`);

    if (this.#enableOmniVoiceSameGpuParallel && this.#parallelLifecycleUrl) {
      this.#emitOmniVoice("same_gpu_parallel_omnivoice_tts_requested", {
        textLength: normalized.length,
        targetUrl: this.#parallelLifecycleUrl.substring(0, 80),
        sessionId: this.#sessionId,
        gpuTarget: this.#gpuTarget,
        timeoutMs: this.#sameGpuParallelTimeoutMs,
        turnId,
        fmp4AvMode,
      });

      const parallelResult = await this.#synthesizeOmniVoiceSameGpuParallel(normalized, turnId, fmp4AvMode);
      if (parallelResult) return parallelResult;

      this.#emitOmniVoice("tts_fallback_to_sarvam", {
        textLength: normalized.length,
        reason: "same_gpu_parallel_failed_or_timeout",
      });
    }

    if (this.#enableOmniVoiceTts && this.#omnivoiceTtsUrl) {
      this.#emitOmniVoice("omnivoice_tts_requested", {
        textLength: normalized.length,
        targetUrl: this.#omnivoiceTtsUrl.substring(0, 80),
      });

      const omniResult = await this.#synthesizeOmniVoice(normalized);
      if (omniResult) return omniResult;

      this.#emitOmniVoice("tts_fallback_to_sarvam", {
        textLength: normalized.length,
        reason: "omnivoice_failed_or_timeout",
      });
    }

    const sarvamResult = await this.#synthesizeSarvam(normalized);
    if (sarvamResult) return sarvamResult;

    console.log("[TTS] Sarvam failed, trying MeloTTS fallback...");
    const meloResult = await this.#synthesizeMelotts(normalized);
    if (meloResult) return meloResult;

    console.error("[TTS] All TTS providers failed");
    return null;
  }

  #emitOmniVoice(event: OmniVoiceTtsEvent["event"], properties: Record<string, unknown>): void {
    this.#omnivoiceEvents.push({ event, properties });
  }

  async #synthesizeOmniVoice(text: string): Promise<TtsResult | null> {
    const url = this.#omnivoiceTtsUrl;
    if (!url) return null;

    const OMNIVOICE_TIMEOUT_MS = 10_000;
    const startedAt = Date.now();

    try {
      console.log("[TTS] OmniVoice: calling Modal endpoint...");
      const bodyJson = JSON.stringify({ text, target_sample_rate: 16000 });
      const signed = await this.#signedHeaders(url, bodyJson);
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...signed },
        body: bodyJson,
        signal: AbortSignal.timeout(OMNIVOICE_TIMEOUT_MS),
      });

      const latencyMs = Date.now() - startedAt;

      if (!response.ok) {
        const errBody = await response.text().catch(() => "");
        const snippet = errBody.substring(0, 300);
        console.error(`[TTS] OmniVoice HTTP ${response.status}: ${snippet}`);
        this.#emitOmniVoice("omnivoice_tts_failed", {
          error: `http_${response.status}`,
          status: response.status,
          bodySnippet: snippet,
          latencyMs,
          textLength: text.length,
          timeout: false,
        });
        return null;
      }

      const body = (await response.json()) as {
        ok: boolean;
        wav_base64?: string;
        sample_rate?: number;
        duration_ms?: number;
        latency_ms?: number;
        rtf?: number;
        output_bytes?: number;
        provider?: string;
        text_length?: number;
        error?: string;
      };

      if (!body.ok || !body.wav_base64) {
        console.error(`[TTS] OmniVoice returned not ok: ${body.error}`);
        this.#emitOmniVoice("omnivoice_tts_failed", {
          error: body.error ?? "response_not_ok",
          latencyMs,
          textLength: text.length,
          timeout: false,
        });
        return null;
      }

      const wavBytes = Uint8Array.from(atob(body.wav_base64), (c) => c.charCodeAt(0));
      const pcm = decodeWavToPcm(wavBytes);
      const wav = encodePcm16ToWav(pcm, 16000);

      this.#emitOmniVoice("omnivoice_tts_completed", {
        latencyMs,
        audioDurationMs: body.duration_ms,
        rtf: body.rtf,
        outputBytes: body.output_bytes,
        sampleRate: body.sample_rate,
        textLength: text.length,
        modalInferenceMs: body.latency_ms,
      });

      console.log(
        `[TTS] OmniVoice: ${wav.byteLength} bytes WAV, modal=${body.latency_ms}ms, ` +
        `e2e=${latencyMs}ms, duration=${body.duration_ms}ms, RTF=${body.rtf}`,
      );

      return {
        audio: wav,
        mimeType: "audio/wav",
        sampleRate: 16000,
        provider: "omnivoice",
        latencyMs,
      };
    } catch (err) {
      const latencyMs = Date.now() - startedAt;
      const isTimeout = err instanceof DOMException && err.name === "AbortError";
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[TTS] OmniVoice error (${isTimeout ? "timeout" : "error"}): ${msg}`);

      this.#emitOmniVoice("omnivoice_tts_failed", {
        error: isTimeout ? `timeout_after_${OMNIVOICE_TIMEOUT_MS}ms` : msg.substring(0, 300),
        latencyMs,
        textLength: text.length,
        timeout: isTimeout,
      });
      return null;
    }
  }

  async #synthesizeOmniVoiceSameGpuParallel(text: string, turnId: number | undefined, fmp4AvMode: boolean): Promise<TtsResult | null> {
    const url = this.#parallelLifecycleUrl;
    if (!url) return null;

    const timeoutMs = this.#sameGpuParallelTimeoutMs;
    const startedAt = Date.now();
    const shouldEnqueue = fmp4AvMode && turnId !== undefined;

    try {
      console.log(`[TTS] Same-GPU Parallel OmniVoice (${this.#gpuTarget ?? "unknown"}): calling parallel lifecycle...${shouldEnqueue ? " (direct enqueue)" : ""}`);
      const reqBody: Record<string, unknown> = {
        action: "tts_same_gpu_parallel",
        text,
        session_id: this.#sessionId,
        target_sample_rate: 16000,
      };
      if (shouldEnqueue) {
        reqBody.enqueueToTimeline = true;
        reqBody.turnId = turnId;
      }
      const reqBodyJson = JSON.stringify(reqBody);
      const signed = await this.#signedHeaders(url, reqBodyJson);
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...signed },
        body: reqBodyJson,
        signal: AbortSignal.timeout(timeoutMs),
      });

      const latencyMs = Date.now() - startedAt;

      if (!response.ok) {
        const errBody = await response.text().catch(() => "");
        const snippet = errBody.substring(0, 300);
        console.error(`[TTS] Same-GPU Parallel OmniVoice HTTP ${response.status}: ${snippet}`);
        this.#emitOmniVoice("same_gpu_parallel_omnivoice_tts_failed", {
          error: `http_${response.status}`,
          status: response.status,
          bodySnippet: snippet,
          latencyMs,
          textLength: text.length,
          timeout: false,
        });
        return null;
      }

      const body = (await response.json()) as {
        ok: boolean;
        wav_base64?: string;
        enqueued?: boolean;
        gpuTarget?: string;
        sample_rate?: number;
        duration_ms?: number;
        validated_duration_ms?: number;
        latency_ms?: number;
        rtf?: number;
        output_bytes?: number;
        provider?: string;
        text_length?: number;
        error?: string;
        container_instance_id?: string;
        peak_vram_gb?: number;
        overlap_ms?: number;
      };

      if (!body.ok) {
        console.error(`[TTS] Same-GPU Parallel OmniVoice returned not ok: ${body.error}`);
        this.#emitOmniVoice("same_gpu_parallel_omnivoice_tts_failed", {
          error: body.error ?? "response_not_ok",
          latencyMs,
          textLength: text.length,
          timeout: false,
          containerInstanceId: body.container_instance_id,
        });
        return null;
      }

      if (body.enqueued) {
        this.#emitOmniVoice("same_gpu_parallel_omnivoice_tts_completed", {
          latencyMs,
          audioDurationMs: body.validated_duration_ms ?? body.duration_ms,
          validatedDurationMs: body.validated_duration_ms,
          rtf: body.rtf,
          outputBytes: body.output_bytes,
          sampleRate: body.sample_rate,
          textLength: text.length,
          modalInferenceMs: body.latency_ms,
          containerInstanceId: body.container_instance_id,
          peakVramGb: body.peak_vram_gb,
          overlapMs: body.overlap_ms,
          gpuTarget: body.gpuTarget ?? this.#gpuTarget,
          directEnqueued: true,
          turnId,
        });

        console.log(
          `[TTS] Same-GPU Parallel OmniVoice (${this.#gpuTarget ?? "unknown"}): DIRECT ENQUEUED, ` +
          `modal=${body.latency_ms}ms, e2e=${latencyMs}ms, ` +
          `duration=${body.validated_duration_ms ?? body.duration_ms}ms, RTF=${body.rtf}, ` +
          `container=${body.container_instance_id}`,
        );

        return {
          audio: new ArrayBuffer(0),
          mimeType: "audio/wav",
          sampleRate: 16000,
          provider: "omnivoice_same_gpu_parallel",
          latencyMs,
          directEnqueued: true,
          directEnqueueDurationMs: body.validated_duration_ms ?? body.duration_ms,
        };
      }

      if (!body.wav_base64) {
        console.error(`[TTS] Same-GPU Parallel OmniVoice: no wav_base64 and not enqueued`);
        this.#emitOmniVoice("same_gpu_parallel_omnivoice_tts_failed", {
          error: "no_wav_base64_and_not_enqueued",
          latencyMs,
          textLength: text.length,
        });
        return null;
      }

      const wavBytes = Uint8Array.from(atob(body.wav_base64), (c) => c.charCodeAt(0));
      const pcm = decodeWavToPcm(wavBytes);
      const wav = encodePcm16ToWav(pcm, 16000);

      this.#emitOmniVoice("same_gpu_parallel_omnivoice_tts_completed", {
        latencyMs,
        audioDurationMs: body.duration_ms,
        validatedDurationMs: body.validated_duration_ms,
        rtf: body.rtf,
        outputBytes: body.output_bytes,
        sampleRate: body.sample_rate,
        textLength: text.length,
        modalInferenceMs: body.latency_ms,
        containerInstanceId: body.container_instance_id,
        peakVramGb: body.peak_vram_gb,
        overlapMs: body.overlap_ms,
        gpuTarget: this.#gpuTarget,
      });

      console.log(
        `[TTS] Same-GPU Parallel OmniVoice (${this.#gpuTarget ?? "unknown"}): ${wav.byteLength}B WAV, ` +
        `modal=${body.latency_ms}ms, e2e=${latencyMs}ms, ` +
        `duration=${body.duration_ms}ms, RTF=${body.rtf}, ` +
        `container=${body.container_instance_id}`,
      );

      return {
        audio: wav,
        mimeType: "audio/wav",
        sampleRate: 16000,
        provider: "omnivoice_same_gpu_parallel",
        latencyMs,
      };
    } catch (err) {
      const latencyMs = Date.now() - startedAt;
      const isTimeout = err instanceof DOMException && err.name === "AbortError";
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[TTS] Same-GPU Parallel OmniVoice error (${isTimeout ? "timeout" : "error"}): ${msg}`);

      this.#emitOmniVoice("same_gpu_parallel_omnivoice_tts_failed", {
        error: isTimeout ? `timeout_after_${timeoutMs}ms` : msg.substring(0, 300),
        latencyMs,
        textLength: text.length,
        timeout: isTimeout,
        gpuTarget: this.#gpuTarget,
      });
      return null;
    }
  }

  async #synthesizeSarvam(text: string): Promise<TtsResult | null> {
    const apiKey = this.#sarvamApiKey;
    if (!apiKey) {
      console.error("[TTS] Sarvam: No API key configured");
      return null;
    }

    const startedAt = Date.now();
    try {
      console.log("[TTS] Sarvam: calling API...");
      const response = await fetch("https://api.sarvam.ai/text-to-speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "api-subscription-key": apiKey,
        },
        body: JSON.stringify({
          text,
          target_language_code: this.#languageCode,
          model: "bulbul:v2",
          speaker: mapSpeakerToSarvam(this.#speaker),
          speech_sample_rate: 16000,
        }),
        signal: AbortSignal.timeout(TTS_TIMEOUT_MS),
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        console.error(`[TTS] Sarvam API error: ${response.status} - ${errText.substring(0, 500)}`);
        return null;
      }

      const result = (await response.json()) as {
        request_id?: string;
        audios?: string[];
      };
      console.log(`[TTS] Sarvam: request_id=${result.request_id}`);

      if (!result.audios || result.audios.length === 0) {
        console.error("[TTS] Sarvam: no audio returned");
        return null;
      }

      const base64 = result.audios[0];
      const wavBytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
      console.log(`[TTS] Sarvam: ${wavBytes.length} bytes of WAV data`);

      const pcm = decodeWavToPcm(wavBytes);
      const wav = encodePcm16ToWav(pcm, 16000);
      const latencyMs = Date.now() - startedAt;

      console.log(`[TTS] Sarvam: ${pcm.byteLength} bytes PCM → ${wav.byteLength} bytes WAV (${latencyMs}ms)`);

      return {
        audio: wav,
        mimeType: "audio/wav",
        sampleRate: 16000,
        provider: "sarvam",
        latencyMs,
      };
    } catch (err) {
      console.error("[TTS] Sarvam error:", err instanceof Error ? err.message : String(err));
      return null;
    }
  }

  async #synthesizeMelotts(text: string): Promise<TtsResult | null> {
    const startedAt = Date.now();
    try {
      console.log("[TTS] MeloTTS: synthesizing...");
      const result = (await this.#ai.run("@cf/myshell-ai/melotts", {
        prompt: text,
        lang: "en",
      })) as { audio?: string } | string;

      if (!result) {
        console.error("[TTS] MeloTTS: null response");
        return null;
      }

      let base64: string;
      if (typeof result === "string") {
        base64 = result;
      } else if (typeof result === "object" && "audio" in result) {
        base64 = (result as { audio: string }).audio;
      } else {
        console.error("[TTS] MeloTTS: unexpected response shape");
        return null;
      }

      const mp3Bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
      console.log(`[TTS] MeloTTS: ${mp3Bytes.length} bytes of MP3 data`);

      const decoded = await decodeMp3(mp3Bytes);
      if (!decoded) {
        console.error("[TTS] MeloTTS: MP3 decode FAILED");
        return null;
      }

      const pcm16 = resampleTo16000(decoded.samples, decoded.sampleRate);
      console.log(`[TTS] MeloTTS: decoded ${decoded.sampleRate}Hz → resampled to 16kHz, ${pcm16.byteLength} bytes PCM`);

      const wav = encodePcm16ToWav(new Int16Array(pcm16), 16000);
      const latencyMs = Date.now() - startedAt;

      return {
        audio: wav,
        mimeType: "audio/wav",
        sampleRate: 16000,
        provider: "melotts",
        latencyMs,
      };
    } catch (err) {
      const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      console.error(`[TTS] MeloTTS error: ${msg}`);
      if (msg.includes("neurons") || msg.includes("429") || msg.includes("paid")) {
        console.error("[TTS] Workers AI Neurons exhausted.");
      }
      return null;
    }
  }
}

function mapSpeakerToSarvam(speaker: DeepgramSpeaker): string {
  const mapping: Record<DeepgramSpeaker, string> = {
    asteria: "anushka",
    athena: "manisha",
    luna: "vidya",
    stella: "arya",
    hera: "manisha",
    angus: "abhilash",
    arcas: "karun",
    orion: "hitesh",
    orpheus: "abhilash",
    perseus: "karun",
    helios: "hitesh",
    zeus: "abhilash",
  };
  return mapping[speaker] ?? "anushka";
}

function decodeWavToPcm(wavBytes: Uint8Array): ArrayBuffer {
  if (
    wavBytes.length < 44 ||
    String.fromCharCode(wavBytes[0], wavBytes[1], wavBytes[2], wavBytes[3]) !== "RIFF"
  ) {
    console.error("[TTS] WAV: not a valid WAV file, returning raw bytes");
    const raw = new ArrayBuffer(wavBytes.byteLength);
    new Uint8Array(raw).set(wavBytes);
    return raw;
  }

  const view = new DataView(wavBytes.buffer, wavBytes.byteOffset, wavBytes.byteLength);
  let offset = 12;
  let channels = 0;
  let bitsPerSample = 0;
  let sampleRate = 0;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= wavBytes.length) {
    const chunkId = String.fromCharCode(
      wavBytes[offset],
      wavBytes[offset + 1],
      wavBytes[offset + 2],
      wavBytes[offset + 3],
    );
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;

    if (chunkId === "fmt " && chunkDataOffset + 16 <= wavBytes.length) {
      channels = view.getUint16(chunkDataOffset + 2, true);
      sampleRate = view.getUint32(chunkDataOffset + 4, true);
      bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
    } else if (chunkId === "data") {
      dataOffset = chunkDataOffset;
      dataSize = Math.min(chunkSize, wavBytes.length - chunkDataOffset);
      break;
    }

    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  if (channels === 0 || bitsPerSample === 0 || dataOffset < 0 || dataSize <= 0) {
    console.error("[TTS] WAV: invalid header, returning raw bytes");
    const raw = new ArrayBuffer(wavBytes.byteLength);
    new Uint8Array(raw).set(wavBytes);
    return raw;
  }

  console.log(
    `[TTS] WAV: ${channels}ch, ${bitsPerSample}bps, ${sampleRate}Hz, ${wavBytes.length} total bytes, dataOffset=${dataOffset}, dataSize=${dataSize}`,
  );

  if (channels === 1 && bitsPerSample === 16 && sampleRate === 16000) {
    const pcm = wavBytes.slice(dataOffset, dataOffset + dataSize);
    const result = new ArrayBuffer(pcm.byteLength);
    new Uint8Array(result).set(pcm);
    return result;
  }

  if (channels === 1 && bitsPerSample === 8 && sampleRate === 16000) {
    const pcm8 = wavBytes.slice(dataOffset);
    const pcm16 = new Int16Array(pcm8.length);
    for (let i = 0; i < pcm8.length; i++) {
      pcm16[i] = (pcm8[i] - 128) * 256;
    }
    console.log(`[TTS] WAV: ${pcm16.byteLength} bytes of 16-bit PCM (converted from 8-bit)`);
    return pcm16.buffer;
  }

  const floatSamples = extractWavToFloat32Mono({
    wavBytes,
    view,
    channels,
    bitsPerSample,
    sampleRate,
    dataOffset,
    dataSize,
  });
  if (!floatSamples) {
    const pcm = wavBytes.slice(dataOffset, dataOffset + dataSize);
    const result = new ArrayBuffer(pcm.byteLength);
    new Uint8Array(result).set(pcm);
    return result;
  }
  const normalized =
    sampleRate === 16000
      ? convertFloat32ToInt16(floatSamples)
      : resampleTo16000(floatSamples, sampleRate);
  console.log(
    `[TTS] WAV: normalized to ${new Int16Array(normalized).length} mono samples at 16000Hz`,
  );
  return normalized;
}

function extractWavToFloat32Mono(input: {
  wavBytes: Uint8Array;
  view: DataView;
  channels: number;
  bitsPerSample: number;
  sampleRate: number;
  dataOffset: number;
  dataSize: number;
}): Float32Array | null {
  const { wavBytes, view, channels, bitsPerSample, dataOffset, dataSize } = input;
  const bytesPerSample = bitsPerSample / 8;
  if (!(bitsPerSample === 8 || bitsPerSample === 16)) {
    return null;
  }
  const samplesPerChannel = Math.floor(dataSize / (channels * bytesPerSample));
  const mono = new Float32Array(samplesPerChannel);
  for (let i = 0; i < samplesPerChannel; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const sampleOffset = dataOffset + (i * channels + c) * bytesPerSample;
      if (bitsPerSample === 16) {
        sum += view.getInt16(sampleOffset, true) / 32768;
      } else {
        sum += (wavBytes[sampleOffset] - 128) / 128;
      }
    }
    mono[i] = sum / channels;
  }
  return mono;
}

function resampleTo16000(float32Samples: Float32Array, sourceRate: number): ArrayBuffer {
  const TARGET_RATE = 16000;
  const sourceLength = float32Samples.length;
  const targetLength = Math.floor((sourceLength * TARGET_RATE) / sourceRate);
  const targetSamples = new Float32Array(targetLength);

  for (let i = 0; i < targetLength; i++) {
    const sourceIndex = (i * sourceRate) / TARGET_RATE;
    const indexFloor = Math.floor(sourceIndex);
    const indexCeil = Math.ceil(sourceIndex);
    const frac = sourceIndex - indexFloor;

    if (indexCeil >= sourceLength) {
      targetSamples[i] = float32Samples[sourceLength - 1];
    } else {
      const s0 = float32Samples[indexFloor];
      const s1 = float32Samples[indexCeil];
      targetSamples[i] = s0 + (s1 - s0) * frac;
    }
  }

  return convertFloat32ToInt16(targetSamples);
}

function convertFloat32ToInt16(samples: Float32Array): ArrayBuffer {
  const int16 = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    let sample = samples[i];
    if (sample > 1.0) sample = 1.0;
    if (sample < -1.0) sample = -1.0;
    int16[i] = Math.floor(sample * 32767);
  }
  return int16.buffer;
}

async function decodeMp3(mp3Data: Uint8Array): Promise<{ samples: Float32Array; sampleRate: number } | null> {
  try {
    const { MPEGDecoder } = await import("mpg123-decoder");
    const decoder = new MPEGDecoder();
    await decoder.ready;

    const { channelData, sampleRate } = decoder.decode(mp3Data);
    console.log(`[TTS] MP3 decoded: sampleRate=${sampleRate}Hz channels=${channelData.length}`);

    if (!channelData || channelData.length === 0) {
      console.error("[TTS] MP3: no channel data returned");
      return null;
    }

    const mono =
      channelData.length > 1 ? downmixToMono(channelData) : channelData[0];
    return { samples: new Float32Array(mono), sampleRate };
  } catch (err) {
    const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    console.error(`[TTS] MP3 decode error: ${msg}`);
    return null;
  }
}

function downmixToMono(channels: Float32Array[]): Float32Array {
  const monoLength = channels[0].length;
  const mono = new Float32Array(monoLength);
  for (let i = 0; i < monoLength; i++) {
    let sum = 0;
    for (let c = 0; c < channels.length; c++) {
      sum += channels[c][i];
    }
    mono[i] = sum / channels.length;
  }
  return mono;
}

export function encodePcm16ToWav(pcm: ArrayBuffer | Int16Array, sampleRate: number): ArrayBuffer {
  const pcmBuffer = pcm instanceof Int16Array ? pcm : new Int16Array(pcm);
  const numSamples = pcmBuffer.length;
  const dataSize = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  const pcmBytes = new Uint8Array(pcmBuffer.buffer, pcmBuffer.byteOffset, pcmBuffer.byteLength);
  new Uint8Array(buffer, 44).set(pcmBytes);

  return buffer;
}

function normalizeTTSInput(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function isSpeakableText(text: string): boolean {
  if (!text) return false;
  return /[\p{L}\p{N}]/u.test(text);
}
