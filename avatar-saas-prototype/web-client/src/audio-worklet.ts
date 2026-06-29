declare function registerProcessor(
  name: string,
  processorCtor: new () => AudioWorkletProcessor,
): void;

declare const sampleRate: number;

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 1600;
const MAX_16BIT = 32767;

class MicCaptureProcessor extends AudioWorkletProcessor {
  #sourceRate: number | null = null;
  #ratio = 1;
  #fractionalPos = 0;
  #outputBuffer: number[] = [];
  #initialized = false;

  process(inputs: Float32Array[][]): boolean {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    if (!this.#initialized) {
      this.#sourceRate = sampleRate;
      this.#ratio = this.#sourceRate / TARGET_RATE;
      this.#initialized = true;
    }

    if (this.#ratio === 1) {
      for (let i = 0; i < channel.length; i++) {
        this.#outputBuffer.push(channel[i]);
      }
    } else {
      const resampled = this.#resample(channel);
      for (let i = 0; i < resampled.length; i++) {
        this.#outputBuffer.push(resampled[i]);
      }
    }

    while (this.#outputBuffer.length >= CHUNK_SAMPLES) {
      const chunk = this.#outputBuffer.splice(0, CHUNK_SAMPLES);
      const rms = this.#computeRms(chunk);
      const peak = this.#computePeak(chunk);
      const nearZeroRatio = this.#computeNearZeroRatio(chunk);
      const clippedRatio = this.#computeClippedRatio(chunk);
      const pcm16 = this.#f32ToI16(chunk);
      this.port.postMessage({ pcm16, rms, peak, nearZeroRatio, clippedRatio }, [pcm16]);
    }

    return true;
  }

  #resample(input: Float32Array): number[] {
    const out: number[] = [];
    let pos = this.#fractionalPos;

    while (Math.floor(pos) + 1 < input.length) {
      const idx = Math.floor(pos);
      const frac = pos - idx;
      out.push(input[idx] + (input[idx + 1] - input[idx]) * frac);
      pos += this.#ratio;
    }

    this.#fractionalPos = pos - input.length;
    return out;
  }

  #f32ToI16(samples: number[]): ArrayBuffer {
    const i16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      i16[i] = clamped < 0 ? clamped * 0x8000 : clamped * MAX_16BIT;
    }
    return i16.buffer;
  }

  #computeRms(samples: number[]): number {
    if (samples.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  #computePeak(samples: number[]): number {
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const abs = Math.abs(samples[i]);
      if (abs > peak) peak = abs;
    }
    return peak;
  }

  #computeNearZeroRatio(samples: number[]): number {
    let count = 0;
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) < 0.001) count++;
    }
    return count / samples.length;
  }

  #computeClippedRatio(samples: number[]): number {
    let count = 0;
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) >= 0.99) count++;
    }
    return count / samples.length;
  }
}

registerProcessor("mic-capture-processor", MicCaptureProcessor);
