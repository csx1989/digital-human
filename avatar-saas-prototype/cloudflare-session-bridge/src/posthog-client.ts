import { PostHog } from "posthog-node";

type PostHogEnv = {
  POSTHOG_API_KEY?: string;
  POSTHOG_HOST?: string;
};

type WaitUntilFn = (promise: Promise<unknown>) => void;

const DEFAULT_HOST = "https://eu.i.posthog.com";
let warnedMissingKey = false;

function createClient(env: PostHogEnv): PostHog | null {
  if (!env.POSTHOG_API_KEY) {
    if (!warnedMissingKey) {
      console.warn("[PostHog] POSTHOG_API_KEY not set — all captures will no-op");
      warnedMissingKey = true;
    }
    return null;
  }
  return new PostHog(env.POSTHOG_API_KEY, {
    host: env.POSTHOG_HOST || DEFAULT_HOST,
    flushAt: 1,
    flushInterval: 0,
  });
}

function scheduleCapture(
  env: PostHogEnv,
  waitUntil: WaitUntilFn,
  work: (client: PostHog) => Promise<void>,
  label?: string,
): void {
  const client = createClient(env);
  if (!client) return;

  const host = env.POSTHOG_HOST || DEFAULT_HOST;
  const keyPrefix = env.POSTHOG_API_KEY ? env.POSTHOG_API_KEY.substring(0, 4) : "none";

  waitUntil(
    work(client)
      .then(() => {
        console.log("[PostHog] capture ok", { event: label, host, keyPrefix });
      })
      .catch((err) => {
        console.error("[PostHog] capture error", {
          event: label,
          host,
          keyPrefix,
          error: err instanceof Error ? err.message : String(err),
          errorStack: err instanceof Error ? err.stack : undefined,
          errorType: err instanceof Error ? err.constructor.name : typeof err,
          status: (err as { status?: number })?.status,
          statusCode: (err as { statusCode?: number })?.statusCode,
        });
      })
      .finally(() => {
        return client.shutdown().catch(() => {});
      }),
  );
}

export function captureEvent(
  env: PostHogEnv,
  waitUntil: WaitUntilFn,
  distinctId: string,
  event: string,
  properties?: Record<string, unknown>,
): void {
  scheduleCapture(env, waitUntil, async (client) => {
    await client.captureImmediate({
      distinctId,
      event,
      properties: {
        source: "worker",
        ...properties,
        timestamp: new Date().toISOString(),
      },
    });
  }, event);
}

export function captureError(
  env: PostHogEnv,
  waitUntil: WaitUntilFn,
  distinctId: string,
  error: unknown,
  context?: Record<string, unknown>,
): void {
  scheduleCapture(env, waitUntil, async (client) => {
    await client.captureImmediate({
      distinctId,
      event: "pipeline_error",
      properties: {
        source: "worker",
        error: error instanceof Error ? error.message : String(error),
        errorStack: error instanceof Error ? error.stack : undefined,
        ...context,
        timestamp: new Date().toISOString(),
      },
    });
  }, "pipeline_error");
}
