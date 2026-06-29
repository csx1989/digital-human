# Web Client

Browser client for the avatar SaaS prototype — Phase 5C.

## Status

**Live at [avatar-web-client.pages.dev](https://avatar-web-client.pages.dev)**

Connects to the deployed Cloudflare Worker, starts an avatar session on Modal (GPU L4),
renders a debug JPEG frame, and displays it in the browser.

## Architecture

```
Browser (this client)
  → POST /sessions → Cloudflare Worker
  → WebSocket upgrade → AvatarSessionDO (Durable Object)
    → Modal AvatarLifecycleL4 (GPU L4, snapshot pipeline)
      → JPEG frame → base64 → back to browser
```

## Local Development

```bash
npm install
npm run dev       # esbuild --watch
npm run preview   # wrangler pages dev on :8788
```

Open http://localhost:8788 in a browser.

## Build & Deploy

```bash
npm run build     # esbuild + copy HTML/CSS to dist/
npm run deploy    # build + wrangler pages deploy
```

Manual deploy:
```bash
npx wrangler pages deploy dist --project-name=avatar-web-client
```

## Scripts

| Script | Description |
|--------|-------------|
| `dev` | esbuild watch mode (IIFE bundle) |
| `build` | Production build to `dist/` |
| `typecheck` | `tsc --noEmit` |
| `preview` | `wrangler pages dev dist` on port 8788 |
| `deploy` | Build + deploy to Cloudflare Pages |

## Client Flow

1. User clicks **Connect**
2. `POST /sessions` → `{ sessionId, wsUrl }`
3. Open WebSocket to `wsUrl`
4. Send `{ type: "session.start" }`
5. Receive `session.ready` → send `{ type: "render_debug_frame" }`
6. Receive `avatar.frame` → display `data:image/jpeg;base64,...`
7. User clicks **Stop**
8. Send `{ type: "session.end" }`
9. Receive `session.ended` → close WebSocket

## UI Elements

- **Worker URL** input (defaults to deployed Worker)
- **Connect** / **Stop** buttons
- **Status bar**: disconnected → creating → connecting → ready → rendering → stopped / error
- **Avatar frame**: displays the rendered JPEG
- **Log panel**: timestamps + elapsed time for every message
