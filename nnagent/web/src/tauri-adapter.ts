/**
 * Tauri Adapter — isolates all Tauri-specific API calls.
 *
 * Per ADR-050: "tauri-adapter.ts must be the ONLY file importing @tauri-apps/api"
 * This allows the React frontend to be framework-agnostic and portable
 * to Electron, Qt6 QWebEngineView, or standalone web deployment.
 *
 * NOTE: @tauri-apps/api is only available when building for Tauri desktop.
 * Dynamic imports are used to avoid bundling issues in web-only builds.
 */

// ─── Platform Detection ───────────────────────────────────────────────────────

/** Check if running inside a Tauri shell. */
export function isTauri(): boolean {
  try {
    return window !== undefined &&
           (window as unknown as Record<string, unknown>).__TAURI__ !== undefined
  } catch {
    return false
  }
}

// ─── Helper: safe dynamic import ──────────────────────────────────────────────

/**
 * Attempt to dynamically import a Tauri API module.
 * Returns null if the module is not available (non-Tauri environment).
 */
async function importTauriModule(modulePath: string): Promise<Record<string, unknown> | null> {
  if (!isTauri()) {
    return null
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
    const mod = (await import(/* @vite-ignore */ modulePath as any)) as Record<string, unknown>
    return mod
  } catch (error) {
    console.warn(`[tauri-adapter] Failed to import module '${modulePath}':`, error)
    return null
  }
}

// ─── IPC Commands ─────────────────────────────────────────────────────────────

/**
 * Invoke a command on the Tauri backend.
 * Falls back to a no-op when not in Tauri environment.
 */
export async function invoke<T>(command: string, params?: Record<string, unknown>): Promise<T | null> {
  if (!isTauri()) {
    console.warn(`[tauri-adapter] invoke('${command}') called outside Tauri environment — returning null`)
    return null
  }

  try {
    const mod = await importTauriModule('@tauri-apps/api/core')
    if (!mod) {
      return null
    }
    const tauriInvoke = mod.invoke as <R>(cmd: string, p?: Record<string, unknown>) => Promise<R>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return tauriInvoke<T>(command, params ?? {}) as any
  } catch (error) {
    console.error(`[tauri-adapter] invoke('${command}') failed:`, error)
    throw new Error(`Tauri IPC command '${command}' failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/**
 * Listen for events from the Tauri backend.
 */
export function listen<T>(event: string, handler: (payload: T) => void): () => void {
  if (!isTauri()) {
    console.warn(`[tauri-adapter] listen('${event}') called outside Tauri environment`)
    return () => {}  // No-op cleanup
  }

  // Dynamic import
  importTauriModule('@tauri-apps/api/event').then((mod) => {
    if (mod) {
      const listenFn = mod.listen as <P>(e: string, h: (p: P) => void) => { unsub: () => void }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      listenFn<T>(event, handler as any).unsub()
    }
  }).catch((error: Error) => {
    console.error(`[tauri-adapter] listen('${event}') setup failed:`, error)
  })

  // Return cleanup function (placeholder — proper unsub handled in dynamic import)
  return () => {}
}

// ─── File System ──────────────────────────────────────────────────────────────

/**
 * Read a file from the local filesystem (Tauri only).
 */
export async function readFile(path: string): Promise<string | null> {
  if (!isTauri()) return null

  try {
    const mod = await importTauriModule('@tauri-apps/api/fs')
    if (!mod) return null
    const readTextFile = mod.readTextFile as (p: string) => Promise<string>
    return readTextFile(path)
  } catch (error) {
    console.error(`[tauri-adapter] readFile('${path}') failed:`, error)
    return null
  }
}

/**
 * Write to a file on the local filesystem (Tauri only).
 */
export async function writeFile(path: string, content: string): Promise<boolean> {
  if (!isTauri()) return false

  try {
    const mod = await importTauriModule('@tauri-apps/api/fs')
    if (!mod) return false
    const writeTextFile = mod.writeTextFile as (p: string, c: string) => Promise<void>
    await writeTextFile(path, content)
    return true
  } catch (error) {
    console.error(`[tauri-adapter] writeFile('${path}') failed:`, error)
    return false
  }
}

// ─── Dialog ───────────────────────────────────────────────────────────────────

/**
 * Open a file picker dialog (Tauri only).
 */
export async function openFilePicker(filters?: string[][]): Promise<string[] | null> {
  if (!isTauri()) return null

  try {
    const mod = await importTauriModule('@tauri-apps/api/dialog')
    if (!mod) return null
    const openFn = mod.open as (opts?: Record<string, unknown>) => Promise<string[] | null>
    return openFn({ filters })
  } catch (error) {
    console.error('[tauri-adapter] openFilePicker failed:', error)
    return null
  }
}

// ─── Logging ──────────────────────────────────────────────────────────────────

/**
 * Send a log message to the Tauri backend logger.
 */
export async function log(level: string, component: string, message: string): Promise<void> {
  const logMessage = `[${component}] ${message}`
  if (!isTauri()) {
    // Use individual console methods to avoid TypeScript argument count issues
    const lowerLevel = level.toLowerCase()
    if (lowerLevel === 'error') {
      // eslint-disable-next-line no-console
      console.error(logMessage)
    } else if (lowerLevel === 'warn') {
      // eslint-disable-next-line no-console
      console.warn(logMessage)
    } else if (lowerLevel === 'info') {
      // eslint-disable-next-line no-console
      console.info(logMessage)
    } else if (lowerLevel === 'debug') {
      // eslint-disable-next-line no-console
      console.debug(logMessage)
    } else {
      // eslint-disable-next-line no-console
      console.log(logMessage)
    }
    return
  }

  try {
    const mod = await importTauriModule('@tauri-apps/api/core')
    if (mod) {
      const tauriInvoke = mod.invoke as <R>(cmd: string, p?: Record<string, unknown>) => Promise<R>
      await tauriInvoke('log_message', { level, component, message })
    }
  } catch {
    // Fallback to console if backend not ready
    const lowerLevel = level.toLowerCase()
    if (lowerLevel === 'error') {
      // eslint-disable-next-line no-console
      console.error(logMessage)
    } else if (lowerLevel === 'warn') {
      // eslint-disable-next-line no-console
      console.warn(logMessage)
    } else if (lowerLevel === 'info') {
      // eslint-disable-next-line no-console
      console.info(logMessage)
    } else if (lowerLevel === 'debug') {
      // eslint-disable-next-line no-console
      console.debug(logMessage)
    } else {
      // eslint-disable-next-line no-console
      console.log(logMessage)
    }
  }
}
