import { plainFailure, plainFailureText } from './plain-errors';
import { showcaseMessages, SHOWCASE_AGENTS } from './chat-showcase';
import { updateAgent } from './agents';
import { TIME_SKIP_DURATION, timeSkipPrompt, timeSkipTabId, withTimeSkipResult, type TimeSkipRun, type TimeSkipVersion } from './time-skip';
import { PREVIEW_EXAMPLES, type PreviewArtifact } from './artifacts';
import { lifecyclePolicy } from './lifecycle';
import { newestServingState } from './serving-state';
import { photoData, photoHistorySlots, pendingChatPhotos, takeChatPhotos } from './chat-photos';
import { nativeState } from './native';
import { setTranslationLanguage, validLanguage, type Language } from './i18n';
/**
 * Strata app store.
 *
 * A single immutable state object + a tiny subscribe/emit pair, consumed through
 * useSyncExternalStore. Data comes from the engine host over /api; live updates
 * arrive on the event stream. No state library, no dependencies.
 */
import { useSyncExternalStore } from 'react';
import { api, onEvent, streamPost, type StreamEvent } from './client';
import type {
  Workspace,
  ChatMessage,
  ToolCall,
  EngineConfig,
  Hardware,
  ModelLibrary,
  RuntimeModels,
  ImageModelLibrary,
  ImageModelChoice,
  ModelProviders,
  ModelConnectionDraft,
  RemoteModelSelection,
  MonitorState,
  ServingState,
  Session,
  SetupState,
  Skill,
  WorkspaceFile,
} from './types';
import { navigate, type Route } from './router';
import { estimate, execution, contextTokens } from './experience';

export interface Toast {
  id: string;
  tone: 'info' | 'success' | 'warn' | 'error';
  title: string;
  body?: string;
  leaving?: boolean;
}

export interface AppState {
  language: Language;
  fontFamily: string;
  configDraft: string | null;
  navOpen: boolean;
  ready: boolean;
  connection: 'connecting' | 'live' | 'offline';
  theme: 'dark' | 'light' | 'system';
  contrast: 'standard' | 'increased';
  rail: boolean;
  /** Distraction-free reading: chrome recedes, the transcript narrows. */
  focus: boolean;
  /** Conversation prose size in px, 10–22. Does not touch the UI chrome. */
  contentFontSize: number;
  density: 'comfortable' | 'compact';
  motion: 'system' | 'full' | 'reduced';
  /** Translucent materials, or flat surfaces for anyone who prefers them. */
  transparency: 'translucent' | 'opaque';
  /** Chrome scale, independent of the conversation's reading size. */
  uiScale: number;
  /** Governs anonymous usage reporting. Nothing is sent in this build; the
   *  choice is recorded so a version that does offer it starts from it. */
  shareTelemetry: boolean;
  paletteOpen: boolean;
  shortcutsOpen: boolean;
  settingsSection: 'Appearance' | 'Models' | 'Storage' | 'Engine' | 'About';
  hardware: Hardware | null;
  library: ModelLibrary | null;
  imageModels: ImageModelLibrary | null;
  runtime: RuntimeModels | null;
  providers: ModelProviders | null;
  remoteModel: RemoteModelSelection | null;
  engine: EngineConfig | null;
  serving: ServingState | null;
  monitor: MonitorState | null;
  setup: SetupState | null;
  skills: Skill[];
  files: WorkspaceFile[];
  workspacePath: string;
  workspaces: Workspace[];
  /** Id of the workspace the file tree is scoped to. */
  activeWorkspace: string | null;
  /** First run, and "run setup again", both present the wizard as a modal. */
  setupOpen: boolean;
  sessions: Session[];
  activeSession: string | null;
  streaming: boolean;
  /** Chats with a turn in flight. `streaming` is about the open chat; this is about all of them. */
  liveChats: number;
  toasts: Toast[];
  /** Files the model has been granted read access to, and when. */
  touchedPaths: Record<string, string>;
  /** The last action that failed, kept until it is retried or dismissed.
   *  A toast disappears after five seconds; a failure the reader has not dealt
   *  with should not. */
  actionError: { message: string; label: string; retry: () => void } | null;
  busy: Record<string, boolean>;
}

const SESSION_KEY = 'strata.sessions.v1';
const PREFS_KEY = 'strata.prefs.v2';
/** What the model has been allowed to read, kept across restarts. */
const TOUCHED_KEY = 'strata.touched.v1';
/** Kept one release so downgrading does not lose the theme and sidebar state. */
const PREFS_KEY_V1 = 'strata.prefs.v1';

const CONTENT_FS_MIN = 10;
const CONTENT_FS_MAX = 22;

export interface Prefs {
  language: Language;
  fontFamily: string;
  theme: 'dark' | 'light' | 'system';
  contrast: 'standard' | 'increased';
  rail: boolean;
  focus: boolean;
  contentFontSize: number;
  density: 'comfortable' | 'compact';
  motion: 'system' | 'full' | 'reduced';
  transparency: 'translucent' | 'opaque';
  uiScale: number;
  /** Governs anonymous usage reporting. Nothing is sent in this build; the
   *  choice is recorded so a version that does offer it starts from it. */
  shareTelemetry: boolean;
}

const PREFS_DEFAULTS: Prefs = {
  language: 'en',
  fontFamily: '',
  theme: 'system',
  contrast: 'standard',
  rail: false,
  focus: false,
  contentFontSize: 14,
  density: 'comfortable',
  motion: 'system',
  transparency: 'translucent',
  uiScale: 1,
  shareTelemetry: false,
};

function uid(prefix = 'id'): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36).slice(-4)}`;
}

function loadSessions(): Session[] {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Session[];
    return Array.isArray(parsed) ? parsed.map(session => ({
      ...session,
      ...(session.showcase && !session.showcaseAgentStates ? {agents:SHOWCASE_AGENTS,showcaseAgentStates:true}:{}),
      messages: session.messages.map(message => message.streaming ? {
        ...message,
        streaming: false,
        error: message.error ?? 'Response interrupted. Ask again to retry; the partial answer is kept.',
      } : message),
    })) : [];
  } catch {
    return [];
  }
}

/**
 * v2 adds focus, reading size, density and the motion override. A v1 blob (only
 * theme and rail) is read as-is and upgraded on the next write, so the first run
 * after this change keeps the user's theme instead of resetting it.
 */
function loadPrefs(): Prefs {
  for (const key of [PREFS_KEY, PREFS_KEY_V1]) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) continue;
      const parsed = JSON.parse(raw) as Partial<Prefs>;
      return {
        ...PREFS_DEFAULTS,
        ...parsed,
        language: validLanguage(parsed.language) ? parsed.language : 'en',
        fontFamily: typeof parsed.fontFamily === 'string' ? parsed.fontFamily.slice(0,128).replace(/[\x00-\x1f]/g,'') : '',
        contentFontSize: Math.min(CONTENT_FS_MAX, Math.max(CONTENT_FS_MIN, Number(parsed.contentFontSize) || 14)),
      };
    } catch {
      /* fall through to the next key */
    }
  }
  return { ...PREFS_DEFAULTS };
}

const prefs = loadPrefs();
// Use the saved language for the very first render, before host bootstrap.
setTranslationLanguage(prefs.language);

let state: AppState = {
  language: prefs.language,
  fontFamily: prefs.fontFamily,
  configDraft: (() => { try { return sessionStorage.getItem('strata.config-draft.v1'); } catch { return null; } })(),
  navOpen: false,
  ready: false,
  connection: 'connecting',
  shareTelemetry: prefs.shareTelemetry,
  theme: prefs.theme,
  contrast: prefs.contrast,
  rail: prefs.rail,
  focus: prefs.focus,
  contentFontSize: prefs.contentFontSize,
  density: prefs.density,
  motion: prefs.motion,
  transparency: prefs.transparency,
  uiScale: prefs.uiScale,
  paletteOpen: false,
  shortcutsOpen: false,
  settingsSection: 'Appearance',
  hardware: null,
  library: null,
  imageModels: null,
  runtime: null,
  providers: null,
  /** The open chat's API model, or null for the local model (the default for every chat). */
  remoteModel: null,
  engine: null,
  serving: null,
  monitor: null,
  setup: null,
  skills: [],
  files: [],
  workspacePath: '',
  workspaces: [],
  activeWorkspace: null,
  setupOpen: false,
  sessions: loadSessions(),
  activeSession: null,
  streaming: false,
  liveChats: 0,
  toasts: [],
  touchedPaths: (() => {
    try {
      return JSON.parse(localStorage.getItem(TOUCHED_KEY) ?? '{}') as Record<string, string>;
    } catch {
      return {};
    }
  })(),
  actionError: null,
  busy: {},
};

const listeners = new Set<() => void>();

/** Chats with a turn in flight: the abort handle, and whether the host is running it as an agent turn. */
const liveTurns = new Map<string, { abort: AbortController; agent: boolean }>();

function set(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)) {
  const before = state.activeSession;
  const next = typeof patch === 'function' ? patch(state) : patch;
  state = { ...state, ...next };
  // Opening another chat shows that chat's model choice and whether that chat is mid-turn.
  if (state.activeSession !== before) {
    const chosen = state.sessions.find(s => s.id === state.activeSession)?.preferences?.remoteModel ?? null;
    state = { ...state, remoteModel: 'remoteModel' in next ? state.remoteModel : chosen, streaming: !!state.activeSession && liveTurns.has(state.activeSession) };
  }
  for (const l of listeners) l();
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function getState(): AppState {
  return state;
}

export function setConfigDraft(configDraft: string | null) {
  set({ configDraft });
  try {
    if (configDraft === null) sessionStorage.removeItem('strata.config-draft.v1');
    else sessionStorage.setItem('strata.config-draft.v1', configDraft);
  } catch { /* Keep the session draft even when browser storage is unavailable. */ }
}

export function setNavOpen(navOpen: boolean) { set({ navOpen }); }
export function setSettingsSection(settingsSection: AppState['settingsSection']) { set({ settingsSection }); }

/** Choose the open chat's model: an API model, or null for the local one. Other chats keep theirs. */
export function selectRemoteModel(remoteModel: RemoteModelSelection | null) {
  if (state.streaming) return;
  if (remoteModel && !state.providers?.connections.some(c => c.id === remoteModel.connectionId && c.models.some(m => m.id === remoteModel.modelId))) return;
  set({ remoteModel });
  const id = state.activeSession;
  if (id) patchSession(id, s => ({ ...s, preferences: { ...s.preferences, remoteModel } }));
  try { localStorage.removeItem('strata.remote-model.v1'); } catch { /* The old global choice is retired. */ }
}
function applyProviders(providers: ModelProviders) {
  // A chat whose API model was removed keeps its choice: the next send says the model is gone, rather
  // than quietly answering from the local model.
  set({ providers });
}
export async function saveModelConnection(draft: ModelConnectionDraft) {
  applyProviders(await api.post<ModelProviders>('/api/model-providers', draft));
}
export async function removeModelConnection(id: string) {
  applyProviders(await api.del<ModelProviders>(`/api/model-providers/${encodeURIComponent(id)}`));
}

function libraryResponse(data: ModelLibrary): ModelLibrary {
  if (!data || !Array.isArray(data.variants)) throw new Error('The engine returned an incomplete model list. Retry the operation.');
  return data;
}

export function lifecycleStatus(s = state) {
  return lifecyclePolicy({ connected: s.connection === 'live', streaming: !!s.streaming || s.liveChats > 0, stats: s.serving?.stats, busy: s.busy });
}
export function lifecycleAvailable(): boolean {
  const result = lifecycleStatus();
  if (!result.allowed) pushToast('warn', 'Model change unavailable', result.reason);
  return result.allowed;
}

export function useStore<T>(selector: (s: AppState) => T): T {
  return useSyncExternalStore(
    subscribe,
    () => selector(state),
    () => selector(state),
  );
}

/* ------------------------------------------------------------------ toasts */

/** Dismissal timers, held so a hovered toast can be paused rather than lost. */
const toastTimers = new Map<string, ReturnType<typeof setTimeout>>();
const TOAST_MS = 5200;

export function pushToast(tone: Toast['tone'], title: string, body?: string) {
  const id = uid('toast');
  // Cap the stack: past three, the oldest collapses rather than pushing the
  // rest off-screen. A toast nobody can read is not a notification.
  const next = [...state.toasts, { id, tone, title, body }];
  const overflow = next.length > 3 ? next.slice(0, next.length - 3) : [];
  for (const gone of overflow) {
    const t = toastTimers.get(gone.id);
    if (t) clearTimeout(t);
    toastTimers.delete(gone.id);
  }
  const kept = overflow.length ? next.slice(-3) : next;
  set({ toasts: kept });
  toastTimers.set(id, setTimeout(() => dismissToast(id), TOAST_MS));
}

/**
 * Stops a toast's clock while the pointer is over it, so reading one does not
 * race its own dismissal. Restarting gives a full window rather than the
 * remainder — the intent is "I am reading this", and a resumed half-second is
 * not enough to act on.
 */
export function holdToast(id: string) {
  const t = toastTimers.get(id);
  if (t) clearTimeout(t);
  toastTimers.delete(id);
}

export function releaseToast(id: string) {
  if (toastTimers.has(id)) return;
  if (!state.toasts.some((t) => t.id === id && !t.leaving)) return;
  toastTimers.set(id, setTimeout(() => dismissToast(id), TOAST_MS));
}

export function dismissToast(id: string) {
  const t = toastTimers.get(id);
  if (t) clearTimeout(t);
  toastTimers.delete(id);
  set((s) => ({ toasts: s.toasts.map((x) => (x.id === id ? { ...x, leaving: true } : x)) }));
  setTimeout(() => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })), 340);
}

/* ------------------------------------------------------------------- prefs */

/**
 * Preferences live on <html> as data attributes plus one custom property, so CSS
 * owns presentation and React owns state — the same split the theme already
 * used. Nothing here re-renders a screen: the attributes do the work.
 */
function applyPrefs() {
  const root = document.documentElement;
  setTranslationLanguage(state.language);
  root.lang = state.language === 'zh' ? 'zh-CN' : state.language;
  if(state.fontFamily)root.style.setProperty('--st-font',`${JSON.stringify(state.fontFamily)},system-ui,-apple-system,"Segoe UI",sans-serif`);
  else root.style.removeProperty('--st-font');
  root.dataset.theme = state.theme === 'system' ? matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light' : state.theme;
  document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.setAttribute('content', root.dataset.theme === 'dark' ? '#0e1113' : '#f9fafb');
  root.dataset.contrast = state.contrast;
  root.dataset.density = state.density;
  root.style.setProperty('--st-content-fs', `${state.contentFontSize}px`);
  // The chrome scale. The reading size is deliberately separate: someone who
  // needs larger prose does not necessarily want larger buttons.
  root.style.setProperty('--st-ui-scale', String(state.uiScale));

  // "system" means the OS decides, so the attribute is absent rather than set
  // to a third value the stylesheet would have to special-case.
  if (state.motion === 'system') delete root.dataset.motion;
  else root.dataset.motion = state.motion;

  if (state.focus) root.dataset.focus = 'true';
  else delete root.dataset.focus;

  // "translucent" is the default, so the attribute only appears when it is off.
  if (state.transparency === 'opaque') root.dataset.transparency = 'opaque';
  else delete root.dataset.transparency;
}

function savePrefs() {
  const snapshot: Prefs = {
    language: state.language,
    fontFamily: state.fontFamily,
    theme: state.theme,
    contrast: state.contrast,
    rail: state.rail,
    focus: state.focus,
    contentFontSize: state.contentFontSize,
    density: state.density,
    motion: state.motion,
    transparency: state.transparency,
    uiScale: state.uiScale,
    shareTelemetry: state.shareTelemetry,
  };
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(snapshot));
  } catch {
    /* a full quota must not break the UI */
  }
  applyPrefs();
}

export function setTheme(theme: 'dark' | 'light' | 'system') {
  set({ theme });
  savePrefs();
}

export function toggleRail() {
  set({ rail: !state.rail });
  savePrefs();
}
export function setLanguage(language:Language){if(!validLanguage(language))return;setTranslationLanguage(language);set({language});savePrefs();}
export function setFontFamily(fontFamily:string){set({fontFamily:fontFamily.trim().slice(0,128).replace(/[\x00-\x1f]/g,'')});savePrefs();}

export function setFocus(focus: boolean) {
  set({ focus });
  savePrefs();
}

export function toggleFocus() {
  setFocus(!state.focus);
}

/**
 * Puts every appearance and reading preference back to its shipped value.
 *
 * Deliberately excludes `shareTelemetry`: a privacy choice is not an appearance
 * setting, and re-enabling reporting as a side effect of resetting one's font
 * size would be exactly the kind of thing this control exists to prevent.
 *
 * Deliberately does **not** touch `rail` or `focus`: those are where you happen
 * to be working, not settings you chose, and collapsing the sidebar on someone
 * who asked to reset their font size would be a surprise. The engine has its own
 * reset, since an expert override and a font size have nothing to do with each
 * other.
 */
export function resetAppearance() {
  set({
    fontFamily: PREFS_DEFAULTS.fontFamily,
    theme: PREFS_DEFAULTS.theme,
    contrast: PREFS_DEFAULTS.contrast,
    contentFontSize: PREFS_DEFAULTS.contentFontSize,
    density: PREFS_DEFAULTS.density,
    motion: PREFS_DEFAULTS.motion,
    transparency: PREFS_DEFAULTS.transparency,
    uiScale: PREFS_DEFAULTS.uiScale,
  });
  savePrefs();
}

/**
 * Records the standing privacy choice.
 *
 * The control was local component state that nothing persisted and nothing
 * read — it snapped back on every reload, so a reader who turned reporting off
 * had no way to know their choice had not been recorded. A privacy control that
 * silently forgets is worse than no control at all.
 */
export function setShareTelemetry(on: boolean) {
  set({ shareTelemetry: on });
  savePrefs();
}

export function setContentFontSize(px: number) {
  const clamped = Math.min(CONTENT_FS_MAX, Math.max(CONTENT_FS_MIN, Math.round(px)));
  set({ contentFontSize: clamped });
  savePrefs();
}

export function setDensity(density: 'comfortable' | 'compact') {
  set({ density });
  savePrefs();
}

export function setMotion(motion: 'system' | 'full' | 'reduced') {
  set({ motion });
  savePrefs();
}

export function setTransparency(transparency: 'translucent' | 'opaque') {
  set({ transparency });
  savePrefs();
}

export const UI_SCALES = [0.9, 1, 1.1, 1.25] as const;

export function setUiScale(uiScale: number) {
  set({ uiScale });
  savePrefs();
}

export const CONTENT_FONT_RANGE = { min: CONTENT_FS_MIN, max: CONTENT_FS_MAX };

/* -------------------------------------------------------------- workspaces */

/** Opens the system folder chooser. Returns null when it is dismissed. */
export async function pickWorkspace(): Promise<string | null> {
  const res = await api.post<{ path?: string; cancelled?: boolean }>('/api/workspaces/pick', {});
  return res.path ?? null;
}

export async function addWorkspace(path: string) {
  try {
    const ws = await api.post<Workspace>('/api/workspaces', { path });
    set({ workspaces: [...state.workspaces, ws] });
    pushToast('success', 'Workspace added', `Strata can now read files in ${ws.name}.`);
  } catch (err) {
    pushToast('error', "Couldn't add that folder", err instanceof Error ? err.message : String(err));
  }
}

export async function removeWorkspace(id: string) {
  const ws = state.workspaces.find((w) => w.id === id);
  try {
    await api.del(`/api/workspaces/${id}`);
    set({ workspaces: state.workspaces.filter((w) => w.id !== id) });
    if (state.activeWorkspace === id) set({ activeWorkspace: state.workspaces[0]?.id ?? null });
  } catch (err) {
    pushToast('error', "Couldn't remove that workspace", err instanceof Error ? err.message : String(err));
  }
  void ws;
}

/**
 * Pins a conversation to the top of its workspace.
 *
 * Sessions are currently ordered by recency, which is right until you have
 * thirty of them and two matter. `Session.pinned` has existed in the type since
 * the beginning and was never written to.
 */
export function toggleArchive(id: string) {
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, archived: !s.archived } : s));
  persistSessions(sessions);
  set({ sessions });
}

export function togglePin(id: string) {
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, pinned: !s.pinned } : s));
  persistSessions(sessions);
  set({ sessions });
}

export function setActiveWorkspace(id: string) {
  set({ activeWorkspace: id });
  void loadWorkspaceFiles();
}

export function openSetup(open: boolean) {
  set({ setupOpen: open });
}

export function setPalette(open: boolean) {
  set({ paletteOpen: open });
}

export function setShortcuts(open: boolean) {
  set({ shortcutsOpen: open });
}

/* --------------------------------------------------------------- transport */

function applyEvent(e: StreamEvent) {
  const data = e.data as Record<string, unknown>;
  switch (e.event) {
    case 'monitor':
      set({ monitor: data as unknown as MonitorState });
      break;
    case 'skills':
      set({ skills: data as unknown as Skill[] });
      break;
    case 'models':
      if (Array.isArray(data.variants)) set({ library: data as unknown as ModelLibrary });
      break;
    case 'runtime':
      set({runtime:data as unknown as RuntimeModels});
      break;
    case 'image-models':
      if (Array.isArray(data.variants)) set({ imageModels: data as unknown as ImageModelLibrary });
      break;
    case 'providers':
      applyProviders(data as unknown as ModelProviders);
      break;
    case 'engine':
      set({ engine: data as unknown as EngineConfig });
      break;
    case 'serving':
      set({ serving: newestServingState(state.serving, data as unknown as ServingState) });
      break;
    case 'setup':
      set({ setup: data as unknown as SetupState });
      break;
    case 'task': {
      // A task finished on its own (the host ran it on schedule): say so, and say when it needs a person's yes or failed.
      const t = data as unknown as { title?: string; status?: string; attention?: string | null; error?: string | null };
      if (t.status === 'error') pushToast('error', 'Task failed', String(t.title ?? ''));
      else if (t.attention) pushToast('warn', 'Task needs your yes', String(t.title ?? ''));
      else if (t.status === 'done') pushToast('success', 'Task finished', String(t.title ?? ''));
      break;
    }
    case 'agent':
      if (typeof data.sessionId === 'string') patchSession(data.sessionId, s => ({ ...s, agents: updateAgent(s.agents ?? [], data.agent) }));
      break;
    default:
      break;
  }
}

let closeEvents: (() => void) | null = null;
export async function bootstrap() {
  applyPrefs();
  try {
    const [setup, hardware, library, engine, serving, monitor, skills, providers, imageModels, runtime] = await Promise.all([
      api.get<SetupState>('/api/setup'),
      api.get<Hardware>('/api/hardware'),
      api.get<ModelLibrary>('/api/models'),
      api.get<EngineConfig>('/api/engine'),
      api.get<ServingState>('/api/serving'),
      api.get<MonitorState>('/api/monitor'),
      api.get<Skill[]>('/api/skills'),
      api.get<ModelProviders>('/api/model-providers'),
      api.get<ImageModelLibrary>('/api/image-models'),
      api.get<RuntimeModels>('/api/runtime'),
    ]);
    const workspaceState = await api.get<{ workspaces: Workspace[] }>('/api/workspaces');
    set({
      ready: true,
      setup,
      hardware,
      library: libraryResponse(library),
      imageModels,
      runtime,
      engine,
      serving: newestServingState(state.serving, serving),
      monitor,
      skills,
      workspacePath: workspaceState.workspaces[0]?.path ?? '',
      workspaces: workspaceState.workspaces,
      activeWorkspace: workspaceState.workspaces.some(w => w.id === state.activeWorkspace) ? state.activeWorkspace : workspaceState.workspaces[0]?.id ?? null,
      connection: 'live',
    });
    applyProviders(providers);
    closeEvents?.();
    let reconnect = false;
    closeEvents = onEvent(applyEvent, (status) => {
      if (status === 'offline') reconnect = true;
      if (status === 'live' && reconnect) {
        reconnect = false;
        // The restarted host may have new logs, config or readiness. Refresh
        // those facts in place while retaining conversations and unsent drafts.
        void bootstrap();
        return;
      }
      set({ connection: status });
    });
    if (!setup.complete && !setup.dismissed && !window.location.hash.includes('setup')) navigate('setup');
  } catch (err) {
    set({ ready: true, connection: 'offline' });
    pushToast(
      'error',
      'Engine host unreachable',
      err instanceof Error ? err.message : 'Start it with `npm run host` and reload.',
    );
  }
}

/* ------------------------------------------------------------- reads/writes */

export async function refreshHardware() {
  set((s) => ({ busy: { ...s.busy, hardware: true } }));
  try {
    set({ hardware: await api.get<Hardware>('/api/hardware?reprobe=1') });
  } finally {
    set((s) => ({ busy: { ...s.busy, hardware: false } }));
  }
}

function modelFits(id: string) {
  const model = state.library?.variants.find(v => v.id === id);
  if (state.library?.mode === 'real' && model && state.hardware && state.engine && ['too-large', 'unsupported'].includes(estimate(model, state.hardware, execution(Object.fromEntries(state.engine.expert.map(p => [p.id, p.value])), contextTokens(state.engine.beginner.contextLength))).fit)) {
    pushToast('warn', 'Choose a smaller model', 'The estimated memory requirement exceeds this computer.');
    return false;
  }
  return true;
}

export async function selectImageModel(selected: ImageModelChoice) {
  set(s => ({ busy: { ...s.busy, 'image:select': true } }));
  try { set({ imageModels: await api.patch<ImageModelLibrary>('/api/image-models', { selected }) }); }
  finally { set(s => ({ busy: { ...s.busy, 'image:select': false } })); }
}
export async function installImageModel(id: Exclude<ImageModelChoice, 'off'>) {
  set({ imageModels: await api.post<ImageModelLibrary>(`/api/image-models/${id}/install`) });
}
export async function cancelImageModel(id: Exclude<ImageModelChoice, 'off'>) {
  set({ imageModels: await api.post<ImageModelLibrary>(`/api/image-models/${id}/cancel`) });
}
export async function applyImageModel() {
  set({ imageModels: await api.post<ImageModelLibrary>('/api/image-models/apply') });
}

export async function installVariant(id: string) {
  // Download eligibility is checked against disk space by the host. Runtime fit
  // is reviewed separately when the model is loaded.
  if (state.busy[`install:${id}`]) return false;
  set((s) => ({ busy: { ...s.busy, [`install:${id}`]: true } }));
  try {
    const library = libraryResponse(await api.post<ModelLibrary>(`/api/models/${id}/install`));
    set({ library });
    const complete = library.variants.find(v => v.id === id)?.state === 'installed';
    pushToast(complete ? 'success' : 'info', complete ? 'Files installed' : library.mode === 'simulated' ? 'Example installation started' : 'Download started', complete ? 'Load the model to start a request.' : 'Progress is shown in Models.');
    return true;
  } catch (err) {
    pushToast('error', 'Download failed', err instanceof Error ? err.message : undefined);
    return false;
  } finally {
    set((s) => ({ busy: { ...s.busy, [`install:${id}`]: false } }));
  }
}

export async function cancelInstall(id: string) {
  try { set({ library: libraryResponse(await api.post<ModelLibrary>(`/api/models/${id}/cancel`)) }); } catch (e) { pushToast('error', 'Cancel failed', e instanceof Error ? e.message : String(e)); }
}

export async function loadVariant(id: string) {
  if (!modelFits(id)) return false;
  if (!lifecycleAvailable()) return false;
  set((s) => ({ busy: { ...s.busy, [`load:${id}`]: true }, actionError: null }));
  try {
    const library = libraryResponse(await api.post<ModelLibrary>(`/api/models/${id}/load`));
    set({ library });
    const model = library.variants.find(v => v.id === id);
    if (library.loaded !== id || model?.state !== 'loaded') throw new Error(model?.error ?? 'The model did not finish loading.');
    pushToast('success', library.mode === 'simulated' ? 'Example model ready' : 'Model loaded', library.mode === 'simulated' ? 'Answers are simulated samples, not from a real model.' : 'Ready for a request.');
    return true;
  } catch (err) {
    const message = err instanceof Error ? err.message : 'The model did not load.';
    set({
      actionError: {
        label: 'Could not load model',
        message,
        retry: () => void loadVariant(id),
      },
    });
    pushToast('error', 'Could not load model', message);
    return false;
  } finally {
    set((s) => ({ busy: { ...s.busy, [`load:${id}`]: false } }));
  }
}

export async function unloadModel() {
  if (!lifecycleAvailable()) return false;
  set(s => ({ busy: { ...s.busy, unload: true } }));
  try {
    const previousInstance=state.runtime?.defaultId;
    const library = libraryResponse(await api.post<ModelLibrary>('/api/models/unload'));
    const runtime=await api.get<RuntimeModels>('/api/runtime');
    set({runtime});
    set({ library });
    if (previousInstance?runtime.instances.some(i=>i.id===previousInstance):library.loaded!==null) throw new Error('The engine still reports this model instance as loaded.');
    pushToast('info', 'Model unloaded', library.mode === 'simulated' ? 'Example load state cleared.' : 'Model memory released.');
    return true;
  } catch (err) { pushToast('error', 'Unload failed', err instanceof Error ? err.message : String(err)); return false; }
  finally { set(s => ({ busy: { ...s.busy, unload: false } })); }
}

export async function removeVariant(id: string) {
  if (state.library?.loaded === id && !lifecycleAvailable()) return false;
  const local = !!state.library?.variants.find(v => v.id === id)?.local;
  try { const library = libraryResponse(await api.del<ModelLibrary>(`/api/models/${id}`)); set({ library });
    const remaining = library.variants.find(v => v.id === id);
    if (remaining && remaining.state !== 'available') throw new Error('The model files were not removed.');
    pushToast('success', local ? 'Local model removed' : 'Files removed', local ? 'The original GGUF files are kept.' : library.mode === 'simulated' ? 'Example install state cleared.' : 'Model files removed from disk.'); return true;
  } catch (e) { pushToast('error', 'Remove failed', e instanceof Error ? e.message : String(e)); return false; }
}

export async function waitForInstalled(id: string) {
  const deadline = Date.now() + 60 * 60 * 1000;
  while (Date.now() < deadline) {
    const v = state.library?.variants.find(v => v.id === id);
    if (v && ['installed', 'loaded'].includes(v.state)) return;
    if (!v || ['available', 'error'].includes(v.state)) throw new Error(v?.error ?? 'Download cancelled.');
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error('Download timed out. Try again from Models.');
}

export async function saveEngine(patch: Partial<EngineConfig> | Record<string, unknown>, silent = false, requireIdle = false) {
  if (requireIdle) {
    const status = lifecycleStatus();
    if (!status.allowed) throw new Error(status.reason);
  }
  const engine = await api.patch<EngineConfig>('/api/engine', patch);
  set({ engine, actionError: null });
  if (!silent) pushToast('success', 'Settings saved', 'Load settings take effect on the next load. Sampling defaults affect the next request.');
  return engine;
}

export async function restartEngine() {
  if (state.library?.mode === 'simulated') { pushToast('info', 'Restart unavailable with the example engine', 'Save settings for the next load.'); return false; }
  if (!lifecycleAvailable()) return false;
  set((s) => ({ busy: { ...s.busy, restart: true }, actionError: null }));
  try {
    const engine = await api.post<EngineConfig>('/api/engine/restart');
    set({ engine });
    if (!engine.engine.running) throw new Error(engine.engine.error ?? 'The engine is not running. Load a model before applying settings.');
    pushToast('success', 'Engine restarted', 'Warm-up takes a few seconds.');
    return true;
  } catch (err) {
    // Reported in place as well as in a toast: the toast is a summary, the
    // surface that failed is where the retry belongs.
    set({
      actionError: {
        label: 'Restart failed',
        message: err instanceof Error ? err.message : 'The engine did not come back up.',
        retry: () => void restartEngine(),
      },
    });
    pushToast('error', 'Restart failed', err instanceof Error ? err.message : undefined);
    return false;
  } finally {
    set((s) => ({ busy: { ...s.busy, restart: false } }));
  }
}

export function dismissActionError() {
  set({ actionError: null });
}

export async function resetEngine() {
  set({ engine: await api.post<EngineConfig>('/api/engine/reset') });
  pushToast('info', 'Back to recommended', 'Every value returned to the safe default.');
}

/**
 * Checks an engine config — the saved one by default, or a draft to preview what
 * a change would do before committing to it.
 */
export async function validateEngine(draft?: unknown) {
  return api.post<{
    ok: boolean;
    issues: { level: 'warn' | 'error'; message: string }[];
    predicted?: EngineConfig['predicted'];
  }>('/api/engine/validate', draft ?? {});
}

export async function exportEngine() {
  return api.get<Record<string, unknown>>('/api/engine/export');
}

export async function setCloudflare(body: {action: 'start' | 'stop' | 'prepare'; mode?: 'quick' | 'domain'; hostname?: string; token?: string}) {
  const result = await api.post<ServingState>('/api/serving/tunnel', body);
  set({ serving: newestServingState(state.serving, result) });
}

export async function setLocalSharing(enabled: boolean) {
  const result = await api.post<ServingState>('/api/serving/share', { enabled });
  set({ serving: newestServingState(state.serving, result) });
}

export async function setServingAccess(patch: { otherDevices?: boolean; apiKey?: string | null }) {
  const result = await api.patch<ServingState>('/api/serving', patch);
  set({ serving: newestServingState(state.serving, result) });
}

export async function pauseServing(paused: boolean) {
  const result = await api.patch<ServingState>('/api/serving', { pauseServing: paused });
  set({ serving: newestServingState(state.serving, result) });
}

export async function cancelRequest(id: string) {
  const result = await api.post<ServingState>(`/api/serving/requests/${id}/cancel`);
  set({ serving: newestServingState(state.serving, result) });
}

export async function completeSetupStep(step: number, choices?: SetupState['choices'], dismissed = false) {
  set({ setup: await api.post<SetupState>('/api/setup/step', { step, choices, dismissed, flowVersion: 2 }) });
}

export async function completeSetup() {
  const setup = await api.post<SetupState>('/api/setup/complete');
  set({ setup });
}

export async function setSkills(skills: Skill[]) {
  set({ skills });
}

/**
 * Re-reads the skill list from the host.
 *
 * For the case where a skill was added to the workspace folder while the app was
 * open — otherwise the only way to see it is to restart.
 */
export async function loadSkills() {
  set({ skills: await api.get<Skill[]>('/api/skills') });
}

/** Re-reads the workspace tree from disk, for the same reason. */
export async function loadWorkspaceFiles() {
  const data = await api.get<{ path: string; entries: WorkspaceFile[] }>(`/api/workspace${state.activeWorkspace ? '?root=' + encodeURIComponent(state.activeWorkspace) : ''}`);
  set({ files: data.entries, workspacePath: data.path });
}

export async function toggleSkill(id: string, enabled: boolean) {
  set({ skills: await api.patch<Skill[]>(`/api/skills/${encodeURIComponent(id)}`, { enabled }) });
}

export async function editSkill(id: string, patch: Pick<Skill, 'name' | 'description' | 'body' | 'triggers' | 'enabled' | 'onlyIn'>) {
  set({ skills: await api.patch<Skill[]>(`/api/skills/${encodeURIComponent(id)}`, patch) });
}

/**
 * Narrows a skill to the workspace you are in, or widens it again.
 *
 * Passing null clears the narrowing. The host resolves `onlyIn` against the
 * workspace the request came from, so the answer depends on where the reader
 * is rather than on a single global switch.
 */
export async function scopeSkill(id: string, workspaceId: string | null) {
  set({ skills: await api.patch<Skill[]>(`/api/skills/${encodeURIComponent(id)}`, { onlyIn: workspaceId }) });
}

/* ---------------------------------------------------------------- sessions */

function persistSessions(sessions: Session[]) {
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(sessions));
  } catch {
    /* quota — sessions are a cache, not the source of truth */
  }
}

export function newSession(): Session {
  const session: Session = {
    id: uid('s'),
    title: 'New conversation',
    defaultTitle: true,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    messages: [],
    workspace: state.workspaces.find(w => w.id === state.activeWorkspace)?.path ?? state.workspacePath,
    tokensIn: 0,
    tokensOut: 0,
    ...(state.activeSession === null && state.remoteModel ? { preferences: { remoteModel: state.remoteModel } } : {}),
  };
  const sessions = [session, ...state.sessions];
  persistSessions(sessions);
  set({ sessions, activeSession: session.id });
  return session;
}

export function setChatPreferences(id: string | undefined, preferences: NonNullable<Session['preferences']>) {
  const sessionId = id ?? newSession().id;
  patchSession(sessionId, s => ({ ...s, preferences: { ...s.preferences, ...preferences } }));
}
export function setPreviewTabClosed(sessionId:string|undefined, artifactId:string, closed:boolean, selected:string|null) {
  if(!sessionId)return;
  patchSession(sessionId,s=>({...s,closedPreviews:closed?[...new Set([...(s.closedPreviews??[]),artifactId])]:(s.closedPreviews??[]).filter(id=>id!==artifactId),previewOpen:selected??undefined}));
}
export function createChatShowcase() {
  const session=newSession();const at=new Date().toISOString();
  patchSession(session.id,s=>({...s,title:'Chat showcase · code, tools & media',showcase:true,statsAvailable:true,tokensIn:28000000,tokensOut:4000000,messages:showcaseMessages(at),agents:SHOWCASE_AGENTS,showcaseAgentStates:true}));
  go('chat',session.id);
}

/** A visible example for checking the subagent interface in the design build. */
export function createSubagentExample() {
  const session = newSession(); const now = Date.now();
  patchSession(session.id, s => ({ ...s, title: 'Subagent preview', showcase: true, showcaseAgentStates: true,
    agents: [
      {id:'example-layout',name:'Layout review',description:'Checking spacing and component alignment.',status:'working',durationMs:39000,tokens:12600,updatedAt:now},
      {id:'example-interactions',name:'Interaction review',description:'Checking menus, controls and keyboard navigation.',status:'working',durationMs:21000,tokens:8400,updatedAt:now},
      {id:'example-copy',name:'Copy review',description:'Labels and descriptions reviewed.',status:'completed',durationMs:18000,tokens:4200,updatedAt:now},
    ], messages: [{id:uid('m'),role:'assistant',at:new Date(now).toISOString(),example:true,content:'Open the subagent count above to inspect this example run. Each row shows status, token usage and elapsed time.'}]
  }));
  go('chat', session.id);
}

export function createPreviewExamples() {
  const session = newSession();
  const at = new Date().toISOString();
  patchSession(session.id, s => ({ ...s, title: 'Design review · six previews', previewOpen:'sample-website', messages: [
    { id:uid('m'), role:'user', at, content:'Open six different output types so I can iterate on the preview design.' },
    { id:uid('m'), role:'assistant', at, demo:true, content:'Here is a design-review workspace for Fieldnotes: a website, PDF, Markdown brief, cover image, TypeScript file and release table.\n\nAll six are loaded in the panel on the right. Switch tabs to explore each view while keeping this conversation in place.', artifacts:PREVIEW_EXAMPLES },
  ] }));
  go('chat', session.id);
}

export function selectSession(id: string) {
  set({ activeSession: id });
  markRead(id);
}

/** Records that a conversation has been looked at. */
export function markRead(id: string) {
  const at = new Date().toISOString();
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, lastReadAt: at } : s));
  persistSessions(sessions);
  set({ sessions });
}

/**
 * Whether a conversation has moved on since it was last looked at.
 *
 * The active conversation is never unread: you are looking at it, or its reply
 * is arriving in front of you.
 */
export function isUnread(s: Session, activeSession: string | null): boolean {
  if (s.id === activeSession) return false;
  if (!s.lastReadAt) return s.messages.length > 0;
  return s.updatedAt > s.lastReadAt;
}

export function deleteSession(id: string) {
  const sessions = state.sessions.filter((s) => s.id !== id);
  persistSessions(sessions);
  set({ sessions, activeSession: state.activeSession === id ? (sessions[0]?.id ?? null) : state.activeSession });
}

export function renameSession(id: string, title: string) {
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, title, defaultTitle: false } : s));
  persistSessions(sessions);
  set({ sessions });
}

function patchSession(id: string, fn: (s: Session) => Session) {
  const sessions = state.sessions.map((s) => (s.id === id ? fn(s) : s));
  persistSessions(sessions);
  set({ sessions });
}

// The demo clock belongs to the session, so hiding the sidebar or navigating
// away never pauses it. Only start and completion are written to storage.
const timeSkipTimers = new Map<string, ReturnType<typeof setTimeout>>();
function scheduleTimeSkip(sessionId: string, run: TimeSkipRun) {
  const timerId = `${sessionId}:${run.id}`;
  clearTimeout(timeSkipTimers.get(timerId));
  if (run.completedAt) return;
  timeSkipTimers.set(timerId, setTimeout(() => {
    timeSkipTimers.delete(timerId);
    const session = state.sessions.find(s => s.id === sessionId);
    const current = session?.timeSkips?.find(r => r.id === run.id);
    if (!current || current.startedAt !== run.startedAt || !session?.messages.some(m => m.timeSkipId === run.id && m.role === 'assistant')) return;
    patchSession(sessionId, s => ({ ...s, updatedAt: new Date().toISOString(),
      timeSkips: s.timeSkips?.map(r => r.id === run.id ? { ...r, completedAt: run.startedAt + TIME_SKIP_DURATION } : r),
      messages: s.messages.map(m => m.timeSkipId === run.id && m.role === 'assistant' ? { ...m, content: 'The updated result is ready. Switch between Original and Updated, or compare them side by side. This example uses sample outputs and simulated checks.' } : m),
    }));
  }, Math.max(0, run.startedAt + TIME_SKIP_DURATION - Date.now())));
}

export function replayTimeSkip(sessionId: string, runId: string) {
  const run = state.sessions.find(s => s.id === sessionId)?.timeSkips?.find(r => r.id === runId);
  if (!run) return;
  const next = { ...run, startedAt: Date.now(), completedAt: undefined, retainedVersion: undefined };
  patchSession(sessionId, s => ({ ...s, timeSkips: s.timeSkips?.map(r => r.id === runId ? next : r),
    messages: s.messages.map(m => m.timeSkipId === runId && m.role === 'assistant' ? { ...m, content: 'Revising the saved draft. Follow the result in the sidebar. This example uses sample outputs and simulated checks.' } : m),
  }));
  scheduleTimeSkip(sessionId, next);
}

export function setTimeSkipResult(sessionId: string, runId: string, version: TimeSkipVersion) {
  const run = state.sessions.find(s => s.id === sessionId)?.timeSkips?.find(r => r.id === runId);
  if (!run) return;
  const next = withTimeSkipResult(run, version, Date.now());
  if (next === run) return;
  patchSession(sessionId, s => ({ ...s, timeSkips: s.timeSkips?.map(r => r.id === runId ? next : r) }));
}

function startTimeSkipDemo(prompt: string) {
  if (!prompt) { pushToast('info', 'Add a prompt after /time-skip', 'For example: /time-skip Make a launch plan for my app'); return; }
  const sessionId = state.activeSession ?? newSession().id;
  const run: TimeSkipRun = { id: uid('ts'), prompt, startedAt: Date.now() };
  const at = new Date().toISOString();
  patchSession(sessionId, s => ({ ...s,
    defaultTitle: false,
    title: s.messages.length ? s.title : prompt.length>48?prompt.slice(0,47).trimEnd()+'…':prompt, updatedAt: at,
    timeSkips: [...(s.timeSkips ?? []), run], previewOpen: timeSkipTabId(run.id),
    messages: [...s.messages,
      { id: uid('m'), role: 'user', at, content: `/time-skip ${prompt}`, timeSkipId: run.id },
      { id: uid('m'), role: 'assistant', at, example: true, timeSkipId: run.id, content: 'Revising the saved draft. Follow the result in the sidebar. This example uses sample outputs and simulated checks; it doesn’t run your model.' },
    ],
  }));
  scheduleTimeSkip(sessionId, run);
  go('chat', sessionId);
}

function markTurn(sessionId: string, turn: { abort: AbortController; agent: boolean } | null) {
  if (turn) liveTurns.set(sessionId, turn); else liveTurns.delete(sessionId);
  set({ streaming: !!state.activeSession && liveTurns.has(state.activeSession), liveChats: liveTurns.size });
  document.documentElement.dataset.streaming = liveTurns.size ? 'true' : 'false';
}

/**
 * Stop the open chat's turn. The host is asked first: for an agent turn it cancels exactly this chat's
 * Harness session and answers only once the turn has ended, and the stream then closes by itself with
 * what was produced. For any other turn the host reports nothing to stop and the request is aborted.
 */
export function stopStreaming() {
  const id = state.activeSession;
  const turn = id ? liveTurns.get(id) : undefined;
  if (!id || !turn) return;
  const abort = () => { turn.abort.abort(); };
  void api.post<{ stopped: boolean }>('/api/chat/stop', { sessionId: id }).then(
    result => {
      if (!result.stopped) abort();
      // An agent turn that was told to stop but is slow to end must not leave the chat stuck.
      else setTimeout(() => { if (liveTurns.get(id) === turn) abort(); }, 30000);
    },
    abort,
  );
}

/**
 * Re-runs the last exchange: drops the assistant reply and asks again.
 *
 * Kept in the store rather than the screen because it has to touch the message
 * list, and every attempt to do this from a component ends up duplicating the
 * stream handling that already lives here.
 */
export async function retryLast(messageId?: string) {
  const source = state.sessions.find(s => s.id === state.activeSession);
  if (!source || state.streaming) return;
  const end = messageId ? source.messages.findIndex(m => m.id === messageId) : source.messages.length - 1;
  const user = source.messages.slice(0, end + 1).reverse().find(m => m.role === 'user');
  if (!user) return;
  const prompt = rewindTo(user.id);
  if (prompt !== null) {
    // Retry sends immediately; only Edit should leave the prompt in the composer.
    sessionStorage.setItem(`strata.chat-draft.${state.activeSession}`, '');
    await sendMessage(prompt);
  }
}

/**
 * Forks a conversation at a message, leaving the original untouched.
 *
 * The alternative people reach for is duplicating the whole session and
 * deleting messages by hand, which loses the original. Branching keeps both:
 * the new session carries the history up to the chosen turn and diverges there.
 */
export function branchFrom(messageId: string): string | null {
  const sessionId = state.activeSession;
  if (!sessionId) return null;
  const source = state.sessions.find((s) => s.id === sessionId);
  if (!source) return null;
  const at = source.messages.findIndex((m) => m.id === messageId);
  if (at < 0) return null;

  const now = new Date().toISOString();
  const branch: Session = {
    ...source,
    id: uid('s'),
    title: `${source.title} (branch)`,
    defaultTitle: false,
    createdAt: now,
    updatedAt: now,
    messages: source.messages.slice(0, at + 1).map((m) => ({ ...m, id: uid('m') })),
    tokensIn: 0,
    tokensOut: source.messages.slice(0, at + 1).reduce((n, m) => n + (m.tokens ?? 0), 0),
    statsAvailable: false,
  };
  const sessions = [branch, ...state.sessions];
  persistSessions(sessions);
  set({ sessions, activeSession: branch.id });
  for (const run of branch.timeSkips ?? []) scheduleTimeSkip(branch.id, run);
  // The URL wins over `activeSession` when it names a session, so a programmatic
  // switch has to move the route too — otherwise the sidebar shows the branch
  // while the transcript still shows the original, which reads as a broken
  // button rather than a created branch.
  navigate('chat', branch.id);
  return branch.id;
}

/**
 * Asks the model to keep going from where it stopped.
 *
 * Sent as an ordinary user turn rather than a special mode, because that is
 * what it is: a continuation is a request, and the transcript should show it.
 */
export async function continueLast() {
  const sessionId = state.activeSession;
  if (!sessionId || state.streaming) return;
  const session = state.sessions.find((s) => s.id === sessionId);
  const last = session?.messages[session.messages.length - 1];
  if (!last || last.role !== 'assistant' || !last.content.trim()) return;
  await sendMessage('Continue from where you left off. Do not repeat what you have already written.');
}

/**
 * Rewinds the transcript to just before a turn so it can be re-asked with a
 * different prompt. Returns the text so the composer can be filled with it.
 */
export function rewindTo(messageId: string): string | null {
  if (state.streaming) return null;
  const source = state.sessions.find(s => s.id === state.activeSession);
  const at = source?.messages.findIndex(m => m.id === messageId) ?? -1;
  if (!source || at < 0) return null;
  const prompt = source.messages[at].content;
  const branchId = branchFrom(messageId);
  if (!branchId) return null;
  patchSession(branchId, s => ({ ...s, messages: s.messages.slice(0, -1), tokensIn: 0, tokensOut: s.messages.slice(0, -1).reduce((n, m) => n + (m.tokens ?? 0), 0) }));
  sessionStorage.setItem(`strata.chat-draft.${branchId}`, prompt);
  return prompt;
}

export function exportConversation() {
  const session = state.sessions.find(s => s.id === state.activeSession);
  if (!session) return;
  const text = '# ' + session.title + '\n\n' + session.messages.map(m => '## ' + m.role + '\n\n' + m.content).join('\n\n');
  const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown' }));
  const a = document.createElement('a'); a.href = url; a.download = session.title.replace(/[^a-z0-9 -]/gi, '').slice(0, 60) + '.md';
  document.body.append(a); a.click(); a.remove();
  // Keep the Blob alive until browsers have started consuming the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function sendMessage(text: string) {
  if(state.streaming)return;
  const demoPrompt = timeSkipPrompt(text.trim());
  if (demoPrompt !== null) { startTimeSkipDemo(demoPrompt); return; }
  const imageTarget=nativeState()?.instances.find(i=>i.id===nativeState()?.defaultId);const canSendPictures=!state.remoteModel&&imageTarget?.status==='ready'&&imageTarget.vision;
  if(pendingChatPhotos(state.activeSession).length&&!canSendPictures){pushToast('error','Choose an image-reading model','Switch to a local model with Read pictures enabled, or remove the attachments.');return;}
  const pictures=takeChatPhotos(state.activeSession);
  const body=text.trim()||(pictures.length?'Describe this picture.':'');
  if(!body)return;

  let sessionId = state.activeSession;
  if (!sessionId) sessionId = newSession().id;

  const userMsg: ChatMessage = { id: uid('m'), role: 'user', content: body, at: new Date().toISOString(),...(pictures.length?{photos:pictures}:{}) };
  const assistantId = uid('m');
  const assistantMsg: ChatMessage = {
    id: assistantId,
    role: 'assistant',
    content: '',
    at: new Date().toISOString(),
    streaming: true,
  };

  /* Everything said before this turn, which the model needs in order to have a
     conversation at all.

     It was never sent. The real-engine path builds its prompt as
     `[...history, message]`, and `history` was always empty, so each message
     arrived as the first thing the model had ever heard — no memory of the
     turns above it in the transcript. The demo path hides this, because its
     replies are canned.

     The context meter was reading the same empty array, which is why it sat at
     ~350 tokens however long the conversation grew. */
  const prior = (state.sessions.find((s) => s.id === sessionId)?.messages ?? [])
    .filter(m => !m.timeSkipId)
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .filter((m) => m.content.trim())
    .map((m) => ({ role: m.role, content: m.content,photos:m.photos }))
    // Bounded so a long conversation cannot send an unbounded body.
    .slice(-40);

  patchSession(sessionId, (s) => ({
    ...s,
    defaultTitle: false,
    title: s.messages.length === 0 ? (body.length>48?body.slice(0,47).trimEnd()+'…':body) : s.title,
    updatedAt: new Date().toISOString(),
    messages: [...s.messages, userMsg, assistantMsg],
  }));

  const turnAbort = new AbortController();
  markTurn(sessionId, { abort: turnAbort, agent: false });
  // Read by the router to skip view transitions mid-stream, and by CSS to keep
  // the transcript still while tokens arrive.

  const patchAssistant = (fn: (m: ChatMessage) => ChatMessage) =>
    patchSession(sessionId, (s) => ({
      ...s,
      updatedAt: new Date().toISOString(),
      messages: s.messages.map((m) => (m.id === assistantId ? fn(m) : m)),
    }));

  const started = performance.now();

  const imageSlots=photoHistorySlots(prior,pictures.length);
  try {
    await streamPost(
      '/api/chat/stream',
      {
        sessionId,
        message: body,
        history:await Promise.all(prior.map(async(m,index)=>{const included=canSendPictures?(m.photos??[]).filter((_,p)=>imageSlots.has(`${index}:${p}`)):[];const omitted=(m.photos??[]).filter(p=>!included.includes(p));const text=m.content+(omitted.length?`\n[Earlier pictures: ${omitted.map(p=>p.name).join(', ')}. Their pixels are outside this model's image context; ask to reattach them if needed.]`:'');return {role:m.role,content:included.length?[{type:'text',text},...await Promise.all(included.map(async photo=>({type:'image_url',image_url:{url:(await photoData(photo)).url}})))]:text};})),
        model: state.library?.loaded ?? state.library?.current ?? null,
        remoteModel: state.remoteModel,
        nativeModelId:nativeState()?.instances.length?nativeState()?.defaultId:undefined,
        images:await Promise.all(pictures.map(photoData)),
        useTools:state.sessions.find(s=>s.id===sessionId)?.preferences?.useTools===true,
        // A chat set to plain talk (preferences.agent === false) never runs a tool: the host answers it directly.
        ...(state.sessions.find(s=>s.id===sessionId)?.preferences?.agent===false?{agent:false}:{}),
        thinking:state.sessions.find(s=>s.id===sessionId)?.preferences?.thinking??'High',
        thinkingBudget:state.sessions.find(s=>s.id===sessionId)?.preferences?.thinkingBudget??0,
        // The Mode menu's choice for this chat, read when the turn starts; the host applies it to the agent.
        mode:state.sessions.find(s=>s.id===sessionId)?.preferences?.mode??'manual',
        // So a skill narrowed to another workspace does not fire here.
        workspaceId: state.activeWorkspace,
      },
      (frame) => {
        const type = frame.type as string;
        if (type === 'start' && frame.agent === true) { const turn = liveTurns.get(sessionId); if (turn) turn.agent = true; }
        if (type === 'queued') {
          patchAssistant(m => ({ ...m, requestId: String(frame.requestId) }));
        } else if (type === 'start') {
          patchAssistant((m) => ({ ...m, demo: frame.demo === true, remoteModel: frame.remoteModel === true, requestId: typeof frame.requestId === 'string' ? frame.requestId : m.requestId, model: String(frame.model), contextMax: Number(frame.contextMax), skillsUsed: Array.isArray(frame.skills) ? frame.skills.map(String) : [] }));
        } else if (type === 'reasoning') {
          patchAssistant((m) => ({ ...m, reasoning: (m.reasoning ?? '') + String(frame.text ?? '') }));
        } else if (type === 'notice') {
          // The host says why this chat is not using the agent. One toast per reason per page load.
          const code = String(frame.code ?? 'notice');
          if (!noticed.has(code)) { noticed.add(code); pushToast('info', 'Agent', plainFailureText(String(frame.message ?? '')).message); }
        } else if (type === 'agent-tool') {
          const tool = frame.tool as ToolCall;
          patchAssistant(m => {
            const existing = m.tools?.find(t => t.id === tool.id);
            const next: ToolCall = { ...existing, ...tool, live: true };
            return { ...m, tools: existing ? m.tools!.map(t => t.id === tool.id ? next : t) : [...(m.tools ?? []), next] };
          });
        } else if (type === 'agent') {
          patchSession(sessionId, s => ({ ...s, agents: updateAgent(s.agents ?? [], frame.agent) }));
        } else if (type === 'delta') {
          patchAssistant((m) => { const content=m.content+String(frame.text??''); const count=Math.max(1,Math.ceil((content.length+(m.reasoning?.length??0))/4)); return {...m,content,tokens:m.usageSource==='estimated'&&m.tps?m.tokens:count,inputTokens:m.inputTokens??Math.ceil((body.length+prior.reduce((n,p)=>n+p.content.length,0))/4),usageSource:'estimated'}; });
        } else if (type === 'artifacts' && Array.isArray(frame.artifacts)) {
          patchAssistant(m => ({ ...m, artifacts: frame.artifacts as PreviewArtifact[] }));
        } else if(type==='native-tool'){
          const event=frame.event as {event:string;id:string;name?:string;arguments?:Record<string,unknown>;text?:string;ok?:boolean;ms?:number};
          if(event.id)patchAssistant(m=>{const existing=m.tools?.find(t=>t.id===event.id);const tool={id:event.id,name:event.name??existing?.name??'Connected tool',args:event.arguments??existing?.args??{},preview:'Authorized connected tool',effect:'Access configured in the MCP server',scope:'connected' as const,state:event.event==='result'?event.ok?'done' as const:'error' as const:'running' as const,result:event.text,durationMs:event.ms};return {...m,tools:[...(m.tools??[]).filter(t=>t.id!==tool.id),{...existing,...tool}]};});
        } else if (type === 'tool') {
          const tool = frame.tool as ChatMessage['tools'] extends (infer T)[] | undefined ? T : never;
          patchAssistant((m) => ({ ...m, tools: [...(m.tools ?? []), { ...tool, state: 'proposed' }] }));
        } else if (type === 'stats') {
          patchAssistant((m) => ({
            ...m,
            tokens: typeof frame.tokensOut === 'number' ? frame.tokensOut : m.tokens,
            tps: typeof frame.tps === 'number' ? frame.tps : m.tps,
          }));
        } else if (type === 'error') {
          patchAssistant((m) => ({ ...m, error: plainFailure(String(frame.message ?? 'Engine error')).message, streaming: false }));
        } else if (type === 'done') {
          patchSession(sessionId, (sess) => ({
            ...sess,
            tokensIn: sess.tokensIn + Number(frame.tokensIn ?? 0),
            tokensOut: sess.tokensOut + Number(frame.tokensOut ?? 0),
            statsAvailable: typeof frame.tokensIn === 'number' && typeof frame.tokensOut === 'number' && sess.messages.filter(m=>m.role==='assistant'&&!m.streaming).every(m=>typeof m.inputTokens==='number'),
          }));
          patchAssistant((m) => ({
            ...m,
            streaming: false,
            durationMs: Number(frame.durationMs ?? performance.now() - started),
            tokens: typeof frame.tokensOut === 'number' ? frame.tokensOut : undefined,
            ttftMs: typeof frame.ttftMs === 'number' ? frame.ttftMs : undefined,
            inputTokens: typeof frame.tokensIn === 'number' ? frame.tokensIn : undefined,
            contextUsed: typeof frame.contextTokens === 'number' ? frame.contextTokens : undefined,
            usageSource: frame.usageSource === 'reported' || frame.usageSource === 'demo' ? frame.usageSource : 'unavailable',
            tps: typeof frame.tps === 'number' ? frame.tps : undefined,
          }));
        }
      },
      turnAbort.signal,
    );
  } catch (err) {
    if ((err as Error).name !== 'AbortError') {
      patchAssistant((m) => ({
        ...m,
        streaming: false,
        error: err instanceof Error ? err.message : 'Engine error',
      }));
      pushToast('error', 'Generation failed', err instanceof Error ? err.message : undefined);
    } else {
      patchAssistant((m) => ({ ...m, streaming: false }));
    }
  } finally {
    markTurn(sessionId, null);
  }
}

/* -------------------------------------------------------------- tool grants */

const noticed = new Set<string>();

/**
 * Approve or deny a tool the agent asked about. The answer goes to the host, which hands it to the
 * Harness approval for exactly this chat and call; the card then follows what the Harness reports
 * (running, done, denied). The card shows the choice straight away and goes back if the host refuses.
 */
function answerAgentTool(sessionId: string, messageId: string, tool: ToolCall, approved: boolean) {
  const setState = (state_: ToolCall['state'], result?: string) => patchSession(sessionId, s => ({
    ...s, messages: s.messages.map(m => m.id === messageId ? { ...m, tools: (m.tools ?? []).map(t => t.id === tool.id ? { ...t, state: state_, ...(result !== undefined ? { result } : {}) } : t) } : m),
  }));
  setState(approved ? 'running' : 'denied');
  api.post('/api/chat/approval', { sessionId, toolId: tool.id, decision: approved ? 'approve' : 'deny' }).catch(error => {
    setState('proposed');
    pushToast('error', 'That answer was not delivered', error instanceof Error ? error.message : undefined);
  });
}

/**
 * Approving a tool in the design build resolves locally: there is no tool
 * runtime behind it yet, so the result is a labelled stand-in. The interaction —
 * what is asked, what is shown, what needs a grant — is the real deliverable.
 */
export function resolveTool(messageId: string, toolId: string, approved: boolean) {
  const sessionId = state.activeSession;
  if (!sessionId) return;
  const tool = state.sessions.find(s => s.id === sessionId)?.messages.find(m => m.id === messageId)?.tools?.find(t => t.id === toolId);
  if (tool?.live) { answerAgentTool(sessionId, messageId, tool, approved); return; }

  patchSession(sessionId, (s) => ({
    ...s,
    messages: s.messages.map((m) =>
      m.id === messageId
        ? {
            ...m,
            tools: (m.tools ?? []).map((t) =>
              t.id === toolId
                ? {
                    ...t,
                    state: approved ? 'done' : 'denied',
                    durationMs: undefined,
                    result: approved ? 'Approval recorded. Nothing was executed.' : 'Denied. Nothing ran.',
                  }
                : t,
            ),
          }
        : m,
    ),
  }));

}

/* ------------------------------------------------------------------- routes */

export function go(route: Route, param?: string) {
  navigate(route, param);
  // Focus mode is a reading mode for the conversation. Leaving it to work on
  // something else leaves the mode, rather than stranding a bare shell.
  if (state.focus && route !== 'chat' && route !== 'settings') setFocus(false);
  set({ paletteOpen: false, navOpen: false });
}

/* ---------------------------------------------------------------- bootstrap */

export function bootOnce() {
  if ((globalThis as { __strataBooted?: boolean }).__strataBooted) return;
  (globalThis as { __strataBooted?: boolean }).__strataBooted = true;
  for (const session of state.sessions) for (const run of session.timeSkips ?? []) scheduleTimeSkip(session.id, run);
  void bootstrap();
  void reconcileAgentTools();
}

/**
 * After a reload, a tool card the browser remembers as waiting or running may belong to a turn that no
 * longer exists. Cards whose approval the host still holds stay answerable; the rest are marked as
 * interrupted so nothing looks like it is still working.
 */
async function reconcileAgentTools() {
  const open = (t: ToolCall) => t.live && (t.state === 'proposed' || t.state === 'running');
  for (const session of state.sessions) {
    if (!session.messages.some(m => m.tools?.some(open))) continue;
    let waiting = new Set<string>();
    try {
      const r = await api.get<{ pending: { callId: string }[] }>(`/api/agent/pending?chatId=${encodeURIComponent(session.id)}`);
      waiting = new Set(r.pending.map(p => p.callId));
    } catch { /* The host is unreachable; leave the cards for the next load. */ continue; }
    patchSession(session.id, sess => ({ ...sess, messages: sess.messages.map(m => m.tools?.some(open) ? { ...m, tools: m.tools.map(t => open(t) && !waiting.has(t.id) ? { ...t, state: 'error' as const, result: 'The page closed before this finished. Check the workspace to see what it did.' } : t) } : m) }));
  }
}


export function setContrast(contrast: 'standard' | 'increased') { set({ contrast }); savePrefs(); }
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.theme === 'system') applyPrefs(); });

export async function loadRuntimeModel(body:unknown){
  const result=await api.post<{state:RuntimeModels;instance:{id:string}}>('/api/runtime/load',body);
  set({runtime:result.state});return result.instance.id;
}
export async function runtimeModelAction(id:string,action:'unload'|'default'){
  set({runtime:await api.post<RuntimeModels>(`/api/runtime/${id}/${action}`)});
}

export async function changeModelDirectory(directory: string, unload: boolean) {
  set(s => ({ busy: { ...s.busy, modelDirectory: true } }));
  try {
    const result = await api.post<{ library: ModelLibrary; imageModels: ImageModelLibrary; runtime: RuntimeModels; hardware: Hardware; setup: SetupState }>('/api/storage/models-directory', { directory, unload });
    set(result);
    return result;
  } finally {
    set(s => ({ busy: { ...s.busy, modelDirectory: false } }));
  }
}

export async function addLocalModel(path: string, name?: string) {
  const result = await api.post<{ library: ModelLibrary; variantId: string }>('/api/local-models', { path, name });
  set({ library: result.library });
  return result.variantId;
}
