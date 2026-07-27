/**
 * Chat Data Models for NNSpire Agent
 *
 * Defines the core data structures for messages, conversations, and folders.
 * Used by ChatContext for state management and persistence layer.
 */

// ─── Message Roles ────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant' | 'system' | 'tool'

// ─── Message Content Types ────────────────────────────────────────────────────

export interface TextContent {
  type: 'text'
  content: string
}

export interface ThinkingContent {
  type: 'thinking'
  content: string
  collapsed?: boolean
}

export interface CodeContent {
  type: 'code'
  language: string
  content: string
}

export interface McpCallContent {
  type: 'mcp_call'
  toolName: string
  arguments: Record<string, unknown>
  result?: string
  success: boolean
}

export interface ErrorContent {
  type: 'error'
  message: string
  details?: string
  code?: string
}

export type MessageContent = TextContent | ThinkingContent | CodeContent | McpCallContent | ErrorContent

// ─── Message ──────────────────────────────────────────────────────────────────

export interface Message {
  id: string
  role: MessageRole
  contents: MessageContent[]
  timestamp: string // ISO 8601
  model?: string // Model used for assistant messages
  rawJson?: string // Optional raw JSON for debugging
}

// ─── Conversation ─────────────────────────────────────────────────────────────

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  folderId: string | null // null = root level
  createdAt: string // ISO 8601
  updatedAt: string // ISO 8601
  model?: string // Default model for this conversation
  systemPrompt?: string
  // PH2-2: Enhanced conversation management
  pinned?: boolean // If true, conversation stays at top of list
  archived?: boolean // If true, conversation is hidden from main list
  tags?: string[] // Tags/labels for categorization
}

// ─── Folder ───────────────────────────────────────────────────────────────────

export interface ConversationFolder {
  id: string
  name: string
  parentId: string | null // null = root level
  order: number // for sorting within parent
  createdAt: string // ISO 8601
}

// ─── Chat State ───────────────────────────────────────────────────────────────

export interface ChatTheme {
  userIcon: string   // emoji for user messages
  aiIcon: string    // emoji for assistant messages
}

export const defaultTheme: ChatTheme = {
  userIcon: '👤',
  aiIcon: '🤖',
}

export interface ChatState {
  conversations: Conversation[]
  folders: ConversationFolder[]
  activeConversationId: string | null
  isLoading: boolean
  error: string | null
  theme: ChatTheme
  // PH2-2: Search and filter
  searchQuery: string
  showArchived: boolean
  activeTagFilter: string | null
}

// ─── Chat Actions ─────────────────────────────────────────────────────────────

export type ChatAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_ACTIVE_CONVERSATION'; payload: string | null }
  | { type: 'CREATE_CONVERSATION'; payload: Conversation }
  | { type: 'UPDATE_CONVERSATION'; payload: Conversation }
  | { type: 'DELETE_CONVERSATION'; payload: string }
  | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: Message } }
  | { type: 'UPDATE_MESSAGE'; payload: { conversationId: string; message: Message } }
  | { type: 'DELETE_MESSAGE'; payload: { conversationId: string; messageId: string } }
  | { type: 'RENAME_CONVERSATION'; payload: { id: string; title: string } }
  | { type: 'CREATE_FOLDER'; payload: ConversationFolder }
  | { type: 'UPDATE_FOLDER'; payload: ConversationFolder }
  | { type: 'DELETE_FOLDER'; payload: string }
  | { type: 'MOVE_CONVERSATION'; payload: { conversationId: string; folderId: string | null } }
  | { type: 'MOVE_FOLDER'; payload: { folderId: string; parentId: string | null } }
  | { type: 'SET_MODEL'; payload: { conversationId: string; model: string } }
  | { type: 'SET_THEME'; payload: Partial<ChatTheme> }
  | { type: 'LOAD_STATE'; payload: ChatState }
  // PH2-2: Pinning
  | { type: 'TOGGLE_PIN'; payload: string }
  | { type: 'SET_PINNED'; payload: { id: string; pinned: boolean } }
  // PH2-2: Archiving
  | { type: 'TOGGLE_ARCHIVE'; payload: string }
  | { type: 'SET_ARCHIVED'; payload: { id: string; archived: boolean } }
  // PH2-2: Tags
  | { type: 'ADD_TAG'; payload: { id: string; tag: string } }
  | { type: 'REMOVE_TAG'; payload: { id: string; tag: string } }
  | { type: 'SET_TAGS'; payload: { id: string; tags: string[] } }
  // PH2-2: Search/Filter
  | { type: 'SET_SEARCH_QUERY'; payload: string }
  | { type: 'TOGGLE_SHOW_ARCHIVED'; payload?: boolean }
  | { type: 'SET_TAG_FILTER'; payload: string | null }
  // PH2-2: Bulk operations
  | { type: 'BULK_DELETE'; payload: string[] }
  | { type: 'BULK_ARCHIVE'; payload: { ids: string[]; archived: boolean } }
  | { type: 'BULK_PIN'; payload: { ids: string[]; pinned: boolean } }
  | { type: 'BULK_ADD_TAG'; payload: { ids: string[]; tag: string } }

// ─── Export/Import Formats ────────────────────────────────────────────────────

export interface ExportOptions {
  format: 'json' | 'markdown' | 'pdf'
  includeThinking?: boolean
  includeRawJson?: boolean
}

export interface ChatGPTMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: { content_type: string; parts?: string[]; text?: string }
  timestamp: string
}

export interface ChatGPTConversation {
  id: string
  title: string
  mapping: Record<string, { id: string; message?: { metadata: Record<string, unknown>; content: { parts: string[] } } }>
  creation_time: number
  update_time: number
}

// ─── Utility Functions ────────────────────────────────────────────────────────

/**
 * Generate a unique ID for messages/conversations.
 * Uses timestamp + random suffix to avoid collisions.
 */
export function generateId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

/**
 * Generate a conversation title from the first user message.
 * Truncates to 50 characters if needed.
 */
export function generateTitleFromMessage(content: string): string {
  const trimmed = content.trim()
  if (trimmed.length <= 50) {
    return trimmed
  }
  return trimmed.substring(0, 47) + '...'
}

/**
 * Format timestamp for display.
 */
export function formatTimestamp(isoString: string): string {
  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`

  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

/**
 * Format timestamp with time for display.
 */
export function formatTimestampWithTime(isoString: string): string {
  const date = new Date(isoString)
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
