/**
 * ChatContext — React Context for managing chat state across the application.
 *
 * Provides conversation CRUD, message management, folder operations,
 * persistence via localStorage (with Tauri backend fallback),
 * and export/import in multiple formats (JSON, Markdown, PDF, ChatGPT).
 */

import React, { createContext, useContext, useReducer, useCallback, useEffect, ReactNode } from 'react'
import {
  ChatState,
  ChatAction,
  Conversation,
  ConversationFolder,
  Message,
  MessageRole,
  ChatTheme,
  generateId,
  generateTitleFromMessage,
  ExportOptions,
  defaultTheme,
} from '@/types/chat'
import { isTauri, invoke } from '@/tauri-adapter'
import {
  formatAsJson,
  formatFolderAsJson,
  formatAllAsJson,
  formatAsMarkdown,
  formatMultipleAsMarkdown,
  createPrintableHtml,
  generateExportFilename,
} from '@/utils/exportFormatter'
import {
  detectChatGPTFormat,
  convertChatGPTToConversation,
} from '@/utils/chatGptImporter'
import { downloadFile, printHtml } from '@/utils/downloadHelper'

// ─── Initial State ────────────────────────────────────────────────────────────

const initialState: ChatState = {
  conversations: [],
  folders: [],
  activeConversationId: null,
  isLoading: false,
  error: null,
  theme: defaultTheme,
  // PH2-2: Search and filter defaults
  searchQuery: '',
  showArchived: false,
  activeTagFilter: null,
}

// ─── Reducer ──────────────────────────────────────────────────────────────────

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload }

    case 'SET_ERROR':
      return { ...state, error: action.payload }

    case 'SET_ACTIVE_CONVERSATION':
      return { ...state, activeConversationId: action.payload }

    case 'CREATE_CONVERSATION': {
      const exists = state.conversations.find((c) => c.id === action.payload.id)
      if (exists) {
        console.warn(`[ChatContext] CREATE_CONVERSATION: Conversation ${action.payload.id} already exists`)
        return state
      }
      return {
        ...state,
        conversations: [action.payload, ...state.conversations],
        activeConversationId: action.payload.id,
      }
    }

    case 'UPDATE_CONVERSATION': {
      const idx = state.conversations.findIndex((c) => c.id === action.payload.id)
      if (idx === -1) {
        console.warn(`[ChatContext] UPDATE_CONVERSATION: Conversation ${action.payload.id} not found`)
        return state
      }
      const updated = [...state.conversations]
      updated[idx] = action.payload
      return { ...state, conversations: updated }
    }

    case 'DELETE_CONVERSATION': {
      const exists = state.conversations.find((c) => c.id === action.payload)
      if (!exists) {
        console.warn(`[ChatContext] DELETE_CONVERSATION: Conversation ${action.payload} not found`)
        return state
      }
      const filtered = state.conversations.filter((c) => c.id !== action.payload)
      const newActive =
        state.activeConversationId === action.payload
          ? (filtered[0]?.id ?? null)
          : state.activeConversationId
      return {
        ...state,
        conversations: filtered,
        activeConversationId: newActive,
      }
    }

    case 'ADD_MESSAGE': {
      const { conversationId, message } = action.payload
      const convIdx = state.conversations.findIndex((c) => c.id === conversationId)
      if (convIdx === -1) {
        console.error(`[ChatContext] ADD_MESSAGE: Conversation ${conversationId} not found — cannot add message`)
        return { ...state, error: `Cannot add message: conversation ${conversationId} does not exist` }
      }
      const updatedConvs = [...state.conversations]
      const conv = { ...updatedConvs[convIdx] }
      conv.messages = [...conv.messages, message]
      conv.updatedAt = new Date().toISOString()

      // Auto-generate title from first user message if title is generic
      if (conv.messages.length === 1 && message.role === 'user') {
        const firstText = message.contents.find((c) => c.type === 'text')
        if (firstText && firstText.type === 'text') {
          conv.title = generateTitleFromMessage(firstText.content)
        }
      }

      updatedConvs[convIdx] = conv
      return { ...state, conversations: updatedConvs }
    }

    case 'UPDATE_MESSAGE': {
      const { conversationId, message } = action.payload
      const convIdx = state.conversations.findIndex((c) => c.id === conversationId)
      if (convIdx === -1) {
        console.warn(`[ChatContext] UPDATE_MESSAGE: Conversation ${conversationId} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      const conv = { ...updatedConvs[convIdx] }
      const msgIdx = conv.messages.findIndex((m) => m.id === message.id)
      if (msgIdx === -1) {
        console.warn(`[ChatContext] UPDATE_MESSAGE: Message ${message.id} not found in conversation ${conversationId}`)
        return state
      }
      conv.messages = [...conv.messages]
      conv.messages[msgIdx] = message
      conv.updatedAt = new Date().toISOString()
      updatedConvs[convIdx] = conv
      return { ...state, conversations: updatedConvs }
    }

    case 'DELETE_MESSAGE': {
      const { conversationId, messageId } = action.payload
      const convIdx = state.conversations.findIndex((c) => c.id === conversationId)
      if (convIdx === -1) {
        console.warn(`[ChatContext] DELETE_MESSAGE: Conversation ${conversationId} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      const conv = { ...updatedConvs[convIdx] }
      conv.messages = conv.messages.filter((m) => m.id !== messageId)
      conv.updatedAt = new Date().toISOString()
      updatedConvs[convIdx] = conv
      return { ...state, conversations: updatedConvs }
    }

    case 'RENAME_CONVERSATION': {
      const { id, title } = action.payload
      const convIdx = state.conversations.findIndex((c) => c.id === id)
      if (convIdx === -1) {
        console.warn(`[ChatContext] RENAME_CONVERSATION: Conversation ${id} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        title,
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    case 'CREATE_FOLDER': {
      const exists = state.folders.find((f) => f.id === action.payload.id)
      if (exists) {
        console.warn(`[ChatContext] CREATE_FOLDER: Folder ${action.payload.id} already exists`)
        return state
      }
      return { ...state, folders: [...state.folders, action.payload] }
    }

    case 'UPDATE_FOLDER': {
      const fIdx = state.folders.findIndex((f) => f.id === action.payload.id)
      if (fIdx === -1) {
        console.warn(`[ChatContext] UPDATE_FOLDER: Folder ${action.payload.id} not found`)
        return state
      }
      const updatedFolders = [...state.folders]
      updatedFolders[fIdx] = action.payload
      return { ...state, folders: updatedFolders }
    }

    case 'DELETE_FOLDER': {
      const exists = state.folders.find((f) => f.id === action.payload)
      if (!exists) {
        console.warn(`[ChatContext] DELETE_FOLDER: Folder ${action.payload} not found`)
        return state
      }
      // Move conversations in this folder to root
      const updatedConvs = state.conversations.map((c) =>
        c.folderId === action.payload ? { ...c, folderId: null } : c
      )
      return {
        ...state,
        folders: state.folders.filter((f) => f.id !== action.payload),
        conversations: updatedConvs,
      }
    }

    case 'MOVE_CONVERSATION': {
      const { conversationId, folderId } = action.payload
      const convIdx = state.conversations.findIndex((c) => c.id === conversationId)
      if (convIdx === -1) {
        console.warn(`[ChatContext] MOVE_CONVERSATION: Conversation ${conversationId} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        folderId,
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    case 'MOVE_FOLDER': {
      const { folderId, parentId } = action.payload
      const folderIdx = state.folders.findIndex((f) => f.id === folderId)
      if (folderIdx === -1) {
        console.warn(`[ChatContext] MOVE_FOLDER: Folder ${folderId} not found`)
        return state
      }
      // Prevent moving folder into itself or its own descendant (infinite loop)
      const isSelfOrDescendant = (candidateId: string, targetId: string): boolean => {
        if (candidateId === targetId) return true
        const children = state.folders.filter((f) => f.parentId === candidateId)
        return children.some((child) => isSelfOrDescendant(child.id, targetId))
      }
      // Check if target (parentId) is the same as source or a descendant of source
      if (parentId !== null && isSelfOrDescendant(folderId, parentId)) {
        console.warn(`[ChatContext] MOVE_FOLDER: Cannot move folder ${folderId} into itself or descendant ${parentId}`)
        return state
      }
      const updatedFolders = [...state.folders]
      updatedFolders[folderIdx] = {
        ...updatedFolders[folderIdx],
        parentId,
        createdAt: new Date().toISOString(),
      }
      return { ...state, folders: updatedFolders }
    }

    case 'SET_MODEL': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload.conversationId)
      if (convIdx === -1) {
        console.warn(`[ChatContext] SET_MODEL: Conversation ${action.payload.conversationId} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        model: action.payload.model,
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    case 'SET_THEME':
      return { ...state, theme: { ...state.theme, ...action.payload } }

    case 'LOAD_STATE':
      return { ...action.payload }

    // PH2-2: Pinning
    case 'TOGGLE_PIN': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload)
      if (convIdx === -1) {
        console.warn(`[ChatContext] TOGGLE_PIN: Conversation ${action.payload} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        pinned: !updatedConvs[convIdx].pinned,
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    case 'SET_PINNED': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload.id)
      if (convIdx === -1) {
        console.warn(`[ChatContext] SET_PINNED: Conversation ${action.payload.id} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        pinned: action.payload.pinned,
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    // PH2-2: Archiving
    case 'TOGGLE_ARCHIVE': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload)
      if (convIdx === -1) {
        console.warn(`[ChatContext] TOGGLE_ARCHIVE: Conversation ${action.payload} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        archived: !updatedConvs[convIdx].archived,
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    case 'SET_ARCHIVED': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload.id)
      if (convIdx === -1) {
        console.warn(`[ChatContext] SET_ARCHIVED: Conversation ${action.payload.id} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        archived: action.payload.archived,
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    // PH2-2: Tags
    case 'ADD_TAG': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload.id)
      if (convIdx === -1) {
        console.warn(`[ChatContext] ADD_TAG: Conversation ${action.payload.id} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      const currentTags = updatedConvs[convIdx].tags ?? []
      if (!currentTags.includes(action.payload.tag)) {
        updatedConvs[convIdx] = {
          ...updatedConvs[convIdx],
          tags: [...currentTags, action.payload.tag],
          updatedAt: new Date().toISOString(),
        }
      }
      return { ...state, conversations: updatedConvs }
    }

    case 'REMOVE_TAG': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload.id)
      if (convIdx === -1) {
        console.warn(`[ChatContext] REMOVE_TAG: Conversation ${action.payload.id} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      const currentTags = updatedConvs[convIdx].tags ?? []
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        tags: currentTags.filter((t) => t !== action.payload.tag),
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    case 'SET_TAGS': {
      const convIdx = state.conversations.findIndex((c) => c.id === action.payload.id)
      if (convIdx === -1) {
        console.warn(`[ChatContext] SET_TAGS: Conversation ${action.payload.id} not found`)
        return state
      }
      const updatedConvs = [...state.conversations]
      updatedConvs[convIdx] = {
        ...updatedConvs[convIdx],
        tags: [...action.payload.tags],
        updatedAt: new Date().toISOString(),
      }
      return { ...state, conversations: updatedConvs }
    }

    // PH2-2: Search/Filter
    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload }

    case 'TOGGLE_SHOW_ARCHIVED':
      return { ...state, showArchived: action.payload ?? !state.showArchived }

    case 'SET_TAG_FILTER':
      return { ...state, activeTagFilter: action.payload }

    // PH2-2: Bulk operations
    case 'BULK_DELETE': {
      const idsSet = new Set(action.payload)
      const filtered = state.conversations.filter((c) => !idsSet.has(c.id))
      const newActive =
        state.activeConversationId && idsSet.has(state.activeConversationId)
          ? (filtered[0]?.id ?? null)
          : state.activeConversationId
      return {
        ...state,
        conversations: filtered,
        activeConversationId: newActive,
      }
    }

    case 'BULK_ARCHIVE': {
      const idsSet = new Set(action.payload.ids)
      const updatedConvs = state.conversations.map((c) =>
        idsSet.has(c.id)
          ? { ...c, archived: action.payload.archived, updatedAt: new Date().toISOString() }
          : c
      )
      return { ...state, conversations: updatedConvs }
    }

    case 'BULK_PIN': {
      const idsSet = new Set(action.payload.ids)
      const updatedConvs = state.conversations.map((c) =>
        idsSet.has(c.id)
          ? { ...c, pinned: action.payload.pinned, updatedAt: new Date().toISOString() }
          : c
      )
      return { ...state, conversations: updatedConvs }
    }

    case 'BULK_ADD_TAG': {
      const idsSet = new Set(action.payload.ids)
      const updatedConvs = state.conversations.map((c) => {
        if (!idsSet.has(c.id)) return c
        const currentTags = c.tags ?? []
        if (currentTags.includes(action.payload.tag)) return c
        return {
          ...c,
          tags: [...currentTags, action.payload.tag],
          updatedAt: new Date().toISOString(),
        }
      })
      return { ...state, conversations: updatedConvs }
    }

    default:
      return state
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────

interface ChatContextType {
  state: ChatState
  dispatch: React.Dispatch<ChatAction>
  // Convenience methods
  createConversation: (folderId?: string | null, model?: string) => Conversation
  addMessage: (conversationId: string, role: MessageRole, contents: any[], model?: string) => Message
  deleteConversation: (id: string) => void
  renameConversation: (id: string, title: string) => void
  setActiveConversation: (id: string | null) => void
  createFolder: (name: string, parentId?: string | null) => ConversationFolder
  // Model switching
  setModel: (conversationId: string, model: string) => void
  // Theming
  setTheme: (theme: Partial<ChatTheme>) => void
  // Persistence
  saveState: () => Promise<void>
  loadState: () => Promise<void>
  // Export
  exportConversation: (id: string, options?: ExportOptions) => Promise<void>
  exportFolder: (folderId: string, options?: ExportOptions) => Promise<void>
  exportAll: (options?: ExportOptions) => Promise<void>
  // Import
  importConversation: (json: string) => Promise<Conversation | null>
  importFromFile: (file: File) => Promise<number>
  // PH2-2: Pinning
  togglePin: (id: string) => void
  // PH2-2: Archiving
  toggleArchive: (id: string) => void
  // PH2-2: Tags
  addTag: (id: string, tag: string) => void
  removeTag: (id: string, tag: string) => void
  setTags: (id: string, tags: string[]) => void
  // PH2-2: Search/Filter
  setSearchQuery: (query: string) => void
  toggleShowArchived: (show?: boolean) => void
  setTagFilter: (tag: string | null) => void
  // PH2-2: Bulk operations
  bulkDelete: (ids: string[]) => void
  bulkArchive: (ids: string[], archived: boolean) => void
  bulkPin: (ids: string[], pinned: boolean) => void
  bulkAddTag: (ids: string[], tag: string) => void
  // PH2-2: Get all unique tags across conversations
  getAllTags: () => string[]
  // PH2-2: Get filtered conversations based on search/tag/archive settings
  getFilteredConversations: () => Conversation[]
}

const ChatContext = createContext<ChatContextType | null>(null)

// ─── Provider ─────────────────────────────────────────────────────────────────

interface ChatProviderProps {
  children: ReactNode
}

export function ChatProvider({ children }: ChatProviderProps) {
  const [state, dispatch] = useReducer(chatReducer, initialState)

  // ─── Load state on mount ────────────────────────────────────────────────

  useEffect(() => {
    const load = async () => {
      try {
        if (isTauri()) {
          // Try loading from Tauri backend (file-based storage)
          const fileStateJson = await invoke<string>('load_chat_state')
          if (fileStateJson) {
            const parsed = JSON.parse(fileStateJson)
            dispatch({ type: 'LOAD_STATE', payload: parsed })
            return
          }
          // No file state — check if we need to migrate from localStorage
          const migrated = await invoke<boolean>('is_migration_complete')
          if (!migrated) {
            const saved = localStorage.getItem('nnagent_chat_state')
            if (saved) {
              const parsed = JSON.parse(saved)
              dispatch({ type: 'LOAD_STATE', payload: parsed })
              // Migrate: save to file storage and clear localStorage
              try {
                await invoke('save_chat_state', { stateJson: saved })
                await invoke('mark_migration_complete')
                localStorage.removeItem('nnagent_chat_state')
                console.info('[ChatContext] Successfully migrated chat state from localStorage to file storage')
              } catch (migrationError) {
                console.error('[ChatContext] Migration to file storage failed, keeping localStorage:', migrationError)
              }
            }
            return
          }
        }
        // Fallback to localStorage (non-Tauri or migration already done)
        const saved = localStorage.getItem('nnagent_chat_state')
        if (saved) {
          const parsed = JSON.parse(saved)
          dispatch({ type: 'LOAD_STATE', payload: parsed })
        }
      } catch (error) {
        console.error('[ChatContext] Failed to load chat state:', error)
        dispatch({
          type: 'SET_ERROR',
          payload: `Failed to load chat state: ${error instanceof Error ? error.message : String(error)}`,
        })
      }
    }
    load()
  }, [])

  // ─── Save state on change ───────────────────────────────────────────────

  useEffect(() => {
    // Debounce saves
    const timer = setTimeout(async () => {
      try {
        const stateJson = JSON.stringify(state)
        if (isTauri()) {
          // Primary: save to file storage via Tauri backend
          await invoke('save_chat_state', { stateJson })
        }
        // Fallback: always keep localStorage in sync
        localStorage.setItem('nnagent_chat_state', stateJson)
      } catch (error) {
        console.error('[ChatContext] Failed to save chat state:', error)
      }
    }, 500)
    return () => clearTimeout(timer)
  }, [state])

  // ─── Convenience Methods ────────────────────────────────────────────────

  const createConversation = useCallback(
    (folderId?: string | null, model?: string): Conversation => {
      const now = new Date().toISOString()
      const conversation: Conversation = {
        id: generateId(),
        title: 'New Conversation',
        messages: [],
        folderId: folderId ?? null,
        createdAt: now,
        updatedAt: now,
        model,
      }
      dispatch({ type: 'CREATE_CONVERSATION', payload: conversation })
      return conversation
    },
    [dispatch]
  )

  const addMessage = useCallback(
    (conversationId: string, role: MessageRole, contents: any[], model?: string): Message => {
      const message: Message = {
        id: generateId(),
        role,
        contents,
        timestamp: new Date().toISOString(),
        model,
      }
      dispatch({ type: 'ADD_MESSAGE', payload: { conversationId, message } })
      return message
    },
    [dispatch]
  )

  const deleteConversation = useCallback(
    (id: string) => {
      dispatch({ type: 'DELETE_CONVERSATION', payload: id })
    },
    [dispatch]
  )

  const renameConversation = useCallback(
    (id: string, title: string) => {
      dispatch({ type: 'RENAME_CONVERSATION', payload: { id, title } })
    },
    [dispatch]
  )

  const setActiveConversation = useCallback(
    (id: string | null) => {
      dispatch({ type: 'SET_ACTIVE_CONVERSATION', payload: id })
    },
    [dispatch]
  )

  const createFolder = useCallback(
    (name: string, parentId?: string | null): ConversationFolder => {
      const folder: ConversationFolder = {
        id: generateId(),
        name,
        parentId: parentId ?? null,
        order: Date.now(),
        createdAt: new Date().toISOString(),
      }
      dispatch({ type: 'CREATE_FOLDER', payload: folder })
      return folder
    },
    [dispatch]
  )

  // ─── Model Switching ─────────────────────────────────────────────────────

  const setModel = useCallback(
    (conversationId: string, model: string) => {
      dispatch({ type: 'SET_MODEL', payload: { conversationId, model } })
    },
    [dispatch]
  )

  // ─── Theming ────────────────────────────────────────────────────────────

  const setTheme = useCallback(
    (theme: Partial<ChatTheme>) => {
      dispatch({ type: 'SET_THEME', payload: theme })
    },
    [dispatch]
  )

  // ─── Persistence ────────────────────────────────────────────────────────

  const saveState = useCallback(async () => {
    try {
      const stateJson = JSON.stringify(state)
      if (isTauri()) {
        await invoke('save_chat_state', { stateJson })
      }
      localStorage.setItem('nnagent_chat_state', stateJson)
    } catch (error) {
      console.error('[ChatContext] saveState failed:', error)
      throw new Error(`Failed to save chat state: ${error instanceof Error ? error.message : String(error)}`)
    }
  }, [state])

  const loadState = useCallback(async () => {
    try {
      if (isTauri()) {
        const fileStateJson = await invoke<string>('load_chat_state')
        if (fileStateJson) {
          const parsed = JSON.parse(fileStateJson)
          dispatch({ type: 'LOAD_STATE', payload: parsed })
          return
        }
      }
      const saved = localStorage.getItem('nnagent_chat_state')
      if (saved) {
        const parsed = JSON.parse(saved)
        dispatch({ type: 'LOAD_STATE', payload: parsed })
      }
    } catch (error) {
      console.error('[ChatContext] loadState failed:', error)
      throw new Error(`Failed to load chat state: ${error instanceof Error ? error.message : String(error)}`)
    }
  }, [dispatch])

  // ─── PH2-2: Pinning ─────────────────────────────────────────────────────

  const togglePin = useCallback(
    (id: string) => {
      dispatch({ type: 'TOGGLE_PIN', payload: id })
    },
    [dispatch]
  )

  // ─── PH2-2: Archiving ───────────────────────────────────────────────────

  const toggleArchive = useCallback(
    (id: string) => {
      dispatch({ type: 'TOGGLE_ARCHIVE', payload: id })
    },
    [dispatch]
  )

  // ─── PH2-2: Tags ────────────────────────────────────────────────────────

  const addTag = useCallback(
    (id: string, tag: string) => {
      dispatch({ type: 'ADD_TAG', payload: { id, tag } })
    },
    [dispatch]
  )

  const removeTag = useCallback(
    (id: string, tag: string) => {
      dispatch({ type: 'REMOVE_TAG', payload: { id, tag } })
    },
    [dispatch]
  )

  const setTags = useCallback(
    (id: string, tags: string[]) => {
      dispatch({ type: 'SET_TAGS', payload: { id, tags } })
    },
    [dispatch]
  )

  // ─── PH2-2: Search/Filter ───────────────────────────────────────────────

  const setSearchQuery = useCallback(
    (query: string) => {
      dispatch({ type: 'SET_SEARCH_QUERY', payload: query })
    },
    [dispatch]
  )

  const toggleShowArchived = useCallback(
    (show?: boolean) => {
      dispatch({ type: 'TOGGLE_SHOW_ARCHIVED', payload: show })
    },
    [dispatch]
  )

  const setTagFilter = useCallback(
    (tag: string | null) => {
      dispatch({ type: 'SET_TAG_FILTER', payload: tag })
    },
    [dispatch]
  )

  // ─── PH2-2: Bulk operations ─────────────────────────────────────────────

  const bulkDelete = useCallback(
    (ids: string[]) => {
      if (ids.length === 0) {
        console.warn('[ChatContext] bulkDelete: No IDs provided')
        return
      }
      dispatch({ type: 'BULK_DELETE', payload: ids })
    },
    [dispatch]
  )

  const bulkArchive = useCallback(
    (ids: string[], archived: boolean) => {
      if (ids.length === 0) {
        console.warn('[ChatContext] bulkArchive: No IDs provided')
        return
      }
      dispatch({ type: 'BULK_ARCHIVE', payload: { ids, archived } })
    },
    [dispatch]
  )

  const bulkPin = useCallback(
    (ids: string[], pinned: boolean) => {
      if (ids.length === 0) {
        console.warn('[ChatContext] bulkPin: No IDs provided')
        return
      }
      dispatch({ type: 'BULK_PIN', payload: { ids, pinned } })
    },
    [dispatch]
  )

  const bulkAddTag = useCallback(
    (ids: string[], tag: string) => {
      if (ids.length === 0) {
        console.warn('[ChatContext] bulkAddTag: No IDs provided')
        return
      }
      dispatch({ type: 'BULK_ADD_TAG', payload: { ids, tag } })
    },
    [dispatch]
  )

  // ─── PH2-2: Get all unique tags ─────────────────────────────────────────

  const getAllTags = useCallback(
    (): string[] => {
      const tagSet = new Set<string>()
      for (const conv of state.conversations) {
        if (conv.tags) {
          for (const tag of conv.tags) {
            tagSet.add(tag)
          }
        }
      }
      return Array.from(tagSet).sort()
    },
    [state.conversations]
  )

  // ─── PH2-2: Get filtered conversations ──────────────────────────────────

  const getFilteredConversations = useCallback(
    (): Conversation[] => {
      let result = state.conversations

      // Filter out archived unless showArchived is true
      if (!state.showArchived) {
        result = result.filter((c) => !c.archived)
      }

      // Apply tag filter
      if (state.activeTagFilter) {
        result = result.filter((c) => c.tags?.includes(state.activeTagFilter!))
      }

      // Apply search query - search in title and message content
      if (state.searchQuery.trim()) {
        const query = state.searchQuery.trim().toLowerCase()
        result = result.filter((c) => {
          // Search in title
          if (c.title.toLowerCase().includes(query)) return true
          // Search in message content
          return c.messages.some((m) =>
            m.contents.some((content) =>
              content.type === 'text' && content.content.toLowerCase().includes(query)
            )
          )
        })
      }

      // Sort: pinned first, then by updatedAt descending
      result.sort((a, b) => {
        // Pinned conversations always at top
        if (a.pinned && !b.pinned) return -1
        if (!a.pinned && b.pinned) return 1
        // Both pinned or both unpinned: sort by updatedAt
        return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
      })

      return result
    },
    [state.conversations, state.showArchived, state.activeTagFilter, state.searchQuery]
  )

  // ─── Export ─────────────────────────────────────────────────────────────

  const exportConversation = useCallback(
    async (id: string, options?: ExportOptions): Promise<void> => {
      const conversation = state.conversations.find((c) => c.id === id)
      if (!conversation) {
        console.error(`[ChatContext] exportConversation: Conversation ${id} not found`)
        dispatch({ type: 'SET_ERROR', payload: `Export failed: conversation ${id} not found` })
        return
      }

      try {
        const format = options?.format || 'json'
        let content: string
        let mimeType: string
        let filename: string

        if (format === 'json') {
          content = formatAsJson(conversation)
          mimeType = 'application/json'
          filename = generateExportFilename(conversation.title, 'json')
        } else if (format === 'markdown') {
          content = formatAsMarkdown(conversation)
          mimeType = 'text/markdown'
          filename = generateExportFilename(conversation.title, 'markdown')
        } else if (format === 'pdf') {
          content = createPrintableHtml(conversation)
          printHtml(content)
          return // PDF is handled by print dialog, no download
        } else {
          console.error(`[ChatContext] exportConversation(${id}): Unknown format ${format}`)
          return
        }

        downloadFile(content, filename, mimeType)
      } catch (error) {
        console.error(`[ChatContext] exportConversation(${id}) failed:`, error)
        dispatch({
          type: 'SET_ERROR',
          payload: `Export failed: ${error instanceof Error ? error.message : String(error)}`,
        })
      }
    },
    [state.conversations, dispatch]
  )

  const exportFolder = useCallback(
    async (folderId: string, options?: ExportOptions): Promise<void> => {
      const folder = state.folders.find((f) => f.id === folderId)
      if (!folder) {
        console.error(`[ChatContext] exportFolder: Folder ${folderId} not found`)
        dispatch({ type: 'SET_ERROR', payload: `Export failed: folder ${folderId} not found` })
        return
      }

      const conversations = state.conversations.filter((c) => c.folderId === folderId)
      if (conversations.length === 0) {
        console.warn(`[ChatContext] exportFolder: Folder ${folderId} has no conversations`)
        return
      }

      try {
        const format = options?.format || 'json'
        let content: string
        let mimeType: string
        let filename: string

        if (format === 'json') {
          content = formatFolderAsJson(conversations, folder.name)
          mimeType = 'application/json'
          filename = generateExportFilename(folder.name, 'json')
        } else if (format === 'markdown') {
          content = formatMultipleAsMarkdown(conversations, folder.name)
          mimeType = 'text/markdown'
          filename = generateExportFilename(folder.name, 'markdown')
        } else {
          console.error(`[ChatContext] exportFolder(${folderId}): Unknown format ${format}`)
          return
        }

        downloadFile(content, filename, mimeType)
      } catch (error) {
        console.error(`[ChatContext] exportFolder(${folderId}) failed:`, error)
        dispatch({
          type: 'SET_ERROR',
          payload: `Export failed: ${error instanceof Error ? error.message : String(error)}`,
        })
      }
    },
    [state.conversations, state.folders, dispatch]
  )

  const exportAll = useCallback(
    async (options?: ExportOptions): Promise<void> => {
      try {
        const format = options?.format || 'json'
        let content: string
        let mimeType: string
        let filename: string

        if (format === 'json') {
          content = formatAllAsJson(state)
          mimeType = 'application/json'
          filename = generateExportFilename('all_conversations', 'json')
        } else if (format === 'markdown') {
          content = formatMultipleAsMarkdown(state.conversations, 'All Conversations')
          mimeType = 'text/markdown'
          filename = generateExportFilename('all_conversations', 'markdown')
        } else {
          console.error(`[ChatContext] exportAll: Unknown format ${format}`)
          return
        }

        downloadFile(content, filename, mimeType)
      } catch (error) {
        console.error('[ChatContext] exportAll failed:', error)
        dispatch({
          type: 'SET_ERROR',
          payload: `Export failed: ${error instanceof Error ? error.message : String(error)}`,
        })
      }
    },
    [state, dispatch]
  )

  // ─── Import ─────────────────────────────────────────────────────────────

  const importConversation = useCallback(
    async (json: string): Promise<Conversation | null> => {
      try {
        // Auto-detect format
        if (detectChatGPTFormat(json)) {
          const converted = convertChatGPTToConversation(json)
          if (converted) {
            dispatch({ type: 'CREATE_CONVERSATION', payload: converted })
            return converted
          }
          throw new Error('Failed to convert ChatGPT format')
        }

        // NNSpire format
        const data = JSON.parse(json)

        // Handle full state import
        if (data.state) {
          const importedCount = data.state.conversations?.length || 0
          if (importedCount > 0) {
            for (const convData of data.state.conversations) {
              const imported: Conversation = {
                id: generateId(),
                title: convData.title || 'Imported Conversation',
                messages: convData.messages,
                folderId: null,
                createdAt: convData.createdAt || new Date().toISOString(),
                updatedAt: new Date().toISOString(),
                model: convData.model,
                systemPrompt: convData.systemPrompt,
              }
              dispatch({ type: 'CREATE_CONVERSATION', payload: imported })
            }
          }
          return null // Bulk import returns null
        }

        // Handle folder bundle import
        if (data.conversations && Array.isArray(data.conversations)) {
          for (const convData of data.conversations) {
            const imported: Conversation = {
              id: generateId(),
              title: convData.title || 'Imported Conversation',
              messages: convData.messages,
              folderId: null,
              createdAt: convData.createdAt || new Date().toISOString(),
              updatedAt: new Date().toISOString(),
              model: convData.model,
              systemPrompt: convData.systemPrompt,
            }
            dispatch({ type: 'CREATE_CONVERSATION', payload: imported })
          }
          return null // Bulk import returns null
        }

        // Handle single conversation import
        if (!data.conversation) {
          throw new Error('Invalid export format: missing "conversation" field')
        }
        const { conversation } = data
        if (!conversation.messages) {
          throw new Error('Invalid export format: conversation missing required field "messages"')
        }

        const imported: Conversation = {
          id: generateId(),
          title: conversation.title || 'Imported Conversation',
          messages: conversation.messages,
          folderId: null,
          createdAt: conversation.createdAt || new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          model: conversation.model,
          systemPrompt: conversation.systemPrompt,
        }

        dispatch({ type: 'CREATE_CONVERSATION', payload: imported })
        return imported
      } catch (error) {
        console.error('[ChatContext] importConversation failed:', error)
        dispatch({
          type: 'SET_ERROR',
          payload: `Import failed: ${error instanceof Error ? error.message : String(error)}`,
        })
        return null
      }
    },
    [dispatch]
  )

  const importFromFile = useCallback(
    async (file: File): Promise<number> => {
      return new Promise((resolve) => {
        const reader = new FileReader()
        reader.onload = async (event) => {
          const text = event.target?.result as string
          if (!text) {
            console.error('[ChatContext] importFromFile: Failed to read file content')
            dispatch({ type: 'SET_ERROR', payload: `Import failed: could not read file ${file.name}` })
            resolve(0)
            return
          }

          let importedCount = 0

          try {
            // Try parsing as JSON
            const data = JSON.parse(text)

            if (detectChatGPTFormat(text)) {
              const converted = convertChatGPTToConversation(text)
              if (converted) {
                dispatch({ type: 'CREATE_CONVERSATION', payload: converted })
                importedCount = 1
              }
            } else if (data.state) {
              // Full state import
              importedCount = data.state.conversations?.length || 0
              for (const convData of data.state.conversations) {
                const imported: Conversation = {
                  id: generateId(),
                  title: convData.title || 'Imported Conversation',
                  messages: convData.messages,
                  folderId: null,
                  createdAt: convData.createdAt || new Date().toISOString(),
                  updatedAt: new Date().toISOString(),
                  model: convData.model,
                  systemPrompt: convData.systemPrompt,
                }
                dispatch({ type: 'CREATE_CONVERSATION', payload: imported })
              }
            } else if (data.conversations && Array.isArray(data.conversations)) {
              // Folder bundle
              importedCount = data.conversations.length
              for (const convData of data.conversations) {
                const imported: Conversation = {
                  id: generateId(),
                  title: convData.title || 'Imported Conversation',
                  messages: convData.messages,
                  folderId: null,
                  createdAt: convData.createdAt || new Date().toISOString(),
                  updatedAt: new Date().toISOString(),
                  model: convData.model,
                  systemPrompt: convData.systemPrompt,
                }
                dispatch({ type: 'CREATE_CONVERSATION', payload: imported })
              }
            } else if (data.conversation) {
              // Single conversation
              const imported: Conversation = {
                id: generateId(),
                title: data.conversation.title || 'Imported Conversation',
                messages: data.conversation.messages,
                folderId: null,
                createdAt: data.conversation.createdAt || new Date().toISOString(),
                updatedAt: new Date().toISOString(),
                model: data.conversation.model,
                systemPrompt: data.conversation.systemPrompt,
              }
              dispatch({ type: 'CREATE_CONVERSATION', payload: imported })
              importedCount = 1
            } else {
              throw new Error('Unrecognized export format')
            }
          } catch (error) {
            console.error(`[ChatContext] importFromFile(${file.name}) failed:`, error)
            dispatch({
              type: 'SET_ERROR',
              payload: `Import failed for ${file.name}: ${error instanceof Error ? error.message : String(error)}`,
            })
          }

          resolve(importedCount)
        }

        reader.onerror = () => {
          console.error(`[ChatContext] importFromFile(${file.name}): FileReader error`)
          dispatch({ type: 'SET_ERROR', payload: `Import failed: could not read file ${file.name}` })
          resolve(0)
        }

        reader.readAsText(file)
      })
    },
    [dispatch]
  )

  // ─── Context Value ──────────────────────────────────────────────────────

  const contextValue: ChatContextType = {
    state,
    dispatch,
    createConversation,
    addMessage,
    deleteConversation,
    renameConversation,
    setActiveConversation,
    createFolder,
    setModel,
    setTheme,
    saveState,
    loadState,
    exportConversation,
    exportFolder,
    exportAll,
    importConversation,
    importFromFile,
    // PH2-2: Pinning
    togglePin,
    // PH2-2: Archiving
    toggleArchive,
    // PH2-2: Tags
    addTag,
    removeTag,
    setTags,
    // PH2-2: Search/Filter
    setSearchQuery,
    toggleShowArchived,
    setTagFilter,
    // PH2-2: Bulk operations
    bulkDelete,
    bulkArchive,
    bulkPin,
    bulkAddTag,
    // PH2-2: Queries
    getAllTags,
    getFilteredConversations,
  }

  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useChat(): ChatContextType {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider')
  }
  return context
}
