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
          // Try loading from Tauri backend
          const data = await invoke<ChatState>('load_chat_state')
          if (data) {
            dispatch({ type: 'LOAD_STATE', payload: data })
            return
          }
        }
        // Fallback to localStorage
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
    const timer = setTimeout(() => {
      try {
        localStorage.setItem('nnagent_chat_state', JSON.stringify(state))
      } catch (error) {
        console.error('[ChatContext] Failed to save chat state to localStorage:', error)
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
      if (isTauri()) {
        await invoke('save_chat_state', { state: JSON.stringify(state) })
      }
      localStorage.setItem('nnagent_chat_state', JSON.stringify(state))
    } catch (error) {
      console.error('[ChatContext] saveState failed:', error)
      throw new Error(`Failed to save chat state: ${error instanceof Error ? error.message : String(error)}`)
    }
  }, [state])

  const loadState = useCallback(async () => {
    try {
      if (isTauri()) {
        const data = await invoke<ChatState>('load_chat_state')
        if (data) {
          dispatch({ type: 'LOAD_STATE', payload: data })
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
