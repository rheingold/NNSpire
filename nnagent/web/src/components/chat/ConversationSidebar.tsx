/**
 * ConversationSidebar — Full-featured conversation list with folder organization.
 *
 * Features:
 * - Multi-level folder nesting with expand/collapse
 * - Rename conversations and folders (inline editing)
 * - Delete conversations and folders with confirmation
 * - Drag-and-drop reorganization (custom pointer-based for WebView2 compat)
 * - Keyboard shortcuts (Ctrl+C/X/V for copy/cut/paste)
 * - Context menu for actions
 * - Auto-generated titles from first prompt
 */

import React, { useState, useCallback, useRef, useEffect } from 'react'
import { useChat } from '@/context/ChatContext'
import { Conversation, ConversationFolder, formatTimestamp } from '@/types/chat'

// ─── Types ────────────────────────────────────────────────────────────────────

interface ContextMenuState {
  visible: boolean
  x: number
  y: number
  targetType: 'conversation' | 'folder' | null
  targetId: string | null
}

/**
 * Internal drag state for our custom pointer-based drag system.
 * Uses mutable ref to avoid stale closures during pointermove.
 */
interface DraggingInfo {
  type: 'conversation' | 'folder'
  id: string
  startFolderId: string | null  // original folder of dragged conv, or parent of dragged folder
  targetFolderId?: string | null  // current hover target for visual feedback
}

// ─── Folder Tree Node ─────────────────────────────────────────────────────────

/**
 * Props for FolderTreeNode. The onDragStart / onDrop callbacks allow the
 * custom drag system to hook into folder headers and content areas.
 */
interface FolderTreeNodeProps {
  folder: ConversationFolder
  allFolders: ConversationFolder[]
  conversations: Conversation[]
  activeConversationId: string | null
  selectedFolderId: string | null
  expandedFolders: Set<string>
  onToggleFolder: (folderId: string) => void
  onSelectFolder: (folderId: string | null) => void
  onSelectConversation: (id: string) => void
  onRenameFolder: (folderId: string, newName: string) => void
  onDeleteFolder: (folderId: string) => void
  onRenameConversation: (convId: string, newTitle: string) => void
  onDeleteConversation: (convId: string) => void
  contextMenu: ContextMenuState
  onContextMenu: (e: React.MouseEvent, type: 'conversation' | 'folder', id: string) => void
  onCloseContextMenu: () => void
  /** Called when mouse down on folder header starts a potential drag */
  onDragStart: (e: React.MouseEvent, type: 'conversation' | 'folder', id: string) => void
  /** The current drag state (used to highlight drop targets) */
  dragState: DraggingInfo | null
  /** Ref to check if drag was activated (prevents click after drag) */
  dragActivatedRef: React.MutableRefObject<boolean>
}

const FolderTreeNode: React.FC<FolderTreeNodeProps> = ({
  folder,
  allFolders,
  conversations,
  activeConversationId,
  selectedFolderId,
  expandedFolders,
  onToggleFolder,
  onSelectFolder,
  onSelectConversation,
  onRenameFolder,
  onDeleteFolder,
  onRenameConversation,
  onDeleteConversation,
  contextMenu,
  onContextMenu,
  onCloseContextMenu,
  onDragStart,
  dragState,
  dragActivatedRef,
}) => {
  const folderConversations = conversations.filter((c) => c.folderId === folder.id)
  const isExpanded = expandedFolders.has(folder.id)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(folder.name)
  const inputRef = useRef<HTMLInputElement>(null)

  // Focus edit input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  const handleToggle = (e: React.MouseEvent) => {
    // Skip if drag was activated to prevent toggle after drag
    if (dragActivatedRef.current) return
    e.stopPropagation()
    onToggleFolder(folder.id)
    // Also select this folder so new conversations go inside
    onSelectFolder(folder.id)
  }

  const handleSaveRename = () => {
    const trimmed = editName.trim()
    if (trimmed && trimmed !== folder.name) {
      onRenameFolder(folder.id, trimmed)
    } else {
      setEditName(folder.name)
    }
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveRename()
    } else if (e.key === 'Escape') {
      setEditName(folder.name)
      setIsEditing(false)
    }
  }

  return (
    <div className="folder-tree-node">
      {/* Folder Header — draggable via onMouseDown, identical to conversation items */}
      <div
        className={`folder-header ${isExpanded ? 'expanded' : ''}`}
        onClick={handleToggle}
        onContextMenu={(e) => onContextMenu(e, 'folder', folder.id)}
        onMouseDown={(e) => {
          // Only start drag if not clicking toggle arrow or input
          if ((e.target as HTMLElement).closest('.folder-toggle') || (e.target as HTMLElement).closest('.folder-edit-input')) {
            return
          }
          onDragStart(e, 'folder', folder.id)
        }}
        style={{
          pointerEvents: 'auto',
          cursor: dragState ? 'grab' : 'default',
        }}
      >
        <span className="folder-toggle" aria-label={isExpanded ? 'Collapse folder' : 'Expand folder'}>
          {isExpanded ? '▼' : '▶'}
        </span>
        <span className="folder-icon">📁</span>
        {isEditing ? (
          <input
            ref={inputRef}
            className="folder-edit-input"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={handleSaveRename}
            onKeyDown={handleKeyDown}
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span
            className="folder-name"
            onDoubleClick={(e) => {
              e.stopPropagation()
              setIsEditing(true)
            }}
          >
            {folder.name}
          </span>
        )}
        <span className="folder-count">{folderConversations.length}</span>
      </div>

      {/* Child Content — drop target area */}
      {isExpanded && (
        <div
          className={`folder-content${dragState && dragState.type === 'folder' && dragState.id === folder.id ? ' drop-active' : ''}`}
          data-drop-target={folder.id}
        >
          <UnifiedTreeNode
            parentId={folder.id}
            allFolders={allFolders}
            conversations={conversations}
            activeConversationId={activeConversationId}
            selectedFolderId={selectedFolderId}
            expandedFolders={expandedFolders}
            onToggleFolder={onToggleFolder}
            onSelectFolder={onSelectFolder}
            onSelectConversation={onSelectConversation}
            onRenameFolder={onRenameFolder}
            onDeleteFolder={onDeleteFolder}
            onRenameConversation={onRenameConversation}
            onDeleteConversation={onDeleteConversation}
            contextMenu={contextMenu}
            onContextMenu={onContextMenu}
            onCloseContextMenu={onCloseContextMenu}
            onDragStart={onDragStart}
            dragState={dragState}
            dragActivatedRef={dragActivatedRef}
          />
        </div>
      )}
    </div>
  )
}

// ─── Conversation Item ────────────────────────────────────────────────────────

interface ConversationItemProps {
  conversation: Conversation
  isActive: boolean
  onSelect: (id: string) => void
  onRename: (convId: string, newTitle: string) => void
  onContextMenu: (e: React.MouseEvent, type: 'conversation', id: string) => void
  /** Called when pointer down starts a potential drag */
  onDragStart: (e: React.MouseEvent, type: 'conversation', id: string) => void
  /** Ref to check if drag was activated (prevents click after drag) */
  dragActivatedRef: React.MutableRefObject<boolean>
}

const ConversationItem: React.FC<ConversationItemProps> = ({
  conversation,
  isActive,
  onSelect,
  onRename,
  onContextMenu,
  onDragStart,
  dragActivatedRef,
}) => {
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState(conversation.title)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  useEffect(() => {
    setEditTitle(conversation.title)
  }, [conversation.title])

  const handleSaveRename = () => {
    const trimmed = editTitle.trim()
    if (trimmed && trimmed !== conversation.title) {
      onRename(conversation.id, trimmed)
    } else {
      setEditTitle(conversation.title)
    }
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveRename()
    } else if (e.key === 'Escape') {
      setEditTitle(conversation.title)
      setIsEditing(false)
    }
  }

  const handleClick = () => {
    // Skip if drag was activated to prevent select after drag
    if (dragActivatedRef.current) return
    if (!isEditing) {
      onSelect(conversation.id)
    }
  }

  return (
    <div
      className={`conversation-item ${isActive ? 'active' : ''}`}
      tabIndex={0}
      onClick={handleClick}
      onContextMenu={(e) => onContextMenu(e, 'conversation', conversation.id)}
      onMouseDown={(e) => {
        // Don't start drag if clicking edit input
        if ((e.target as HTMLElement).closest('.conversation-edit-input')) return
        onDragStart(e, 'conversation', conversation.id)
      }}
      style={{ pointerEvents: 'auto', cursor: 'grab' }}
    >
      {isEditing ? (
        <input
          ref={inputRef}
          className="conversation-edit-input"
          value={editTitle}
          onChange={(e) => setEditTitle(e.target.value)}
          onBlur={handleSaveRename}
          onKeyDown={handleKeyDown}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <>
          <span className="conversation-title">{conversation.title}</span>
          <span className="conversation-meta">
            <span className="conversation-date">{formatTimestamp(conversation.updatedAt)}</span>
            <span className="conversation-message-count">{conversation.messages.length}</span>
          </span>
        </>
      )}
    </div>
  )
}

// ─── Context Menu ─────────────────────────────────────────────────────────────

interface ContextMenuProps {
  visible: boolean
  x: number
  y: number
  targetType: 'conversation' | 'folder' | null
  targetId: string | null
  onCopy: () => void
  onCut: () => void
  onPaste: () => void
  canPaste: boolean
  onRename: () => void
  onDelete: () => void
  onExport?: () => void
  onClose: () => void
}

const ContextMenu: React.FC<ContextMenuProps> = ({
  visible,
  x,
  y,
  targetType,
  onCopy,
  onCut,
  onPaste,
  canPaste,
  onRename,
  onDelete,
  onExport,
  onClose,
}) => {
  if (!visible || !targetType) return null

  return (
    <>
      <div className="context-menu-overlay" onClick={onClose} />
      <div className="context-menu" style={{ top: y, left: x }}>
        <button onClick={onCopy} className="context-menu-item">
          <span className="context-menu-icon">📋</span>
          Copy
        </button>
        <button onClick={onCut} className="context-menu-item">
          <span className="context-menu-icon">✂️</span>
          Cut
        </button>
        {canPaste && (
          <button onClick={onPaste} className="context-menu-item">
            <span className="context-menu-icon">📌</span>
            Paste
          </button>
        )}
        {onExport && (
          <button onClick={onExport} className="context-menu-item">
            <span className="context-menu-icon">📤</span>
            Export
          </button>
        )}
        <button onClick={onRename} className="context-menu-item">
          <span className="context-menu-icon">✏️</span>
          Rename
        </button>
        <button onClick={onDelete} className="context-menu-item context-menu-item-danger">
          <span className="context-menu-icon">🗑️</span>
          Delete
        </button>
      </div>
    </>
  )
}

// ─── Delete Confirmation Dialog ───────────────────────────────────────────────

interface DeleteConfirmDialogProps {
  visible: boolean
  itemType: 'conversation' | 'folder' | null
  itemName: string
  onConfirm: () => void
  onCancel: () => void
}

const DeleteConfirmDialog: React.FC<DeleteConfirmDialogProps> = ({
  visible,
  itemType,
  itemName,
  onConfirm,
  onCancel,
}) => {
  if (!visible || !itemType) return null

  return (
    <>
      <div className="modal-overlay" onClick={onCancel} />
      <div className="modal-dialog delete-confirm-dialog">
        <h3>Delete {itemType === 'folder' ? 'Folder' : 'Conversation'}</h3>
        <p>Are you sure you want to delete <strong>{itemName}</strong>?</p>
        {itemType === 'folder' && (
          <p className="delete-warning">
            Conversations in this folder will be moved to the root level.
          </p>
        )}
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button className="btn btn-danger" onClick={onConfirm}>
            Delete
          </button>
        </div>
      </div>
    </>
  )
}

// ─── Unified Tree Node ────────────────────────────────────────────────────────
// Renders conversations and folders interleaved at each level of the tree

interface UnifiedTreeNodeProps {
  parentId: string | null  // null = root level
  allFolders: ConversationFolder[]
  conversations: Conversation[]
  activeConversationId: string | null
  selectedFolderId: string | null
  expandedFolders: Set<string>
  onToggleFolder: (folderId: string) => void
  onSelectFolder: (folderId: string | null) => void
  onSelectConversation: (id: string) => void
  onRenameFolder: (folderId: string, newName: string) => void
  onDeleteFolder: (folderId: string) => void
  onRenameConversation: (convId: string, newTitle: string) => void
  onDeleteConversation: (convId: string) => void
  contextMenu: ContextMenuState
  onContextMenu: (e: React.MouseEvent, type: 'conversation' | 'folder', id: string) => void
  onCloseContextMenu: () => void
  /** Drag start callback passed down */
  onDragStart: (e: React.MouseEvent, type: 'conversation' | 'folder', id: string) => void
  /** Current drag state for highlighting */
  dragState: DraggingInfo | null
  /** Ref to check if drag was activated (prevents click after drag) */
  dragActivatedRef: React.MutableRefObject<boolean>
}

const UnifiedTreeNode: React.FC<UnifiedTreeNodeProps> = ({
  parentId,
  allFolders,
  conversations,
  activeConversationId,
  selectedFolderId,
  expandedFolders,
  onToggleFolder,
  onSelectFolder,
  onSelectConversation,
  onRenameFolder,
  onDeleteFolder,
  onRenameConversation,
  onDeleteConversation,
  contextMenu,
  onContextMenu,
  onCloseContextMenu,
  onDragStart,
  dragState,
  dragActivatedRef,
}) => {
  // Get folders at this level
  const foldersAtLevel = allFolders.filter((f) => f.parentId === parentId)
  // Get conversations at this level
  const convsAtLevel = conversations.filter((c) => c.folderId === parentId)

  // Build a unified, interleaved list sorted by most recent activity
  // Folders are treated as having "updatedAt" = their latest child conversation's updatedAt, or createdAt if empty
  const getFolderEffectiveDate = (folder: ConversationFolder): number => {
    const childConvs = conversations.filter((c) => c.folderId === folder.id)
    if (childConvs.length === 0) {
      return new Date(folder.createdAt).getTime()
    }
    return Math.max(...childConvs.map((c) => new Date(c.updatedAt).getTime()))
  }

  const unifiedItems = [
    ...convsAtLevel.map((conv) => ({
      type: 'conversation' as const,
      conv,
      date: new Date(conv.updatedAt).getTime(),
    })),
    ...foldersAtLevel.map((folder) => ({
      type: 'folder' as const,
      folder,
      date: getFolderEffectiveDate(folder),
    })),
  ].sort((a, b) => b.date - a.date) // Most recent first

  return (
    <div className="unified-tree-node">
      {unifiedItems.map((item) =>
        item.type === 'conversation' ? (
          <div key={item.conv.id} data-conv-id={item.conv.id}>
            <ConversationItem
              conversation={item.conv}
              isActive={activeConversationId === item.conv.id}
              onSelect={onSelectConversation}
              onRename={onRenameConversation}
              onContextMenu={onContextMenu}
              onDragStart={onDragStart}
              dragActivatedRef={dragActivatedRef}
            />
          </div>
        ) : (
          <div key={item.folder.id} data-folder-id={item.folder.id}>
            <FolderTreeNode
              folder={item.folder}
              allFolders={allFolders}
              conversations={conversations}
              activeConversationId={activeConversationId}
              selectedFolderId={selectedFolderId}
              expandedFolders={expandedFolders}
              onToggleFolder={onToggleFolder}
              onSelectFolder={onSelectFolder}
              onSelectConversation={onSelectConversation}
              onRenameFolder={onRenameFolder}
              onDeleteFolder={onDeleteFolder}
              onRenameConversation={onRenameConversation}
              onDeleteConversation={onDeleteConversation}
              contextMenu={contextMenu}
              onContextMenu={onContextMenu}
              onCloseContextMenu={onCloseContextMenu}
              onDragStart={onDragStart}
              dragState={dragState}
              dragActivatedRef={dragActivatedRef}
            />
          </div>
        )
      )}
    </div>
  )
}

// ─── Main Sidebar Component ───────────────────────────────────────────────────

export const ConversationSidebar: React.FC = () => {
  const { state, dispatch, deleteConversation, renameConversation, createFolder, createConversation, exportConversation, exportFolder, importFromFile } = useChat()
  // Auto-expand all folders by default so conversations inside are visible
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(() => {
    // Initialize with all folder IDs so nothing appears empty
    return new Set(state.folders.map((f) => f.id))
  })
  // Keep expandedFolders synced when folders are added/removed
  useEffect(() => {
    setExpandedFolders((prev) => {
      const next = new Set(prev)
      // Auto-expand newly created folders
      state.folders.forEach((f) => next.add(f.id))
      return next
    })
  }, [state.folders.length])
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({
    visible: false,
    x: 0,
    y: 0,
    targetType: null,
    targetId: null,
  })
  const [deleteConfirm, setDeleteConfirm] = useState<{
    visible: boolean
    type: 'conversation' | 'folder'
    id: string
    name: string
  }>({ visible: false, type: 'conversation', id: '', name: '' })
  // Track which folder is currently "active" - new conversations go here
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null)
  // dragState moved to dragStateRef to avoid stale closure in handleDrop
  // Clipboard supports both conversations and folders
  const [clipboard, setClipboard] = useState<{
    action: 'copy' | 'cut'
    type: 'conversation' | 'folder'
    conversationId?: string
    folderId?: string
  } | null>(null)
  const [importResult, setImportResult] = useState<{ count: number; success: boolean } | null>(null)
  const sidebarRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Expand folder when conversation is selected inside
  useEffect(() => {
    if (state.activeConversationId) {
      const activeConv = state.conversations.find((c) => c.id === state.activeConversationId)
      if (activeConv?.folderId) {
        setExpandedFolders((prev) => new Set(prev).add(activeConv.folderId!))
      }
    }
  }, [state.activeConversationId, state.conversations])

  // Toggle folder expand/collapse
  const handleToggleFolder = useCallback((folderId: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev)
      if (next.has(folderId)) {
        next.delete(folderId)
      } else {
        next.add(folderId)
      }
      return next
    })
  }, [])

  const handleSelectFolder = useCallback((folderId: string | null) => {
    setSelectedFolderId(folderId)
    // Clear active conversation when selecting a folder
    if (folderId !== null) {
      dispatch({ type: 'SET_ACTIVE_CONVERSATION', payload: null })
    }
  }, [dispatch])

  // Rename folder
  const handleRenameFolder = useCallback(
    (folderId: string, newName: string) => {
      const folder = state.folders.find((f) => f.id === folderId)
      if (folder) {
        dispatch({
          type: 'UPDATE_FOLDER',
          payload: { ...folder, name: newName },
        })
      }
    },
    [state.folders, dispatch]
  )

  // Delete folder
  const handleDeleteFolder = useCallback(
    (folderId: string) => {
      dispatch({ type: 'DELETE_FOLDER', payload: folderId })
    },
    [dispatch]
  )

  // Handle context menu
  const handleContextMenu = useCallback(
    (e: React.MouseEvent, type: 'conversation' | 'folder', id: string) => {
      e.preventDefault()
      e.stopPropagation()
      setContextMenu({
        visible: true,
        x: e.clientX,
        y: e.clientY,
        targetType: type,
        targetId: id,
      })
    },
    []
  )

  const handleCloseContextMenu = useCallback(() => {
    setContextMenu((prev) => ({ ...prev, visible: false }))
  }, [])

  // Handle rename from context menu
  const handleContextRename = useCallback(() => {
    if (contextMenu.targetType === 'conversation' && contextMenu.targetId) {
      // Find the conversation item and trigger edit mode
      const convItem = document.querySelector(`[data-conv-id="${contextMenu.targetId}"]`)
      if (convItem) {
        const titleEl = convItem.querySelector('.conversation-title')
        titleEl?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
      }
    } else if (contextMenu.targetType === 'folder' && contextMenu.targetId) {
      const folderItem = document.querySelector(`[data-folder-id="${contextMenu.targetId}"]`)
      if (folderItem) {
        const nameEl = folderItem.querySelector('.folder-name')
        nameEl?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
      }
    }
    handleCloseContextMenu()
  }, [contextMenu, handleCloseContextMenu])

  // Handle delete from context menu
  const handleContextDelete = useCallback(() => {
    if (contextMenu.targetType === 'conversation' && contextMenu.targetId) {
      const conv = state.conversations.find((c) => c.id === contextMenu.targetId)
      setDeleteConfirm({
        visible: true,
        type: 'conversation',
        id: contextMenu.targetId,
        name: conv?.title || 'Unknown',
      })
    } else if (contextMenu.targetType === 'folder' && contextMenu.targetId) {
      const folder = state.folders.find((f) => f.id === contextMenu.targetId)
      setDeleteConfirm({
        visible: true,
        type: 'folder',
        id: contextMenu.targetId,
        name: folder?.name || 'Unknown',
      })
    }
    handleCloseContextMenu()
  }, [contextMenu, state.conversations, state.folders, handleCloseContextMenu])

  // Handle copy from context menu
  const handleContextCopy = useCallback(() => {
    if (contextMenu.targetType === 'conversation' && contextMenu.targetId) {
      setClipboard({ action: 'copy', type: 'conversation', conversationId: contextMenu.targetId })
    } else if (contextMenu.targetType === 'folder' && contextMenu.targetId) {
      setClipboard({ action: 'copy', type: 'folder', folderId: contextMenu.targetId })
    }
    handleCloseContextMenu()
  }, [contextMenu, handleCloseContextMenu])

  // Handle cut from context menu
  const handleContextCut = useCallback(() => {
    if (contextMenu.targetType === 'conversation' && contextMenu.targetId) {
      setClipboard({ action: 'cut', type: 'conversation', conversationId: contextMenu.targetId })
    } else if (contextMenu.targetType === 'folder' && contextMenu.targetId) {
      setClipboard({ action: 'cut', type: 'folder', folderId: contextMenu.targetId })
    }
    handleCloseContextMenu()
  }, [contextMenu, handleCloseContextMenu])

  // Handle paste from context menu
  const handleContextPaste = useCallback(() => {
    if (clipboard) {
      if (clipboard.type === 'conversation' && clipboard.conversationId) {
        const sourceConv = state.conversations.find((c) => c.id === clipboard.conversationId)
        if (sourceConv) {
          const now = new Date().toISOString()
          const newConv: Conversation = {
            ...sourceConv,
            id: `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
            title: sourceConv.title + (clipboard.action === 'copy' ? ' (Copy)' : ''),
            folderId: selectedFolderId,
            updatedAt: now,
          }
          dispatch({ type: 'CREATE_CONVERSATION', payload: newConv })
          // On cut, delete the original source
          if (clipboard.action === 'cut') {
            dispatch({ type: 'DELETE_CONVERSATION', payload: clipboard.conversationId })
            setClipboard(null)
          }
        }
      } else if (clipboard.type === 'folder' && clipboard.folderId) {
        const sourceFolder = state.folders.find((f) => f.id === clipboard.folderId)
        if (sourceFolder) {
          const newFolder: ConversationFolder = {
            ...sourceFolder,
            id: `folder-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
            name: sourceFolder.name + (clipboard.action === 'copy' ? ' (Copy)' : ''),
            parentId: selectedFolderId,
          }
          dispatch({ type: 'CREATE_FOLDER', payload: newFolder })
          // On cut, delete the original source
          if (clipboard.action === 'cut') {
            // Delete source folder (its children are moved to root by reducer)
            dispatch({ type: 'DELETE_FOLDER', payload: clipboard.folderId })
            setClipboard(null)
          }
        }
      }
    }
    handleCloseContextMenu()
  }, [clipboard, state.conversations, state.folders, selectedFolderId, dispatch, handleCloseContextMenu])

  // Confirm delete
  const handleConfirmDelete = useCallback(() => {
    if (deleteConfirm.type === 'conversation') {
      deleteConversation(deleteConfirm.id)
    } else {
      handleDeleteFolder(deleteConfirm.id)
    }
    setDeleteConfirm((prev) => ({ ...prev, visible: false }))
  }, [deleteConfirm, deleteConversation, handleDeleteFolder])

  // Custom mouse-based drag system (dnd-kit removed due to WebView2 incompatibility)
  const dragInfoRef = useRef<DraggingInfo | null>(null)
  const [dragState, setDragState] = useState<DraggingInfo | null>(null)
  // Ref to track if drag was activated (prevents click handlers from firing after drag)
  const dragActivatedRef = useRef(false)

  const handleMouseDownDrag = useCallback(
    (e: React.MouseEvent, type: 'conversation' | 'folder', id: string) => {
      // Only left button
      if (e.button !== 0) return
      // Prevent default to avoid text selection during drag
      e.preventDefault()
      // Stop propagation so onClick on same element doesn't fire after drag
      e.stopPropagation()

      // Determine the source folder
      let startFolderId: string | null = null
      if (type === 'conversation') {
        const conv = state.conversations.find((c) => c.id === id)
        startFolderId = conv?.folderId ?? null
      } else {
        const folder = state.folders.find((f) => f.id === id)
        startFolderId = folder?.parentId ?? null
      }

      const info: DraggingInfo = { type, id, startFolderId }
      dragInfoRef.current = info
      dragActivatedRef.current = false

      const startX = e.clientX
      const startY = e.clientY

      // Drop indicator line element
      let dropLine: HTMLDivElement | null = null

      const showDropLine = (targetEl: HTMLElement, above: boolean) => {
        if (!dropLine) {
          dropLine = document.createElement('div')
          dropLine.className = 'drag-drop-line'
          document.body.appendChild(dropLine)
        }
        const rect = targetEl.getBoundingClientRect()
        const scrollY = window.scrollY
        dropLine.style.top = `${rect.top + scrollY + (above ? 0 : rect.height)}px`
        dropLine.style.left = `${rect.left}px`
        dropLine.style.width = `${rect.width}px`
        dropLine.style.display = 'block'
      }

      const hideDropLine = () => {
        if (dropLine) {
          dropLine.style.display = 'none'
        }
      }

      const handleMouseMove = (me: MouseEvent) => {
        // Activate drag only after 5px movement to distinguish from click
        if (!dragActivatedRef.current && (Math.abs(me.clientX - startX) > 5 || Math.abs(me.clientY - startY) > 5)) {
          dragActivatedRef.current = true
          setDragState({ ...dragInfoRef.current! })
        }
        if (dragActivatedRef.current) {
          setDragState({ ...dragInfoRef.current! })
          // Visual drop target highlighting
          const elem = document.elementFromPoint(me.clientX, me.clientY)
          // Clear previous highlights
          document.querySelectorAll('.drag-drop-target').forEach(el => el.classList.remove('drag-drop-target'))
          hideDropLine()
          if (elem) {
            const convEl = elem.closest('[data-conv-id]')
            const folderEl = elem.closest('[data-folder-id]')
            if (convEl) {
              convEl.classList.add('drag-drop-target')
              // Determine drop position: above or below based on Y position
              const rect = convEl.getBoundingClientRect()
              const midY = rect.top + rect.height / 2
              showDropLine(convEl as HTMLElement, me.clientY < midY)
            } else if (folderEl) {
              folderEl.classList.add('drag-drop-target')
              const rect = folderEl.getBoundingClientRect()
              const midY = rect.top + rect.height / 2
              showDropLine(folderEl as HTMLElement, me.clientY < midY)
            } else {
              // Dropped on empty area - show line at the current Y position
              const sidebarEl = document.querySelector('.conversation-sidebar')
              if (sidebarEl) {
                const rect = sidebarEl.getBoundingClientRect()
                const fakeTarget = document.createElement('div')
                fakeTarget.style.top = `${rect.top}px`
                fakeTarget.style.height = '0px'
                showDropLine(fakeTarget as HTMLElement, true)
                hideDropLine() // Don't show on empty area - just highlight the area
              }
            }
          }
        }
      }

      const handleMouseUp = (me: MouseEvent) => {
        // Remove listeners first
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)

        if (!dragInfoRef.current) return

        // Find the element under the pointer
        const elem = document.elementFromPoint(me.clientX, me.clientY)
        if (!elem) {
          dragInfoRef.current = null
          setDragState(null)
          hideDropLine()
          setTimeout(() => { dragActivatedRef.current = false }, 0)
          return
        }

        // Determine target folder by walking up DOM
        let targetFolderId: string | null = null
        const convEl = elem.closest('[data-conv-id]')
        const folderEl = elem.closest('[data-folder-id]')
        const rootEl = elem.closest('#sidebar-root-drop')

        if (convEl) {
          const convId = convEl.getAttribute('data-conv-id')
          const targetConv = state.conversations.find((c) => c.id === convId)
          targetFolderId = targetConv?.folderId ?? null
        } else if (folderEl) {
          targetFolderId = folderEl.getAttribute('data-folder-id') ?? null
        } else if (rootEl) {
          // Dropped on empty area of sidebar = root
          targetFolderId = null
        }

        // Prevent dropping on self or into descendants
        const { type: dragType, id: dragId, startFolderId } = dragInfoRef.current

        if (dragType === 'folder' && targetFolderId) {
          const isSelfOrDescendant = (candidateId: string, targetId: string): boolean => {
            if (candidateId === targetId) return true
            const childFolders = state.folders.filter((f) => f.parentId === candidateId)
            return childFolders.some((f) => isSelfOrDescendant(f.id, targetId))
          }
          if (isSelfOrDescendant(dragId, targetFolderId)) {
            dragInfoRef.current = null
            setDragState(null)
            hideDropLine()
            setTimeout(() => { dragActivatedRef.current = false }, 0)
            return
          }
        }

        // Execute the move only if target differs from source AND drag was activated
        if (dragActivatedRef.current) {
          if (dragType === 'conversation' && targetFolderId !== startFolderId) {
            dispatch({
              type: 'MOVE_CONVERSATION',
              payload: { conversationId: dragId, folderId: targetFolderId },
            })
          } else if (dragType === 'folder' && targetFolderId !== startFolderId) {
            dispatch({
              type: 'MOVE_FOLDER',
              payload: { folderId: dragId, parentId: targetFolderId },
            })
          }
        }

        dragInfoRef.current = null
        setDragState(null)
        // Defer resetting dragActivatedRef so that the browser's subsequent click event
        // (which fires after mouseup) can still see it as true and be suppressed.
        setTimeout(() => {
          dragActivatedRef.current = false
          hideDropLine()
        }, 0)
      }

      // Attach to document to capture events even when cursor leaves the element
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    },
    [dispatch, state.conversations, state.folders]
  )

  // Keyboard shortcuts -- scoped to sidebar element only
  useEffect(() => {
    const sidebarEl = sidebarRef.current
    if (!sidebarEl) return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle shortcuts when sidebar has focus (not when typing in chat input)
      if (!sidebarEl.contains(document.activeElement)) return

      // Ctrl+C / Cmd+C - Copy selected conversation or folder
      if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !window.getSelection()?.toString()) {
        if (state.activeConversationId) {
          setClipboard({ action: 'copy', type: 'conversation', conversationId: state.activeConversationId })
        } else if (selectedFolderId) {
          setClipboard({ action: 'copy', type: 'folder', folderId: selectedFolderId })
        }
      }
      // Ctrl+X / Cmd+X - Cut selected conversation or folder
      if ((e.ctrlKey || e.metaKey) && e.key === 'x' && !window.getSelection()?.toString()) {
        if (state.activeConversationId) {
          setClipboard({ action: 'cut', type: 'conversation', conversationId: state.activeConversationId })
        } else if (selectedFolderId) {
          setClipboard({ action: 'cut', type: 'folder', folderId: selectedFolderId })
        }
      }
      // Ctrl+V / Cmd+V - Paste (duplicate) conversation or folder into selected folder
      if ((e.ctrlKey || e.metaKey) && e.key === 'v' && clipboard) {
        e.preventDefault()
        if (clipboard.type === 'conversation' && clipboard.conversationId) {
          const sourceConv = state.conversations.find((c) => c.id === clipboard.conversationId)
          if (sourceConv) {
            const now = new Date().toISOString()
            const newConv: Conversation = {
              ...sourceConv,
              id: `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
              title: sourceConv.title + (clipboard.action === 'copy' ? ' (Copy)' : ''),
              folderId: selectedFolderId, // Paste into selected folder
              createdAt: now,
              updatedAt: now,
            }
            dispatch({ type: 'CREATE_CONVERSATION', payload: newConv })
            if (clipboard.action === 'cut') {
              dispatch({ type: 'DELETE_CONVERSATION', payload: clipboard.conversationId })
            }
            setClipboard(null)
          }
        } else if (clipboard.type === 'folder' && clipboard.folderId) {
          const sourceFolder = state.folders.find((f) => f.id === clipboard.folderId)
          if (sourceFolder) {
            const newFolder: ConversationFolder = {
              ...sourceFolder,
              id: `folder-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
              name: sourceFolder.name + (clipboard.action === 'copy' ? ' (Copy)' : ''),
              parentId: selectedFolderId, // Nest into selected folder
              createdAt: new Date().toISOString(),
            }
            dispatch({ type: 'CREATE_FOLDER', payload: newFolder })
            if (clipboard.action === 'cut') {
              dispatch({ type: 'DELETE_FOLDER', payload: clipboard.folderId })
            }
            setClipboard(null)
          }
        }
      }
      // Delete key - delete selected conversation or folder
      if (e.key === 'Delete') {
        if (state.activeConversationId) {
          const conv = state.conversations.find((c) => c.id === state.activeConversationId)
          setDeleteConfirm({
            visible: true,
            type: 'conversation',
            id: state.activeConversationId,
            name: conv?.title || 'Unknown',
          })
        } else if (selectedFolderId) {
          const folder = state.folders.find((f) => f.id === selectedFolderId)
          setDeleteConfirm({
            visible: true,
            type: 'folder',
            id: selectedFolderId,
            name: folder?.name || 'Unknown',
          })
        }
      }
      // F2 - rename selected conversation or folder
      if (e.key === 'F2') {
        e.preventDefault()
        if (state.activeConversationId) {
          const convItem = document.querySelector(`[data-conv-id="${state.activeConversationId}"]`)
          if (convItem) {
            const titleEl = convItem.querySelector('.conversation-title')
            titleEl?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
          }
        } else if (selectedFolderId) {
          const folderItem = document.querySelector(`[data-folder-id="${selectedFolderId}"]`)
          if (folderItem) {
            const nameEl = folderItem.querySelector('.folder-name')
            nameEl?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
          }
        }
      }
    }

    sidebarEl.addEventListener('keydown', handleKeyDown)
    return () => {
      sidebarEl.removeEventListener('keydown', handleKeyDown)
    }
  }, [state.activeConversationId, state.conversations, state.folders, selectedFolderId, clipboard, dispatch])

  // Close context menu on click outside
  useEffect(() => {
    const handleClick = () => {
      handleCloseContextMenu()
    }
    if (contextMenu.visible) {
      document.addEventListener('click', handleClick)
      return () => document.removeEventListener('click', handleClick)
    }
  }, [contextMenu.visible, handleCloseContextMenu])

  const handleNewConversation = () => {
    // Create conversation in selected folder (or root if none selected)
    createConversation(selectedFolderId)
    // Clear selected folder after creating conversation
    setSelectedFolderId(null)
  }

  const handleNewFolder = () => {
    createFolder('New Folder')
  }

  // Export handlers
  const handleExportConversation = useCallback(() => {
    if (contextMenu.targetType === 'conversation' && contextMenu.targetId) {
      exportConversation(contextMenu.targetId)
    } else if (contextMenu.targetType === 'folder' && contextMenu.targetId) {
      exportFolder(contextMenu.targetId)
    }
    handleCloseContextMenu()
  }, [contextMenu, exportConversation, exportFolder, handleCloseContextMenu])

  // Import handler
  const handleImportClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      try {
        const count = await importFromFile(file)
        setImportResult({ count, success: true })
      } catch (err) {
        console.error('[ConversationSidebar] Import failed:', err)
        setImportResult({ count: 0, success: false })
      }
    }
    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Dismiss import result after 3 seconds
  useEffect(() => {
    if (importResult) {
      const timer = setTimeout(() => setImportResult(null), 3000)
      return () => clearTimeout(timer)
    }
  }, [importResult])

  return (
    <div className="conversation-sidebar" ref={sidebarRef} tabIndex={0}>
      {/* Sidebar Header */}
      <div className="sidebar-header">
        <h3>Conversations</h3>
        <div className="sidebar-actions">
          <button
            className="sidebar-btn sidebar-btn-icon"
            onClick={handleImportClick}
            title="Import Conversations"
            aria-label="Import conversations from file"
          >
            📥
          </button>
          <button
            className="sidebar-btn sidebar-btn-icon"
            onClick={handleNewFolder}
            title="New Folder"
            aria-label="Create new folder"
          >
            📁
          </button>
          <button
            className="sidebar-btn sidebar-btn-primary"
            onClick={handleNewConversation}
            title="New Conversation"
            aria-label="Create new conversation"
          >
            + New
          </button>
        </div>
      </div>

      {/* Hidden file input for imports */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,.txt"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* Sidebar Content — Unified Tree with custom pointer-based drag */}
      <div className="sidebar-content">
        {state.conversations.length === 0 && state.folders.length === 0 ? (
          <div className="sidebar-empty-state">
            <span className="empty-state-icon">💬</span>
            <p>No conversations yet</p>
            <button className="btn btn-secondary" onClick={handleNewConversation}>
              Start a conversation
            </button>
          </div>
        ) : (
          <div id="sidebar-root-drop">
            <UnifiedTreeNode
              parentId={null}
              allFolders={state.folders}
              conversations={state.conversations}
              activeConversationId={state.activeConversationId}
              selectedFolderId={selectedFolderId}
              expandedFolders={expandedFolders}
              onToggleFolder={handleToggleFolder}
              onSelectFolder={handleSelectFolder}
              onSelectConversation={(id) => {
                dispatch({ type: 'SET_ACTIVE_CONVERSATION', payload: id })
                setSelectedFolderId(null) // Clear folder selection when selecting conversation
              }}
              onRenameFolder={handleRenameFolder}
              onDeleteFolder={(folderId) =>
                setDeleteConfirm({
                  visible: true,
                  type: 'folder',
                  id: folderId,
                  name: state.folders.find((f) => f.id === folderId)?.name || 'Unknown',
                })
              }
              onRenameConversation={renameConversation}
              onDeleteConversation={(convId) =>
                setDeleteConfirm({
                  visible: true,
                  type: 'conversation',
                  id: convId,
                  name: state.conversations.find((c) => c.id === convId)?.title || 'Unknown',
                })
              }
              contextMenu={contextMenu}
              onContextMenu={handleContextMenu}
              onCloseContextMenu={handleCloseContextMenu}
              onDragStart={handleMouseDownDrag}
              dragState={dragState}
              dragActivatedRef={dragActivatedRef}
            />
          </div>
        )}
      </div>

      {/* Sidebar Footer */}
      <div className="sidebar-footer">
        <span className="sidebar-stats">
          {state.conversations.length} conversation{state.conversations.length !== 1 ? 's' : ''}
          {state.folders.length > 0 && ` · ${state.folders.length} folder${state.folders.length !== 1 ? 's' : ''}`}
        </span>
        {clipboard && (
          <span className="clipboard-indicator" title={`Ready to ${clipboard.action} ${clipboard.type}`} onClick={() => setClipboard(null)}>
            📋 {clipboard.action === 'copy' ? 'Copied' : 'Cut'} {clipboard.type === 'folder' ? '(folder)' : ''}
          </span>
        )}
      </div>

      {/* Context Menu */}
      <ContextMenu
        visible={contextMenu.visible}
        x={contextMenu.x}
        y={contextMenu.y}
        targetType={contextMenu.targetType}
        targetId={contextMenu.targetId}
        onCopy={handleContextCopy}
        onCut={handleContextCut}
        onPaste={handleContextPaste}
        canPaste={!!clipboard}
        onRename={handleContextRename}
        onDelete={handleContextDelete}
        onExport={handleExportConversation}
        onClose={handleCloseContextMenu}
      />

      {/* Import Result Toast */}
      {importResult && (
        <div className="import-toast" style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '10px 20px',
          borderRadius: '6px',
          background: importResult.success ? '#4caf50' : '#f44336',
          color: 'white',
          zIndex: 1000,
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        }}>
          {importResult.success
            ? `✅ Imported ${importResult.count} conversation${importResult.count !== 1 ? 's' : ''}`
            : '❌ Import failed'}
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      <DeleteConfirmDialog
        visible={deleteConfirm.visible}
        itemType={deleteConfirm.type}
        itemName={deleteConfirm.name}
        onConfirm={handleConfirmDelete}
        onCancel={() => setDeleteConfirm((prev) => ({ ...prev, visible: false }))}
      />
    </div>
  )
}

export default ConversationSidebar
