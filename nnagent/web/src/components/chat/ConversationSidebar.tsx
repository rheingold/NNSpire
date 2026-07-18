/**
 * ConversationSidebar — Full-featured conversation list with folder organization.
 *
 * Features:
 * - Multi-level folder nesting with expand/collapse
 * - Rename conversations and folders (inline editing)
 * - Delete conversations and folders with confirmation
 * - Drag-and-drop reorganization
 * - Keyboard shortcuts (Ctrl+C/X/V for copy/cut/paste)
 * - Context menu for actions
 * - Auto-generated titles from first prompt
 */

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { useChat } from '@/context/ChatContext'
import { Conversation, ConversationFolder, formatTimestamp } from '@/types/chat'

// ─── Types ────────────────────────────────────────────────────────────────────

interface DragState {
  type: 'conversation' | 'folder'
  id: string
}

interface ContextMenuState {
  visible: boolean
  x: number
  y: number
  targetType: 'conversation' | 'folder' | null
  targetId: string | null
}

// ─── Folder Tree Node ─────────────────────────────────────────────────────────

interface FolderTreeNodeProps {
  folder: ConversationFolder
  allFolders: ConversationFolder[]
  conversations: Conversation[]
  activeConversationId: string | null
  expandedFolders: Set<string>
  onToggleFolder: (folderId: string) => void
  onSelectConversation: (id: string) => void
  onRenameFolder: (folderId: string, newName: string) => void
  onDeleteFolder: (folderId: string) => void
  onRenameConversation: (convId: string, newTitle: string) => void
  onDeleteConversation: (convId: string) => void
  onDragStart: (item: DragState) => void
  onDrop: (targetFolderId: string | null) => void
  contextMenu: ContextMenuState
  onContextMenu: (e: React.MouseEvent, type: 'conversation' | 'folder', id: string) => void
  onCloseContextMenu: () => void
}

const FolderTreeNode: React.FC<FolderTreeNodeProps> = ({
  folder,
  allFolders,
  conversations,
  activeConversationId,
  expandedFolders,
  onToggleFolder,
  onSelectConversation,
  onRenameFolder,
  onDeleteFolder,
  onRenameConversation,
  onDeleteConversation,
  onDragStart,
  onDrop,
  contextMenu,
  onContextMenu,
  onCloseContextMenu,
}) => {
  const childFolders = allFolders.filter((f) => f.parentId === folder.id)
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
    e.stopPropagation()
    onToggleFolder(folder.id)
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

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }

  const handleDropOnFolder = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onDrop(folder.id)
  }

  return (
    <div className="folder-tree-node">
      {/* Folder Header */}
      <div
        className={`folder-header ${isExpanded ? 'expanded' : ''}`}
        onClick={handleToggle}
        onContextMenu={(e) => onContextMenu(e, 'folder', folder.id)}
        draggable
        onDragStart={(e) => {
          e.dataTransfer.effectAllowed = 'move'
          onDragStart({ type: 'folder', id: folder.id })
        }}
        onDragOver={handleDragOver}
        onDrop={handleDropOnFolder}
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

      {/* Child Content */}
      {isExpanded && (
        <div className="folder-content">
          {/* Child Folders First */}
          {childFolders
            .sort((a, b) => a.order - b.order)
            .map((childFolder) => (
              <FolderTreeNode
                key={childFolder.id}
                folder={childFolder}
                allFolders={allFolders}
                conversations={conversations}
                activeConversationId={activeConversationId}
                expandedFolders={expandedFolders}
                onToggleFolder={onToggleFolder}
                onSelectConversation={onSelectConversation}
                onRenameFolder={onRenameFolder}
                onDeleteFolder={onDeleteFolder}
                onRenameConversation={onRenameConversation}
                onDeleteConversation={onDeleteConversation}
                onDragStart={onDragStart}
                onDrop={onDrop}
                contextMenu={contextMenu}
                onContextMenu={onContextMenu}
                onCloseContextMenu={onCloseContextMenu}
              />
            ))}

          {/* Conversations in this Folder */}
          {folderConversations
            .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
            .map((conv) => (
              <ConversationItem
                key={conv.id}
                conversation={conv}
                isActive={activeConversationId === conv.id}
                onSelect={onSelectConversation}
                onRename={onRenameConversation}
                onDragStart={onDragStart}
                onContextMenu={onContextMenu}
              />
            ))}
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
  onDragStart: (item: DragState) => void
  onContextMenu: (e: React.MouseEvent, type: 'conversation', id: string) => void
}

const ConversationItem: React.FC<ConversationItemProps> = ({
  conversation,
  isActive,
  onSelect,
  onRename,
  onDragStart,
  onContextMenu,
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

  return (
    <div
      className={`conversation-item ${isActive ? 'active' : ''}`}
      onClick={() => {
        if (!isEditing) onSelect(conversation.id)
      }}
      onContextMenu={(e) => onContextMenu(e, 'conversation', conversation.id)}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = 'move'
        onDragStart({ type: 'conversation', id: conversation.id })
      }}
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
  // dragState moved to dragStateRef to avoid stale closure in handleDrop
  const [clipboard, setClipboard] = useState<{ action: 'copy' | 'cut'; conversationId: string } | null>(null)
  const [importResult, setImportResult] = useState<{ count: number; success: boolean } | null>(null)
  const sidebarRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Root-level conversations (not in any folder)
  const rootConversations = useMemo(
    () => state.conversations.filter((c) => !c.folderId),
    [state.conversations]
  )

  // Root-level folders
  const rootFolders = useMemo(
    () => state.folders.filter((f) => !f.parentId),
    [state.folders]
  )

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

  // Confirm delete
  const handleConfirmDelete = useCallback(() => {
    if (deleteConfirm.type === 'conversation') {
      deleteConversation(deleteConfirm.id)
    } else {
      handleDeleteFolder(deleteConfirm.id)
    }
    setDeleteConfirm((prev) => ({ ...prev, visible: false }))
  }, [deleteConfirm, deleteConversation, handleDeleteFolder])

  // Drag and drop handlers -- use refs to avoid stale closures
  const dragStateRef = useRef<DragState | null>(null)

  const handleDragStart = useCallback((item: DragState) => {
    dragStateRef.current = item
  }, [])

  const handleDrop = useCallback(
    (targetFolderId: string | null) => {
      const currentDrag = dragStateRef.current
      if (currentDrag) {
        if (currentDrag.type === 'conversation') {
          dispatch({
            type: 'MOVE_CONVERSATION',
            payload: { conversationId: currentDrag.id, folderId: targetFolderId },
          })
        }
        // Note: Moving folders is more complex (need to move children too)
        // For now, only conversation moves are supported
      }
      dragStateRef.current = null
    },
    [dispatch]
  )

  // Keyboard shortcuts -- scoped to sidebar element only
  useEffect(() => {
    const sidebarEl = sidebarRef.current
    if (!sidebarEl) return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle shortcuts when sidebar has focus (not when typing in chat input)
      if (!sidebarEl.contains(document.activeElement)) return

      // Ctrl+C / Cmd+C - Copy selected conversation
      if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !window.getSelection()?.toString()) {
        if (state.activeConversationId) {
          setClipboard({ action: 'copy', conversationId: state.activeConversationId })
        }
      }
      // Ctrl+X / Cmd+X - Cut selected conversation
      if ((e.ctrlKey || e.metaKey) && e.key === 'x' && !window.getSelection()?.toString()) {
        if (state.activeConversationId) {
          setClipboard({ action: 'cut', conversationId: state.activeConversationId })
        }
      }
      // Ctrl+V / Cmd+V - Paste (duplicate) conversation
      if ((e.ctrlKey || e.metaKey) && e.key === 'v' && clipboard) {
        e.preventDefault()
        const sourceConv = state.conversations.find((c) => c.id === clipboard.conversationId)
        if (sourceConv) {
          const now = new Date().toISOString()
          const newConv: Conversation = {
            ...sourceConv,
            id: `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
            title: sourceConv.title + (clipboard.action === 'copy' ? ' (Copy)' : ''),
            createdAt: now,
            updatedAt: now,
          }
          dispatch({ type: 'CREATE_CONVERSATION', payload: newConv })
          if (clipboard.action === 'cut') {
            // Delete original after paste
            dispatch({ type: 'DELETE_CONVERSATION', payload: clipboard.conversationId })
          }
          setClipboard(null)
        }
      }
      // Delete key - delete selected conversation
      if (e.key === 'Delete' && state.activeConversationId) {
        const conv = state.conversations.find((c) => c.id === state.activeConversationId)
        setDeleteConfirm({
          visible: true,
          type: 'conversation',
          id: state.activeConversationId,
          name: conv?.title || 'Unknown',
        })
      }
      // F2 - rename selected conversation
      if (e.key === 'F2' && state.activeConversationId) {
        e.preventDefault()
        const convItem = document.querySelector(`[data-conv-id="${state.activeConversationId}"]`)
        if (convItem) {
          const titleEl = convItem.querySelector('.conversation-title')
          titleEl?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
        }
      }
    }

    sidebarEl.addEventListener('keydown', handleKeyDown)
    return () => {
      sidebarEl.removeEventListener('keydown', handleKeyDown)
    }
  }, [state.activeConversationId, state.conversations, clipboard, dispatch])

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
    createConversation()
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

      {/* Sidebar Content */}
      <div className="sidebar-content">
        {/* Root-level conversations */}
        {rootConversations.length > 0 && (
          <div className="sidebar-section">
            {rootConversations
              .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
              .map((conv) => (
                <div key={conv.id} data-conv-id={conv.id}>
                  <ConversationItem
                    conversation={conv}
                    isActive={state.activeConversationId === conv.id}
                    onSelect={(id) => dispatch({ type: 'SET_ACTIVE_CONVERSATION', payload: id })}
                    onRename={renameConversation}
                    onDragStart={handleDragStart}
                    onContextMenu={handleContextMenu}
                  />
                </div>
              ))}
          </div>
        )}

        {/* Folder Tree */}
        {rootFolders.length > 0 && (
          <div className="sidebar-section folder-section">
            <div className="sidebar-section-header">Folders</div>
            {rootFolders
              .sort((a, b) => a.order - b.order)
              .map((folder) => (
                <div key={folder.id} data-folder-id={folder.id}>
                  <FolderTreeNode
                    folder={folder}
                    allFolders={state.folders}
                    conversations={state.conversations}
                    activeConversationId={state.activeConversationId}
                    expandedFolders={expandedFolders}
                    onToggleFolder={handleToggleFolder}
                    onSelectConversation={(id) =>
                      dispatch({ type: 'SET_ACTIVE_CONVERSATION', payload: id })
                    }
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
                    onDragStart={handleDragStart}
                    onDrop={handleDrop}
                    contextMenu={contextMenu}
                    onContextMenu={handleContextMenu}
                    onCloseContextMenu={handleCloseContextMenu}
                  />
                </div>
              ))}
          </div>
        )}

        {/* Empty State */}
        {state.conversations.length === 0 && state.folders.length === 0 && (
          <div className="sidebar-empty-state">
            <span className="empty-state-icon">💬</span>
            <p>No conversations yet</p>
            <button className="btn btn-secondary" onClick={handleNewConversation}>
              Start a conversation
            </button>
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
          <span className="clipboard-indicator" title={`Ready to ${clipboard.action}`} onClick={() => setClipboard(null)}>
            📋 {clipboard.action === 'copy' ? 'Copied' : 'Cut'}
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
