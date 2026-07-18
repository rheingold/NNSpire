/**
 * Integration tests for ConversationSidebar component
 *
 * These tests render the actual DOM and verify user interactions work correctly,
 * unlike the existing renderHook tests which only test reducer logic.
 */

import { describe, it, expect, beforeEach, vi, waitFor } from 'vitest'
import { render, screen, fireEvent, waitFor as rtlWaitFor } from '@testing-library/react'
import React from 'react'
import { ConversationSidebar } from './ConversationSidebar'
import { ChatProvider } from '@/context/ChatContext'
import { Conversation, ConversationFolder, generateId } from '@/types/chat'

// ─── Test Wrapper ─────────────────────────────────────────────────────────────

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ChatProvider>{children}</ChatProvider>
)

// ─── Helper Functions ─────────────────────────────────────────────────────────

function createTestConversation(
  title = 'Test Conversation',
  folderId: string | null = null,
  messageCount = 0
): Conversation {
  const now = new Date().toISOString()
  const messages = []
  for (let i = 0; i < messageCount; i++) {
    messages.push({
      id: generateId(),
      role: (i % 2 === 0 ? 'user' : 'assistant') as any,
      contents: [{ type: 'text', content: `Message ${i + 1}` }],
      timestamp: now,
    })
  }
  return {
    id: generateId(),
    title,
    messages,
    folderId,
    createdAt: now,
    updatedAt: now,
  }
}

function createTestFolder(
  name = 'Test Folder',
  parentId: string | null = null,
  order = 0
): ConversationFolder {
  return {
    id: generateId(),
    name,
    parentId,
    order,
    createdAt: new Date().toISOString(),
  }
}

// ─── Mock localStorage ────────────────────────────────────────────────────────

beforeEach(() => {
  localStorage.clear()
})

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('ConversationSidebar Integration Tests', () => {
  it('renders the sidebar with header and new conversation button', () => {
    render(<ConversationSidebar />, { wrapper })

    expect(screen.getByText('Conversations')).toBeInTheDocument()
    expect(screen.getByText('+ New')).toBeInTheDocument()
  })

  it('creates a new conversation when clicking + New button', async () => {
    render(<ConversationSidebar />, { wrapper })

    const newButton = screen.getByText('+ New')
    fireEvent.click(newButton)

    // Conversation should appear in the list
    await rtlWaitFor(() => {
      expect(screen.getByText('New Conversation')).toBeInTheDocument()
    })
  })

  it('creates a new folder when clicking folder button', async () => {
    render(<ConversationSidebar />, { wrapper })

    // Click the folder button (📁)
    const folderButton = screen.getByTitle('New Folder')
    fireEvent.click(folderButton)

    // Folder should appear in the tree
    await rtlWaitFor(() => {
      expect(screen.getByText('New Folder')).toBeInTheDocument()
    })
  })

  it('shows conversations inside expanded folder', async () => {
    render(<ConversationSidebar />, { wrapper })

    // Create a folder
    const folderButton = screen.getByTitle('New Folder')
    fireEvent.click(folderButton)

    await rtlWaitFor(() => {
      expect(screen.getByText('New Folder')).toBeInTheDocument()
    })

    // Folders are auto-expanded by default, so .folder-content should exist immediately
    await rtlWaitFor(() => {
      const folderContent = document.querySelector('.folder-content')
      expect(folderContent).toBeInTheDocument()
    })
  })

  it('displays conversation count in folder badge', async () => {
    render(<ConversationSidebar />, { wrapper })

    // Create a folder
    const folderButton = screen.getByTitle('New Folder')
    fireEvent.click(folderButton)

    await rtlWaitFor(() => {
      expect(screen.getByText('New Folder')).toBeInTheDocument()
    })

    // Create a conversation in the folder
    const newButton = screen.getByText('+ New')
    fireEvent.click(newButton)

    // The folder should show conversation count
    const folderCount = document.querySelector('.folder-count')
    expect(folderCount).toBeInTheDocument()
  })

  it('shows empty state when no conversations exist', () => {
    render(<ConversationSidebar />, { wrapper })

    expect(screen.getByText('No conversations yet')).toBeInTheDocument()
    expect(screen.getByText('Start a conversation')).toBeInTheDocument()
  })

  it('hides empty state after creating conversation', async () => {
    render(<ConversationSidebar />, { wrapper })

    expect(screen.getByText('No conversations yet')).toBeInTheDocument()

    const newButton = screen.getByText('+ New')
    fireEvent.click(newButton)

    await rtlWaitFor(() => {
      expect(screen.queryByText('No conversations yet')).not.toBeInTheDocument()
    })
  })

  it('selects conversation when clicked', async () => {
    render(<ConversationSidebar />, { wrapper })

    // Create a conversation
    const newButton = screen.getByText('+ New')
    fireEvent.click(newButton)

    await rtlWaitFor(() => {
      expect(screen.getByText('New Conversation')).toBeInTheDocument()
    })

    // Click the conversation
    const convItem = screen.getByText('New Conversation')
    fireEvent.click(convItem)

    // Should be marked as active
    const activeItem = document.querySelector('.conversation-item.active')
    expect(activeItem).toBeInTheDocument()
  })

  it('shows sidebar stats with conversation and folder counts', async () => {
    render(<ConversationSidebar />, { wrapper })

    // Create a conversation
    const newButton = screen.getByText('+ New')
    fireEvent.click(newButton)

    // Create a folder
    const folderButton = screen.getByTitle('New Folder')
    fireEvent.click(folderButton)

    await rtlWaitFor(() => {
      const stats = document.querySelector('.sidebar-stats')
      expect(stats).toBeInTheDocument()
      expect(stats?.textContent).toContain('conversation')
    })
  })

  it('opens context menu on right-click', async () => {
    render(<ConversationSidebar />, { wrapper })

    // Create a conversation
    const newButton = screen.getByText('+ New')
    fireEvent.click(newButton)

    await rtlWaitFor(() => {
      expect(screen.getByText('New Conversation')).toBeInTheDocument()
    })

    // Right-click on conversation
    const convItem = screen.getByText('New Conversation')
    fireEvent.contextMenu(convItem)

    // Context menu should appear with rename and delete options
    await rtlWaitFor(() => {
      expect(screen.getByText('Rename')).toBeInTheDocument()
      expect(screen.getByText('Delete')).toBeInTheDocument()
    })
  })

  it('shows delete confirmation dialog', async () => {
    render(<ConversationSidebar />, { wrapper })

    // Create a conversation
    const newButton = screen.getByText('+ New')
    fireEvent.click(newButton)

    await rtlWaitFor(() => {
      expect(screen.getByText('New Conversation')).toBeInTheDocument()
    })

    // Right-click and select delete
    const convItem = screen.getByText('New Conversation')
    fireEvent.contextMenu(convItem)

    await rtlWaitFor(() => {
      expect(screen.getByText('Delete')).toBeInTheDocument()
    })

    const deleteButton = screen.getByText('Delete')
    fireEvent.click(deleteButton)

    // Delete confirmation dialog should appear
    await rtlWaitFor(() => {
      expect(screen.getByText(/Are you sure you want to delete/)).toBeInTheDocument()
    })
  })
})
