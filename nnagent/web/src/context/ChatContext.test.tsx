/**
 * Tests for ChatContext and chat reducer
 */

import { describe, it, expect } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { ChatProvider, useChat } from './ChatContext'
import { ChatState, ChatAction } from '@/types/chat'
import React from 'react'

// Wrapper for testing hooks with context
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ChatProvider>{children}</ChatProvider>
)

describe('ChatContext', () => {
  it('provides initial state', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    expect(result.current.state.conversations).toEqual([])
    expect(result.current.state.folders).toEqual([])
    expect(result.current.state.activeConversationId).toBeNull()
    expect(result.current.state.isLoading).toBe(false)
    expect(result.current.state.error).toBeNull()
  })

  it('creates a new conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    act(() => {
      result.current.createConversation()
    })

    expect(result.current.state.conversations).toHaveLength(1)
    expect(result.current.state.conversations[0].title).toBe('New Conversation')
    expect(result.current.state.activeConversationId).toBe(result.current.state.conversations[0].id)
  })

  it('adds a message to a conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    let convId: string

    act(() => {
      const conv = result.current.createConversation()
      convId = conv.id
    })

    act(() => {
      result.current.addMessage(convId, 'user', [{ type: 'text', content: 'Hello' }])
    })

    const conversation = result.current.state.conversations.find((c) => c.id === convId)
    expect(conversation).toBeDefined()
    expect(conversation!.messages).toHaveLength(1)
    expect(conversation!.messages[0].role).toBe('user')
    // Title should be auto-generated from first message
    expect(conversation!.title).toBe('Hello')
  })

  it('deletes a conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    let convId: string

    act(() => {
      const conv = result.current.createConversation()
      convId = conv.id
    })

    act(() => {
      result.current.deleteConversation(convId)
    })

    expect(result.current.state.conversations).toHaveLength(0)
    expect(result.current.state.activeConversationId).toBeNull()
  })

  it('renames a conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    let convId: string

    act(() => {
      const conv = result.current.createConversation()
      convId = conv.id
    })

    act(() => {
      result.current.renameConversation(convId, 'My New Title')
    })

    const conversation = result.current.state.conversations.find((c) => c.id === convId)
    expect(conversation!.title).toBe('My New Title')
  })

  it('sets active conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    let conv1Id: string
    let conv2Id: string

    act(() => {
      const conv1 = result.current.createConversation()
      conv1Id = conv1.id
      const conv2 = result.current.createConversation()
      conv2Id = conv2.id
    })

    act(() => {
      result.current.setActiveConversation(conv1Id)
    })

    expect(result.current.state.activeConversationId).toBe(conv1Id)
  })

  it('creates a folder', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    act(() => {
      result.current.createFolder('My Folder')
    })

    expect(result.current.state.folders).toHaveLength(1)
    expect(result.current.state.folders[0].name).toBe('My Folder')
  })

  it('handles adding message to non-existent conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    act(() => {
      result.current.addMessage('non-existent-id', 'user', [{ type: 'text', content: 'Hello' }])
    })

    expect(result.current.state.error).toContain('non-existent-id')
    expect(result.current.state.conversations).toHaveLength(0)
  })

  it('handles deleting non-existent conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    act(() => {
      result.current.deleteConversation('non-existent-id')
    })

    // Should not throw, just log a warning
    expect(result.current.state.conversations).toHaveLength(0)
  })

  it('exports conversation as JSON', async () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    let convId: string

    act(() => {
      const conv = result.current.createConversation()
      convId = conv.id
      result.current.addMessage(convId, 'user', [{ type: 'text', content: 'Test message' }])
    })

    // exportConversation now triggers download (void return), test formatter directly
    const conversation = result.current.state.conversations.find((c) => c.id === convId)
    expect(conversation).toBeDefined()

    // Import formatter to test export format directly
    const { formatAsJson } = await import('@/utils/exportFormatter')
    const exportedJson = formatAsJson(conversation!)
    const parsed = JSON.parse(exportedJson)
    expect(parsed.version).toBe('1.0')
    expect(parsed.conversation.id).toBe(convId)
    expect(parsed.conversation.messages).toHaveLength(1)
  })

  it('imports conversation from JSON', async () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    const initialCount = result.current.state.conversations.length
    const importData = {
      version: '1.0',
      conversation: {
        id: 'imported-123',
        title: 'Imported Chat',
        messages: [
          {
            id: 'msg-1',
            role: 'user',
            contents: [{ type: 'text', content: 'Hello from import' }],
            timestamp: new Date().toISOString(),
          },
        ],
      },
    }

    let imported: any = null
    await act(async () => {
      imported = await result.current.importConversation(JSON.stringify(importData))
    })

    expect(result.current.state.conversations.length).toBe(initialCount + 1)
    expect(imported.title).toBe('Imported Chat')
    // ID should be regenerated to avoid conflicts
    expect(imported.id).not.toBe('imported-123')
  })

  it('rejects invalid import JSON', async () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    let importResult: any = null
    await act(async () => {
      importResult = await result.current.importConversation('not valid json')
      expect(importResult).toBeNull()
    })

    expect(result.current.state.error).toBeDefined()
  })
})
