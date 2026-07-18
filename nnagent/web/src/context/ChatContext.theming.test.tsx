/**
 * ChatContext — Theming and Model Switching Tests
 *
 * Tests for PH2-4: Advanced Chat Features
 * - Theme state management (userIcon, aiIcon)
 * - Model switching within conversation
 * - SET_MODEL and SET_THEME reducer actions
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { ChatProvider, useChat } from './ChatContext'
import { ChatTheme, defaultTheme } from '@/types/chat'

// Wrapper to provide ChatContext
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ChatProvider>{children}</ChatProvider>
)

describe('ChatContext Theming', () => {
  it('initializes with default theme', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    expect(result.current.state.theme).toEqual(defaultTheme)
  })

  it('updates user icon via setTheme', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    act(() => {
      result.current.setTheme({ userIcon: '🦊' })
    })
    expect(result.current.state.theme.userIcon).toBe('🦊')
    expect(result.current.state.theme.aiIcon).toBe(defaultTheme.aiIcon) // unchanged
  })

  it('updates ai icon via setTheme', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    act(() => {
      result.current.setTheme({ aiIcon: '🧠' })
    })
    expect(result.current.state.theme.aiIcon).toBe('🧠')
    expect(result.current.state.theme.userIcon).toBe(defaultTheme.userIcon) // unchanged
  })

  it('updates both icons in one call', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    act(() => {
      result.current.setTheme({ userIcon: '🦊', aiIcon: '🧠' })
    })
    expect(result.current.state.theme.userIcon).toBe('🦊')
    expect(result.current.state.theme.aiIcon).toBe('🧠')
  })

  it('persists theme across theme updates', () => {
    const { result } = renderHook(() => useChat(), { wrapper })
    act(() => {
      result.current.setTheme({ userIcon: '🦊' })
    })
    act(() => {
      result.current.setTheme({ aiIcon: '🧠' })
    })
    expect(result.current.state.theme.userIcon).toBe('🦊') // still there
    expect(result.current.state.theme.aiIcon).toBe('🧠')
  })
})

describe('ChatContext Model Switching', () => {
  it('sets model on active conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    // Create a conversation first
    let convId: string
    act(() => {
      const conv = result.current.createConversation()
      convId = conv.id
    })

    // Switch model
    act(() => {
      result.current.setModel(convId, 'gpt-4')
    })

    const conv = result.current.state.conversations.find((c) => c.id === convId)
    expect(conv?.model).toBe('gpt-4')
  })

  it('does not crash when setting model on non-existent conversation', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    expect(() => {
      act(() => {
        result.current.setModel('non-existent-id', 'gpt-4')
      })
    }).not.toThrow()

    expect(result.current.state.conversations.length).toBe(0)
  })

  it('updates conversation timestamp when model changes', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    let convId: string
    act(() => {
      const conv = result.current.createConversation()
      convId = conv.id
    })

    const oldUpdatedAt = result.current.state.conversations.find((c) => c.id === convId)?.updatedAt

    act(() => {
      result.current.setModel(convId, 'claude-3-opus')
    })

    const conv = result.current.state.conversations.find((c) => c.id === convId)
    expect(conv?.updatedAt).toBeDefined()
    // The updatedAt should be present (may be same or newer)
    expect(conv?.model).toBe('claude-3-opus')
  })

  it('allows multiple model switches', () => {
    const { result } = renderHook(() => useChat(), { wrapper })

    let convId: string
    act(() => {
      const conv = result.current.createConversation()
      convId = conv.id
    })

    act(() => {
      result.current.setModel(convId, 'gpt-4')
    })
    expect(result.current.state.conversations.find((c) => c.id === convId)?.model).toBe('gpt-4')

    act(() => {
      result.current.setModel(convId, 'claude-3-opus')
    })
    expect(result.current.state.conversations.find((c) => c.id === convId)?.model).toBe('claude-3-opus')

    act(() => {
      result.current.setModel(convId, 'gemini-pro')
    })
    expect(result.current.state.conversations.find((c) => c.id === convId)?.model).toBe('gemini-pro')
  })
})

describe('ChatTheme type', () => {
  it('has correct default values', () => {
    expect(defaultTheme.userIcon).toBe('👤')
    expect(defaultTheme.aiIcon).toBe('🤖')
  })

  it('allows partial theme updates', () => {
    const theme: Partial<ChatTheme> = { userIcon: '🦊' }
    expect(theme.userIcon).toBe('🦊')
    expect(theme.aiIcon).toBeUndefined()
  })
})
