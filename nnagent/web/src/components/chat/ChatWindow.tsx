/**
 * ChatWindow — Main chat display area containing messages and input.
 *
 * Features:
 * - Scrollable message area
 * - Auto-scroll to latest message
 * - Loading indicator
 * - Empty state when no conversation is active
 * - Mock AI response for testing (no backend required)
 */

import React, { useEffect, useRef, useCallback } from 'react'
import { useChat } from '@/context/ChatContext'
import { Message } from '@/types/chat'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'

// ─── Mock AI Response Generator ───────────────────────────────────────────────

/**
 * Generates a mock AI response for testing purposes.
 * In production, this will be replaced with actual provider integration.
 */
function generateMockResponse(userMessage: string): Promise<string> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const responses = [
        `I received your message: "${userMessage.substring(0, 50)}${userMessage.length > 50 ? '...' : ''}"\n\nThis is a mock response from the AI. In production, this will be replaced with actual LLM responses.`,
        `Here's my response to: "${userMessage}"\n\nI can help you with:\n- Neural network design\n- Model training configuration\n- Data analysis\n- Code generation\n\nWhat would you like to explore?`,
        `Great question! Let me think about "${userMessage}"...\n\nBased on my analysis, I would recommend considering the following approaches:\n\n1. **First approach**: Start with a simple model\n2. **Second approach**: Use transfer learning\n3. **Third approach**: Fine-tune on your specific data\n\nWould you like me to elaborate on any of these?`,
      ]
      resolve(responses[Math.floor(Math.random() * responses.length)])
    }, 500 + Math.random() * 1000) // Simulate network delay
  })
}

// ─── Loading Indicator ────────────────────────────────────────────────────────

const LoadingIndicator: React.FC = () => (
  <div className="loading-indicator">
    <div className="loading-dots">
      <span>.</span><span>.</span><span>.</span>
    </div>
    <span>AI is thinking</span>
  </div>
)

// ─── Empty State ──────────────────────────────────────────────────────────────

const EmptyState: React.FC = () => (
  <div className="chat-empty-state">
    <div className="empty-state-icon">💬</div>
    <h3>No Conversation Selected</h3>
    <p>Select a conversation from the sidebar or start a new one to begin chatting.</p>
  </div>
)

// ─── Main Component ───────────────────────────────────────────────────────────

// Available models for selection (mock list for now)
const availableModels = [
  'mock-model',
  'gpt-4',
  'gpt-3.5-turbo',
  'claude-3-opus',
  'claude-3-sonnet',
  'gemini-pro',
  'llama-3-70b',
]

// Icon presets for theming
const iconPresets = ['👤', '🧑', '👩', '👨', '🦊', '🐱', '🐶', '🦁', '🐸', '👾', '🤠', '🥸', '🧙', '🧚', '🧛', '🧜']
const aiIconPresets = ['🤖', '🧠', '💡', '🔮', '🌟', '✨', '🦉', '🐙', '🐲', '👾', '🎭', '🎨', '🔬', '⚡', '🌈', '🚀']

export const ChatWindow: React.FC = () => {
  const { state, addMessage, setModel, setTheme } = useChat()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [isResponding, setIsResponding] = React.useState(false)
  const [showSettings, setShowSettings] = React.useState(false)

  const activeConversation = state.conversations.find(
    (c) => c.id === state.activeConversationId
  )

  const theme = state.theme

  // Auto-scroll to bottom when messages change
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [activeConversation?.messages, scrollToBottom])

  const handleSend = async (userMessage: string) => {
    if (!activeConversation || isResponding) return

    try {
      // Add user message
      addMessage(
        activeConversation.id,
        'user',
        [{ type: 'text', content: userMessage }]
      )

      // Show loading and generate mock response
      setIsResponding(true)
      const aiResponse = await generateMockResponse(userMessage)

      // Add AI response
      addMessage(
        activeConversation.id,
        'assistant',
        [{ type: 'text', content: aiResponse }],
        activeConversation.model || 'mock-model'
      )
    } catch (error) {
      console.error('[ChatWindow] Failed to send message:', error)
      // Add error message to chat
      addMessage(
        activeConversation.id,
        'assistant',
        [
          {
            type: 'error',
            message: 'Failed to get response',
            details: error instanceof Error ? error.message : String(error),
            code: 'SEND_FAILED',
          },
        ]
      )
    } finally {
      setIsResponding(false)
    }
  }

  if (!activeConversation) {
    return (
      <div className="chat-window">
        <EmptyState />
      </div>
    )
  }

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h2>{activeConversation.title}</h2>
        <div className="chat-header-controls">
          {activeConversation.model && (
            <span className="chat-model-badge" title="Current model">{activeConversation.model}</span>
          )}
          <select
            className="chat-model-selector"
            value={activeConversation.model || 'mock-model'}
            onChange={(e) => setModel(activeConversation.id, e.target.value)}
            title="Switch model"
            aria-label="Select AI model"
          >
            {availableModels.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <button
            className="chat-settings-btn"
            onClick={() => setShowSettings(!showSettings)}
            title="Chat settings"
            aria-label="Toggle chat settings"
          >
            ⚙️
          </button>
        </div>
      </div>

      {showSettings && (
        <div className="chat-settings-panel">
          <h4>Customize Icons</h4>
          <div className="settings-row">
            <label>User Icon:</label>
            <div className="icon-palette">
              {iconPresets.map((icon) => (
                <button
                  key={icon}
                  className={`icon-btn ${theme.userIcon === icon ? 'active' : ''}`}
                  onClick={() => setTheme({ userIcon: icon })}
                  title={icon}
                >
                  {icon}
                </button>
              ))}
            </div>
          </div>
          <div className="settings-row">
            <label>AI Icon:</label>
            <div className="icon-palette">
              {aiIconPresets.map((icon) => (
                <button
                  key={icon}
                  className={`icon-btn ${theme.aiIcon === icon ? 'active' : ''}`}
                  onClick={() => setTheme({ aiIcon: icon })}
                  title={icon}
                >
                  {icon}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="chat-messages" data-testid="chat-messages">
        {activeConversation.messages.map((message: Message) => (
          <ChatMessage key={message.id} message={message} userIcon={theme.userIcon} aiIcon={theme.aiIcon} />
        ))}
        {isResponding && <LoadingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      <ChatInput
        onSend={handleSend}
        disabled={isResponding}
        placeholder={isResponding ? 'AI is responding...' : 'Type your message...'}
      />
    </div>
  )
}

export default ChatWindow
