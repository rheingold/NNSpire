/**
 * ChatInput — Input area for sending messages in chat.
 *
 * Features:
 * - Auto-resizing textarea
 * - Send button (disabled when empty)
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 * - Character count display
 */

import React, { useState, useRef, useEffect, useCallback } from 'react'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  placeholder?: string
  maxHeight?: number
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  disabled = false,
  placeholder = 'Type your message...',
  maxHeight = 200,
}) => {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      const newHeight = Math.min(textarea.scrollHeight, maxHeight)
      textarea.style.height = `${newHeight}px`
    }
  }, [maxHeight])

  useEffect(() => {
    adjustHeight()
  }, [value, adjustHeight])

  // Focus on mount
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [])

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return

    try {
      onSend(trimmed)
      setValue('')
      // Reset height after sending
      setTimeout(() => adjustHeight(), 0)
    } catch (error) {
      console.error('[ChatInput] Failed to send message:', error)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter without Shift sends the message
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
  }

  const canSend = value.trim().length > 0 && !disabled

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          aria-label="Chat message input"
        />
        <div className="chat-input-actions">
          <span className="char-count">{value.length}</span>
          <button
            className="send-button"
            onClick={handleSend}
            disabled={!canSend}
            aria-label="Send message"
            title="Send message (Enter)"
          >
            ➤
          </button>
        </div>
      </div>
      <div className="chat-input-hint">
        Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line
      </div>
    </div>
  )
}

export default ChatInput
