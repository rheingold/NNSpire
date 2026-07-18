/**
 * ChatMessage — Renders a single chat message with proper visual distinction.
 *
 * Supports multiple content types: text, code blocks, thinking blocks,
 * MCP calls, and error messages. Includes timestamps and collapsible sections.
 */

import React, { useState } from 'react'
import { Message, MessageContent, formatTimestampWithTime } from '@/types/chat'
import { CodeBlock } from './CodeBlock'

interface ChatMessageProps {
  message: Message
  showRawJson?: boolean
  userIcon?: string
  aiIcon?: string
}

// ─── Content Renderers ────────────────────────────────────────────────────────

const TextContentRenderer: React.FC<{ content: string }> = ({ content }) => {
  // Render markdown-like content with basic formatting
  const lines = content.split('\n')
  return (
    <div className="message-text">
      {lines.map((line, i) => (
        <p key={i} className="message-text-line">
          {line}
        </p>
      ))}
    </div>
  )
}

const ThinkingContentRenderer: React.FC<{ content: string }> = ({ content }) => {
  const [collapsed, setCollapsed] = useState(true)

  return (
    <div className="message-thinking">
      <button
        className="thinking-toggle"
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        aria-label="Toggle thinking block"
      >
        {collapsed ? '▶' : '▼'} Thinking ({content.length} chars)
      </button>
      {!collapsed && (
        <div className="thinking-content">
          <pre>{content}</pre>
        </div>
      )}
    </div>
  )
}

const McpCallContentRenderer: React.FC<{
  toolName: string
  arguments: Record<string, unknown>
  result?: string
  success: boolean
}> = ({ toolName, arguments: args, result, success }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className={`message-mcp-call ${success ? 'success' : 'error'}`}>
      <button
        className="mcp-call-toggle"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-label="Toggle MCP call details"
      >
        {expanded ? '▼' : '▶'} MCP Call: {toolName} {success ? '✓' : '✗'}
      </button>
      {expanded && (
        <div className="mcp-call-details">
          <div className="mcp-call-section">
            <strong>Arguments:</strong>
            <pre>{JSON.stringify(args, null, 2)}</pre>
          </div>
          {result && (
            <div className="mcp-call-section">
              <strong>Result:</strong>
              <pre>{result}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const ErrorContentRenderer: React.FC<{
  message: string
  details?: string
  code?: string
}> = ({ message: errorMsg, details, code }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="message-error">
      <div className="error-header">
        <span className="error-icon">⚠️</span>
        <span className="error-message">{errorMsg}</span>
        {code && <span className="error-code">[{code}]</span>}
      </div>
      {details && (
        <button
          className="error-details-toggle"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
        >
          {expanded ? 'Hide Details' : 'Show Details'}
        </button>
      )}
      {expanded && details && (
        <div className="error-details">
          <pre>{details}</pre>
        </div>
      )}
    </div>
  )
}

// ─── Content Type Router ──────────────────────────────────────────────────────

const ContentRenderer: React.FC<{ content: MessageContent }> = ({ content }) => {
  switch (content.type) {
    case 'text':
      return <TextContentRenderer content={content.content} />

    case 'thinking':
      return <ThinkingContentRenderer content={content.content} />

    case 'code':
      return <CodeBlock language={content.language} content={content.content} />

    case 'mcp_call':
      return (
        <McpCallContentRenderer
          toolName={content.toolName}
          arguments={content.arguments}
          result={content.result}
          success={content.success}
        />
      )

    case 'error':
      return (
        <ErrorContentRenderer
          message={content.message}
          details={content.details}
          code={content.code}
        />
      )

    default:
      return (
        <div className="message-text">
          <p>Unknown content type: {String((content as Record<string, unknown>).type)}</p>
        </div>
      )
  }
}

// ─── Role Icon ────────────────────────────────────────────────────────────────

const RoleIcon: React.FC<{ role: string; userIcon?: string; aiIcon?: string }> = ({ role, userIcon, aiIcon }) => {
  switch (role) {
    case 'user':
      return <span className="message-icon user-icon">{userIcon || '👤'}</span>
    case 'assistant':
      return <span className="message-icon assistant-icon">{aiIcon || '🤖'}</span>
    case 'system':
      return <span className="message-icon system-icon">⚙️</span>
    case 'tool':
      return <span className="message-icon tool-icon">🔧</span>
    default:
      return <span className="message-icon">💬</span>
  }
}

// ─── Main Component ───────────────────────────────────────────────────────────

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, showRawJson, userIcon, aiIcon }) => {
  const [showJson, setShowJson] = useState(showRawJson ?? false)

  return (
    <div
      className={`chat-message chat-message-${message.role}`}
      data-message-id={message.id}
    >
      <div className="message-header">
        <RoleIcon role={message.role} userIcon={userIcon} aiIcon={aiIcon} />
        <span className="message-role">
          {message.role.charAt(0).toUpperCase() + message.role.slice(1)}
        </span>
        {message.model && (
          <span className="message-model">{message.model}</span>
        )}
        <span className="message-timestamp">
          {formatTimestampWithTime(message.timestamp)}
        </span>
      </div>

      <div className="message-body">
        {message.contents.map((content, idx) => (
          <ContentRenderer key={idx} content={content} />
        ))}
      </div>

      {message.rawJson && (
        <div className="message-json-toggle">
          <button onClick={() => setShowJson(!showJson)}>
            {showJson ? 'Hide' : 'Show'} Raw JSON
          </button>
          {showJson && (
            <pre className="message-json-content">{message.rawJson}</pre>
          )}
        </div>
      )}
    </div>
  )
}

export default ChatMessage
