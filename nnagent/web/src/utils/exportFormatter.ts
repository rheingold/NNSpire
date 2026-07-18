/**
 * Export formatters for NNSpire Agent conversations.
 *
 * Supports JSON, Markdown, and printable HTML (for PDF via window.print).
 */

import { Conversation, ChatState, MessageContent } from '@/types/chat'

// ─── Internal helpers ─────────────────────────────────────────────────────────

/** Extract plain text from a MessageContent array */
function extractPlainText(contents: MessageContent[]): string {
  return contents
    .filter((c) => c.type === 'text')
    .map((c) => (c as { type: 'text'; content: string }).content)
    .join('\n')
}

/** Format a single message as Markdown */
function formatMessageMarkdown(msg: { role: string; contents: MessageContent[]; timestamp: string }): string {
  const roleLabel = msg.role === 'user' ? '**You**' : msg.role === 'assistant' ? '**Assistant**' : `**${msg.role.toUpperCase()}**`
  const time = new Date(msg.timestamp).toLocaleString()
  const text = extractPlainText(msg.contents)
  return `${roleLabel} (${time}):\n\n${text}\n`
}

// ─── JSON Export ──────────────────────────────────────────────────────────────

/**
 * Export a single conversation as JSON.
 */
export function formatAsJson(conversation: Conversation): string {
  const exportData = {
    version: '1.0',
    exportedAt: new Date().toISOString(),
    app: 'NNSpire-Agent',
    format: 'single',
    conversation: {
      id: conversation.id,
      title: conversation.title,
      model: conversation.model,
      systemPrompt: conversation.systemPrompt,
      createdAt: conversation.createdAt,
      updatedAt: conversation.updatedAt,
      messages: conversation.messages,
    },
  }
  return JSON.stringify(exportData, null, 2)
}

/**
 * Export a folder bundle (multiple conversations) as JSON.
 */
export function formatFolderAsJson(
  conversations: Conversation[],
  folderName: string,
): string {
  const exportData = {
    version: '1.0',
    exportedAt: new Date().toISOString(),
    app: 'NNSpire-Agent',
    format: 'folder',
    folder: {
      name: folderName,
      conversationCount: conversations.length,
    },
    conversations: conversations.map((c) => ({
      id: c.id,
      title: c.title,
      model: c.model,
      systemPrompt: c.systemPrompt,
      createdAt: c.createdAt,
      updatedAt: c.updatedAt,
      messages: c.messages,
    })),
  }
  return JSON.stringify(exportData, null, 2)
}

/**
 * Export full state (all conversations + folders) as JSON.
 */
export function formatAllAsJson(state: ChatState): string {
  const exportData = {
    version: '1.0',
    exportedAt: new Date().toISOString(),
    app: 'NNSpire-Agent',
    format: 'full',
    state: {
      conversations: state.conversations,
      folders: state.folders,
    },
  }
  return JSON.stringify(exportData, null, 2)
}

// ─── Markdown Export ──────────────────────────────────────────────────────────

/**
 * Export a single conversation as Markdown.
 */
export function formatAsMarkdown(conversation: Conversation): string {
  const lines: string[] = []

  // Header
  lines.push(`# ${conversation.title}`)
  lines.push('')
  if (conversation.model) {
    lines.push(`*Model: ${conversation.model}*`)
  }
  lines.push(`*Created: ${new Date(conversation.createdAt).toLocaleString()}*`)
  lines.push('')
  lines.push('---')
  lines.push('')

  // Messages
  for (const msg of conversation.messages) {
    lines.push(formatMessageMarkdown(msg))
    lines.push('')
  }

  // Footer
  lines.push('---')
  lines.push('')
  lines.push(`*Exported from NNSpire Agent on ${new Date().toLocaleString()}*`)

  return lines.join('\n')
}

/**
 * Export multiple conversations as Markdown.
 */
export function formatMultipleAsMarkdown(
  conversations: Conversation[],
  title: string = 'Conversation Bundle',
): string {
  const lines: string[] = []

  // Header
  lines.push(`# ${title}`)
  lines.push('')
  lines.push(`*${conversations.length} conversations*`)
  lines.push(`*Exported from NNSpire Agent on ${new Date().toLocaleString()}*`)
  lines.push('')
  lines.push('---')
  lines.push('')

  for (const conv of conversations) {
    lines.push(`## ${conv.title}`)
    lines.push('')
    if (conv.model) {
      lines.push(`*Model: ${conv.model}*`)
    }
    lines.push('')

    for (const msg of conv.messages) {
      lines.push(formatMessageMarkdown(msg))
      lines.push('')
    }

    lines.push('---')
    lines.push('')
  }

  return lines.join('\n')
}

// ─── HTML (Printable / PDF) Export ────────────────────────────────────────────

/**
 * Create a printable HTML document for a single conversation.
 * Callers should open this in a new window and call window.print().
 */
export function createPrintableHtml(conversation: Conversation): string {
  const messagesHtml = conversation.messages
    .map((msg) => {
      const roleClass = msg.role === 'user' ? 'user-message' : 'assistant-message'
      const text = extractPlainText(msg.contents)
      const time = new Date(msg.timestamp).toLocaleString()
      // Escape HTML to prevent XSS in printed output
      const escapedText = escapeHtml(text).replace(/\n/g, '<br/>')
      return `
        <div class="${roleClass}">
          <div class="message-header">
            <span class="role-label">${msg.role === 'user' ? 'You' : 'Assistant'}</span>
            <span class="timestamp">${time}</span>
          </div>
          <div class="message-body">${escapedText}</div>
        </div>
      `
    })
    .join('\n')

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>${escapeHtml(conversation.title)} — NNSpire Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; color: #1a1a1a; line-height: 1.6; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .meta { color: #666; font-size: 0.85rem; margin-bottom: 2rem; }
    .message { margin-bottom: 1.5rem; padding: 1rem; border-radius: 8px; }
    .user-message { background: #e8f4fd; border-left: 4px solid #0078d4; }
    .assistant-message { background: #f3f3f3; border-left: 4px solid #888; }
    .message-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.8rem; color: #555; }
    .role-label { font-weight: 600; }
    .message-body { white-space: pre-wrap; }
    @media print {
      body { padding: 0; }
      .no-print { display: none; }
    }
  </style>
</head>
<body>
  <h1>${escapeHtml(conversation.title)}</h1>
  <div class="meta">
    ${conversation.model ? `Model: ${escapeHtml(conversation.model)} &middot; ` : ''}
    Created: ${new Date(conversation.createdAt).toLocaleString()} &middot;
    ${conversation.messages.length} messages
  </div>
  ${messagesHtml}
  <div class="meta" style="margin-top: 2rem;">
    Exported from NNSpire Agent on ${new Date().toLocaleString()}
  </div>
</body>
</html>`
}

/** Escape HTML special characters */
function escapeHtml(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

// ─── Filename generation ──────────────────────────────────────────────────────

/**
 * Generate a safe filename from conversation title.
 */
export function generateExportFilename(title: string, format: 'json' | 'markdown' | 'html'): string {
  const safeTitle = title
    .replace(/[^a-zA-Z0-9\s-]/g, '')
    .replace(/\s+/g, '_')
    .substring(0, 80)
  const ext = format === 'json' ? 'json' : format === 'markdown' ? 'md' : 'html'
  const timestamp = new Date().toISOString().slice(0, 10)
  return `nnspire_${safeTitle || 'export'}_${timestamp}.${ext}`
}
