/**
 * ChatGPT JSON import converter.
 *
 * Detects ChatGPT export format and converts to NNSpire Conversation format.
 */

import { Conversation, Message, ChatGPTConversation, generateId } from '@/types/chat'

// ─── Format Detection ────────────────────────────────────────────────────────

/**
 * Detect if JSON string is in ChatGPT export format.
 * ChatGPT format has a "mapping" object with message nodes.
 */
export function detectChatGPTFormat(json: string): boolean {
  try {
    const data = JSON.parse(json)
    // ChatGPT format has a "mapping" field with message nodes
    return typeof data === 'object' && data !== null && 'mapping' in data
  } catch {
    return false
  }
}

/**
 * Detect if JSON string is in NNSpire export format.
 * NNSpire format has "version" and "conversation" or "conversations" or "state" fields.
 */
export function detectNNSpireFormat(json: string): boolean {
  try {
    const data = JSON.parse(json)
    return (
      typeof data === 'object' &&
      data !== null &&
      Boolean(data.version) &&
      ('conversation' in data || 'conversations' in data || 'state' in data)
    )
  } catch {
    return false
  }
}

// ─── Conversion ──────────────────────────────────────────────────────────────

/**
 * Convert a ChatGPT conversation JSON to NNSpire Conversation.
 *
 * ChatGPT stores messages in a "mapping" object where keys are message IDs
 * and values contain the message content.
 */
export function convertChatGPTToConversation(json: string): Conversation | null {
  try {
    const data = JSON.parse(json) as ChatGPTConversation

    if (!data.mapping) {
      console.error('[chatGptImporter] Invalid ChatGPT format: missing "mapping" field')
      return null
    }

    // Extract messages from mapping - filter for actual messages (not metadata nodes)
    const messages: Message[] = []
    const mappingEntries = Object.values(data.mapping)

    for (const entry of mappingEntries) {
      if (!entry.message?.content?.parts) continue

      // The ChatGPT mapping type has message.metadata and message.content, but role
      // is stored as a top-level property on the mapping entry in some formats.
      // Cast the entry to access role, or infer from content structure.
      const entryWithRole = entry as { message?: { role?: string; metadata?: Record<string, unknown>; content: { parts: string[] } } }
      const role = (entryWithRole.message?.role || 'assistant') as 'user' | 'assistant' | 'system'
      if (!role) continue

      // Extract text content from parts
      const textParts = entry.message.content.parts.filter((part: string) => typeof part === 'string')
      const textContent = textParts.join('\n')

      if (!textContent) continue

      const timestampRaw = entry.message.metadata?.timestamp_iso
      const timestamp = (typeof timestampRaw === 'string' ? timestampRaw : new Date().toISOString()) as string

      messages.push({
        id: generateId(),
        role,
        contents: [{ type: 'text', content: textContent }],
        timestamp,
      })
    }

    // Sort messages by timestamp
    messages.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())

    const createdAt = data.creation_time
      ? new Date(data.creation_time * 1000).toISOString()
      : new Date().toISOString()

    const conversation: Conversation = {
      id: generateId(),
      title: data.title || 'Imported from ChatGPT',
      messages,
      folderId: null,
      createdAt,
      updatedAt: new Date().toISOString(),
    }

    return conversation
  } catch (error) {
    console.error('[chatGptImporter] convertChatGPTToConversation failed:', error)
    return null
  }
}

/**
 * Convert multiple ChatGPT conversations (array of JSON strings).
 */
export function convertMultipleChatGPT(jsons: string[]): Conversation[] {
  const results: Conversation[] = []

  for (let i = 0; i < jsons.length; i++) {
    const conv = convertChatGPTToConversation(jsons[i])
    if (conv) {
      results.push(conv)
    } else {
      console.warn(`[chatGptImporter] Failed to convert ChatGPT conversation #${i + 1}`)
    }
  }

  return results
}
