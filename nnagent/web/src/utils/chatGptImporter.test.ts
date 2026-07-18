/**
 * Tests for ChatGPT format importer
 */

import { describe, it, expect } from 'vitest'
import {
  detectChatGPTFormat,
  detectNNSpireFormat,
  convertChatGPTToConversation,
} from './chatGptImporter'

// Sample ChatGPT format JSON (matching the structure expected by convertChatGPTToConversation)
const sampleChatGPTJson = JSON.stringify({
  mapping: {
    'msg-1': {
      id: 'msg-1',
      message: {
        role: 'user',
        content: { content_type: 'text', parts: ['Hello, how are you?'] },
        metadata: { timestamp_iso: '2024-01-01T10:00:00Z' },
      },
      children: ['msg-2'],
    },
    'msg-2': {
      id: 'msg-2',
      message: {
        role: 'assistant',
        content: { content_type: 'text', parts: ['I am doing well, thank you!'] },
        metadata: { timestamp_iso: '2024-01-01T10:00:01Z' },
      },
      children: [],
    },
  },
  title: 'Test ChatGPT Conversation',
  creation_time: 1704067200,
})

// Sample NNSpire format JSON
const sampleNNSpireJson = JSON.stringify({
  version: '1.0',
  app: 'NNSpire-Agent',
  conversation: {
    id: 'conv-123',
    title: 'Test Chat',
    messages: [
      {
        id: 'msg-1',
        role: 'user',
        contents: [{ type: 'text', content: 'Hello' }],
        timestamp: new Date().toISOString(),
      },
    ],
  },
})

describe('detectChatGPTFormat', () => {
  it('detects ChatGPT format correctly', () => {
    expect(detectChatGPTFormat(sampleChatGPTJson)).toBe(true)
  })

  it('returns false for NNSpire format', () => {
    expect(detectChatGPTFormat(sampleNNSpireJson)).toBe(false)
  })

  it('returns false for invalid JSON', () => {
    expect(detectChatGPTFormat('not valid json')).toBe(false)
  })

  it('returns false for JSON without mapping', () => {
    const json = JSON.stringify({ foo: 'bar' })
    expect(detectChatGPTFormat(json)).toBe(false)
  })
})

describe('detectNNSpireFormat', () => {
  it('detects NNSpire format correctly', () => {
    expect(detectNNSpireFormat(sampleNNSpireJson)).toBe(true)
  })

  it('returns false for ChatGPT format', () => {
    expect(detectNNSpireFormat(sampleChatGPTJson)).toBe(false)
  })

  it('returns false for invalid JSON', () => {
    expect(detectNNSpireFormat('not valid json')).toBe(false)
  })
})

describe('convertChatGPTToConversation', () => {
  it('converts ChatGPT format to NNSpire conversation', () => {
    const conversation = convertChatGPTToConversation(sampleChatGPTJson)

    expect(conversation).not.toBeNull()
    expect(conversation!.title).toBe('Test ChatGPT Conversation')
    expect(conversation!.messages).toHaveLength(2)
    expect(conversation!.messages[0].role).toBe('user')
    expect(conversation!.messages[1].role).toBe('assistant')
  })

  it('generates new ID for imported conversation', () => {
    const conversation = convertChatGPTToConversation(sampleChatGPTJson)

    expect(conversation!.id).not.toBe('msg-1')
    // generateId uses 'msg-' prefix for all IDs
    expect(conversation!.id).toMatch(/^msg-/)
  })

  it('returns null for invalid ChatGPT format', () => {
    const result = convertChatGPTToConversation('not valid json')
    expect(result).toBeNull()
  })

  it('returns null for NNSpire format', () => {
    const result = convertChatGPTToConversation(sampleNNSpireJson)
    expect(result).toBeNull()
  })

  it('handles missing title with default', () => {
    const jsonNoTitle = JSON.stringify({
      mapping: {
        'msg-1': {
          id: 'msg-1',
          message: {
            role: 'user',
            content: { content_type: 'text', parts: ['Hello'] },
            metadata: { timestamp_iso: '2024-01-01T10:00:00Z' },
          },
        },
      },
    })
    const conversation = convertChatGPTToConversation(jsonNoTitle)
    expect(conversation!.title).toBe('Imported from ChatGPT')
  })
})