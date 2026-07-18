/**
 * Tests for export formatters
 */

import { describe, it, expect } from 'vitest'
import {
  formatAsJson,
  formatFolderAsJson,
  formatAllAsJson,
  formatAsMarkdown,
  formatMultipleAsMarkdown,
  generateExportFilename,
} from './exportFormatter'
import { MessageContent } from '@/types/chat'

function createTestMessage(content: string, role: 'user' | 'assistant' = 'user'): any {
  return {
    id: `msg-${Date.now()}`,
    role,
    contents: [{ type: 'text', content }] as MessageContent[],
    timestamp: new Date().toISOString(),
  }
}

function createTestConversation(title: string): any {
  return {
    id: `conv-${Date.now()}`,
    title,
    messages: [
      createTestMessage('Hello', 'user'),
      createTestMessage('Hi there!', 'assistant'),
    ],
    folderId: null,
    model: 'test-model',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  }
}

describe('formatAsJson', () => {
  it('formats conversation as valid JSON', () => {
    const conversation = createTestConversation('Test Chat')
    const json = formatAsJson(conversation)

    const parsed = JSON.parse(json)
    expect(parsed.version).toBe('1.0')
    expect(parsed.app).toBe('NNSpire-Agent')
    expect(parsed.format).toBe('single')
    expect(parsed.conversation.id).toBe(conversation.id)
    expect(parsed.conversation.title).toBe('Test Chat')
    expect(parsed.conversation.messages).toHaveLength(2)
  })

  it('includes export metadata', () => {
    const conversation = createTestConversation('Test Chat')
    const json = formatAsJson(conversation)

    const parsed = JSON.parse(json)
    expect(parsed.exportedAt).toBeDefined()
    expect(new Date(parsed.exportedAt)).toBeInstanceOf(Date)
  })
})

describe('formatFolderAsJson', () => {
  it('formats multiple conversations as bundle', () => {
    const conversations = [
      createTestConversation('Chat 1'),
      createTestConversation('Chat 2'),
    ]
    const json = formatFolderAsJson(conversations, 'My Folder')

    const parsed = JSON.parse(json)
    expect(parsed.format).toBe('folder')
    expect(parsed.folder.name).toBe('My Folder')
    expect(parsed.conversations).toHaveLength(2)
  })
})

describe('formatAllAsJson', () => {
  it('formats full state with conversations and folders', () => {
    const state = {
      conversations: [createTestConversation('Chat 1')],
      folders: [{ id: 'folder-1', name: 'Work', parentId: null, order: 0 }],
      activeConversationId: null,
      isLoading: false,
      error: null,
    }
    const json = formatAllAsJson(state)

    const parsed = JSON.parse(json)
    expect(parsed.format).toBe('full')
    expect(parsed.state.conversations).toHaveLength(1)
    expect(parsed.state.folders).toHaveLength(1)
  })
})

describe('formatAsMarkdown', () => {
  it('generates markdown with title and messages', () => {
    const conversation = createTestConversation('Test Chat')
    const markdown = formatAsMarkdown(conversation)

    expect(markdown).toContain('# Test Chat')
    expect(markdown).toContain('Hello')
    expect(markdown).toContain('Hi there!')
    expect(markdown).toContain('**You**')
    expect(markdown).toContain('**Assistant**')
  })
})

describe('formatMultipleAsMarkdown', () => {
  it('combines multiple conversations', () => {
    const conversations = [
      createTestConversation('Chat 1'),
      createTestConversation('Chat 2'),
    ]
    const markdown = formatMultipleAsMarkdown(conversations, 'Export Bundle')

    expect(markdown).toContain('# Export Bundle')
    expect(markdown).toContain('## Chat 1')
    expect(markdown).toContain('## Chat 2')
  })
})

describe('generateExportFilename', () => {
  it('generates correct filename for json', () => {
    const filename = generateExportFilename('My Chat', 'json')
    expect(filename).toContain('My_Chat')
    expect(filename).toMatch(/\.json$/)
  })

  it('generates correct filename for markdown', () => {
    const filename = generateExportFilename('My Chat', 'markdown')
    expect(filename).toContain('My_Chat')
    expect(filename).toMatch(/\.md$/)
  })

  it('generates correct filename for html', () => {
    const filename = generateExportFilename('My Chat', 'html')
    expect(filename).toContain('My_Chat')
    expect(filename).toMatch(/\.html$/)
  })
})