/**
 * Tests for chat data models and utility functions
 */

import { describe, it, expect } from 'vitest'
import {
  generateId,
  generateTitleFromMessage,
  formatTimestamp,
  formatTimestampWithTime,
  type Message,
  type Conversation,
} from './chat'

describe('generateId', () => {
  it('generates unique IDs', () => {
    const id1 = generateId()
    const id2 = generateId()
    expect(id1).not.toBe(id2)
  })

  it('generates IDs with correct prefix', () => {
    const id = generateId()
    expect(id).toMatch(/^msg-\d{13}-[a-z0-9]{7}$/)
  })
})

describe('generateTitleFromMessage', () => {
  it('returns short messages unchanged', () => {
    const title = generateTitleFromMessage('Hello world')
    expect(title).toBe('Hello world')
  })

  it('truncates long messages with ellipsis', () => {
    const longMessage = 'A'.repeat(100)
    const title = generateTitleFromMessage(longMessage)
    expect(title.length).toBe(50)
    expect(title).toMatch(/\.{3}$/)
  })

  it('trims whitespace before processing', () => {
    const title = generateTitleFromMessage('  Hello world  ')
    expect(title).toBe('Hello world')
  })

  it('handles exactly 50 character messages', () => {
    const exactMessage = 'A'.repeat(50)
    const title = generateTitleFromMessage(exactMessage)
    expect(title).toBe(exactMessage)
  })
})

describe('formatTimestamp', () => {
  it('formats recent timestamps as "Just now"', () => {
    const now = new Date().toISOString()
    expect(formatTimestamp(now)).toBe('Just now')
  })

  it('formats minutes ago', () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60000).toISOString()
    expect(formatTimestamp(fiveMinAgo)).toBe('5m ago')
  })

  it('formats hours ago', () => {
    const threeHoursAgo = new Date(Date.now() - 3 * 3600000).toISOString()
    expect(formatTimestamp(threeHoursAgo)).toBe('3h ago')
  })

  it('formats days ago', () => {
    const threeDaysAgo = new Date(Date.now() - 3 * 86400000).toISOString()
    expect(formatTimestamp(threeDaysAgo)).toBe('3d ago')
  })

  it('formats old dates as locale date string', () => {
    const oldDate = new Date('2023-01-15').toISOString()
    const result = formatTimestamp(oldDate)
    expect(result).not.toBe('Just now')
    expect(result).not.toContain('ago')
  })
})

describe('formatTimestampWithTime', () => {
  it('formats with month, day, hour, minute', () => {
    const now = new Date().toISOString()
    const result = formatTimestampWithTime(now)
    // Format is like "Jul 18, 12:44 PM" - includes AM/PM suffix
    expect(result).toMatch(/[A-Z][a-z]{2} \d{1,2}, \d{1,2}:\d{2} [AP]M/)
  })
})

describe('Message type', () => {
  it('creates a valid message object', () => {
    const message: Message = {
      id: generateId(),
      role: 'user',
      contents: [{ type: 'text', content: 'Hello' }],
      timestamp: new Date().toISOString(),
    }
    expect(message.role).toBe('user')
    expect(message.contents).toHaveLength(1)
    expect(message.contents[0].type).toBe('text')
  })
})

describe('Conversation type', () => {
  it('creates a valid conversation object', () => {
    const now = new Date().toISOString()
    const conversation: Conversation = {
      id: generateId(),
      title: 'Test Conversation',
      messages: [],
      folderId: null,
      createdAt: now,
      updatedAt: now,
    }
    expect(conversation.messages).toHaveLength(0)
    expect(conversation.folderId).toBeNull()
  })
})
